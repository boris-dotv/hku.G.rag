"""
Final Production RAG Pipeline
Changes:
1. Added Vector Persistence (Caching)
2. Configurable Model Name (4B compatible)
3. Hybrid Retrieve supports 'dense_only' mode for fair comparison
"""

import json
import requests
import jieba
import os
import numpy as np
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

# ============================================
# Configuration
# ============================================
class RAGConfig:
    # AutoDL/Server config compatible
    EMBEDDING_MODEL = "qwen3-embedding-4b"
    RERANK_MODEL = "qwen3-reranker-8b"
    CHAT_MODEL = "glm-4"

    # Caching paths
    CACHE_DIR = "data/cache"
    VECTOR_CACHE_PATH = "data/cache/vectors.npy"
    DOC_ID_CACHE_PATH = "data/cache/doc_ids.json"

@dataclass
class Document:
    content: str
    doc_id: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

# ============================================
# API Clients
# ============================================
class QianfanClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://qianfan.baidubce.com/v2"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def embed(self, texts: List[str]) -> np.ndarray:
        BATCH_SIZE = 8  # Safe batch size
        all_embeddings = []

        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i : i + BATCH_SIZE]
            payload = {"model": RAGConfig.EMBEDDING_MODEL, "input": batch}
            try:
                res = requests.post(f"{self.base_url}/embeddings", headers=self.headers, json=payload, timeout=60)
                res.raise_for_status()
                data = res.json()
                if "data" in data:
                    all_embeddings.extend([item["embedding"] for item in data["data"]])
                else:
                    # Fallback for empty/error
                    print(f"API Warning: {data}")
                    all_embeddings.extend([np.zeros(2560 if '4b' in RAGConfig.EMBEDDING_MODEL else 4096).tolist() for _ in batch])
            except Exception as e:
                print(f"Embedding Error: {e}")
                all_embeddings.extend([np.zeros(2560).tolist() for _ in batch])

        return np.array(all_embeddings)

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        if not documents:
            return []
        payload = {"model": RAGConfig.RERANK_MODEL, "query": query, "documents": documents}
        if top_k:
            payload["top_n"] = top_k

        try:
            res = requests.post(f"{self.base_url}/rerank", headers=self.headers, json=payload, timeout=30)
            res.raise_for_status()
            results = res.json().get("results", [])
            return [(item["index"], item["relevance_score"]) for item in results]
        except Exception as e:
            print(f"Rerank Error: {e}")
            return [(i, 0.0) for i in range(len(documents))]

# ============================================
# Retrievers
# ============================================
class BM25Retriever:
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.doc_freqs = {}
        self.avg_doc_len = 0
        self.k1 = 1.5
        self.b = 0.75
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        return [t for t in jieba.lcut(text) if len(t) > 1]

    def _build_index(self):
        total_len = 0
        for doc in self.documents:
            tokens = set(self._tokenize(doc.content))
            for token in tokens:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
            total_len += len(doc.content)
        self.avg_doc_len = total_len / len(self.documents) if self.documents else 1

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        query_tokens = self._tokenize(query)
        scores = []
        N = len(self.documents)

        for doc in self.documents:
            doc_tokens = self._tokenize(doc.content)
            doc_len = len(doc_tokens)
            token_counts = defaultdict(int)
            for t in doc_tokens:
                token_counts[t] += 1

            score = 0.0
            for token in query_tokens:
                if token not in token_counts:
                    continue
                tf = token_counts[token]
                df = self.doc_freqs.get(token, 1)
                # Smoothed IDF: log(N/df + 1) to ensure positivity
                idf = np.log((N - df + 0.5) / (df + 0.5) + 1)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
            scores.append((doc, score))

        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

class DenseRetriever:
    def __init__(self, documents: List[Document], api_key: str):
        self.documents = documents
        self.client = QianfanClient(api_key)
        self.embeddings = None

    def load_or_build_index(self):
        # Implement Persistence
        os.makedirs(RAGConfig.CACHE_DIR, exist_ok=True)

        # Calculate a hash of the current documents to ensure cache validity
        doc_content_hash = hashlib.md5("".join([d.doc_id for d in self.documents]).encode()).hexdigest()
        vector_path = f"{RAGConfig.CACHE_DIR}/vectors_{doc_content_hash}.npy"

        if os.path.exists(vector_path):
            print(f"Loading cached vectors from {vector_path}...")
            self.embeddings = np.load(vector_path)
        else:
            print("Building new embedding index (calling API)...")
            texts = [d.content for d in self.documents]
            self.embeddings = self.client.embed(texts)

            # Normalize
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embeddings = self.embeddings / norms

            np.save(vector_path, self.embeddings)
            print("Index saved.")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        if self.embeddings is None:
            self.load_or_build_index()

        q_emb = np.array(self.client.embed([query])[0])
        if q_emb.size == 0:
            return []

        q_norm = np.linalg.norm(q_emb)
        if q_norm > 0:
            q_emb = q_emb / q_norm

        scores = np.dot(self.embeddings, q_emb)
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [(self.documents[i], float(scores[i])) for i in top_indices]

# ============================================
# Pipeline
# ============================================
class RAGPipeline:
    def __init__(self, documents: List[Document], qianfan_key: str, glm_key: str):
        self.documents = documents
        self.qianfan_client = QianfanClient(qianfan_key)
        self.glm_key = glm_key

        self.bm25 = BM25Retriever(documents)
        self.dense = DenseRetriever(documents, qianfan_key)
        # Auto-load index on init
        self.dense.load_or_build_index()

    def hybrid_retrieve(self, query: str, top_k: int = 10, mode: str = 'hybrid') -> List[Tuple[Document, float]]:
        """
        Supports modes:
        - 'dense_only': Returns Raw Cosine Similarity (0.0-1.0) -> FIXES "0.0167" BUG
        - 'bm25_only': Returns BM25 Score
        - 'hybrid': Returns RRF Score
        """
        if mode == 'dense_only':
            return self.dense.retrieve(query, top_k)

        if mode == 'bm25_only':
            return self.bm25.retrieve(query, top_k)

        # Hybrid Mode (RRF)
        import re
        is_keyword = bool(re.search(r'\b[A-Z]{2,}\b|\d+\.\d+', query))
        bm25_w = 0.7 if is_keyword else 0.3
        dense_w = 0.3 if is_keyword else 0.7

        candidate_k = top_k * 2
        r1 = self.bm25.retrieve(query, top_k=candidate_k)
        r2 = self.dense.retrieve(query, top_k=candidate_k)

        rrf_map = defaultdict(float)
        k_const = 60

        for rank, (doc, _) in enumerate(r1):
            rrf_map[doc.doc_id] += bm25_w * (1 / (k_const + rank))
        for rank, (doc, _) in enumerate(r2):
            rrf_map[doc.doc_id] += dense_w * (1 / (k_const + rank))

        sorted_ids = sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)
        doc_lookup = {d.doc_id: d for d in self.documents}

        return [(doc_lookup[did], score) for did, score in sorted_ids[:top_k] if did in doc_lookup]

    def generate(self, query: str, docs: List[Document]) -> str:
        MAX_CHARS = 6000 # Safety limit
        context = []
        curr_len = 0
        for i, d in enumerate(docs):
            content = d.content.strip()
            if curr_len + len(content) > MAX_CHARS:
                break
            context.append(f"[{i+1}] {content}")
            curr_len += len(content)

        prompt = f"Context:\n{chr(10).join(context)}\n\nQuestion: {query}"

        url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        headers = {"Authorization": f"Bearer {self.glm_key}"}
        payload = {
            "model": RAGConfig.CHAT_MODEL,
            "messages": [{"role": "user", "content": prompt}]
        }
        try:
            res = requests.post(url, headers=headers, json=payload).json()
            return res["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {e}"

    def run(self, query: str, mode: str = 'hybrid'):
        """
        Main entry point.
        mode: 'hybrid' (uses reranker), 'dense_only' (no reranker), 'bm25_only'
        """
        retrieved_tuples = self.hybrid_retrieve(query, top_k=15, mode=mode)

        if mode == 'hybrid':
            # Only rerank in hybrid/enhanced mode
            reranked = self.qianfan_client.rerank(query, [d.content for d, _ in retrieved_tuples], top_k=5)
            final_docs = [retrieved_tuples[i][0] for i, score in reranked]
            top_scores = [f"{score:.4f}" for _, score in reranked]
            method_name = "Enhanced RAG (RRF + Rerank)"
        else:
            # Naive mode: take top 3 directly
            final_docs = [d for d, _ in retrieved_tuples[:3]]
            top_scores = [f"{s:.4f}" for _, s in retrieved_tuples[:3]]
            method_name = f"Naive RAG ({mode})"

        answer = self.generate(query, final_docs)

        return {
            "method": method_name,
            "top_scores": top_scores,
            "retrieved_chunks": [{"chunk_id": d.doc_id, "source_type": d.metadata.get("source_type", "unknown"), "content": d.content[:200] + "..."} for d in final_docs],
            "answer": answer
        }
