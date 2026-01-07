"""
Complete RAG Pipeline with Qianfan API

Implements the full retrieval-augmented generation pipeline:
1. Query Router: Detects query type (keyword vs semantic)
2. Hybrid Retrieval: BM25 + Qwen3-Embedding-8B
3. Reranking: Qwen3-Reranker-8B
4. Answer Generation: Using GLM-4.7

Uses Baidu Qianfan API for embedding and reranking.
"""

import json
import requests
import jieba
import hashlib
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    """Document with content and metadata"""
    content: str
    doc_id: str
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ============================================
# Qianfan API Clients
# ============================================

class QianfanEmbedding:
    """Qianfan Embedding API Client (Qwen3-Embedding-8B)"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://qianfan.baidubce.com/v2/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts"""
        payload = {
            "model": "qwen3-embedding-8b",
            "input": texts
        }

        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        return [item["embedding"] for item in result["data"]]

    def embed_single(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        result = self.embed([text])
        return result[0]


class QianfanReranker:
    """Qianfan Reranker API Client (Qwen3-Reranker-8B)"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://qianfan.baidubce.com/v2/rerank"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Tuple[int, float]]:
        """
        Rerank documents based on query

        Returns:
            List of (index, score) tuples sorted by relevance
        """
        payload = {
            "model": "qwen3-reranker-8b",
            "query": query,
            "documents": documents
        }

        if top_k:
            payload["top_n"] = top_k

        response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        # Return (index, score) pairs
        return [(item["index"], item.get("relevance_score", item.get("score", 0)))
                for item in result["results"]]


# ============================================
# Retrievers
# ============================================

class BM25Retriever:
    """BM25 Sparse Retriever using Jieba tokenization"""

    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.doc_freqs = self._build_doc_freqs()
        self.avg_doc_len = sum(len(doc.content) for doc in documents) / len(documents)
        self.k1 = 1.5
        self.b = 0.75

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize Chinese text using jieba"""
        return [t for t in jieba.lcut(text) if len(t) > 1]

    def _build_doc_freqs(self) -> Dict[str, int]:
        """Build document frequency dictionary"""
        df = {}
        for doc in self.documents:
            tokens = set(self._tokenize(doc.content))
            for token in tokens:
                df[token] = df.get(token, 0) + 1
        return df

    def _score(self, query: str, doc: Document) -> float:
        """Calculate BM25 score"""
        query_tokens = self._tokenize(query)
        doc_tokens = self._tokenize(doc.content)

        doc_len = len(doc_tokens)
        token_counts = {}
        for t in doc_tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        score = 0
        for token in query_tokens:
            if token not in token_counts:
                continue

            tf = token_counts[token]
            df = self.doc_freqs.get(token, 1)
            idf = len(self.documents) / df

            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))

            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve top-k documents"""
        scores = [(doc, self._score(query, doc)) for doc in self.documents]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class DenseRetriever:
    """Dense Retriever using Qwen3-Embedding-8B"""

    def __init__(self, documents: List[Document], api_key: str):
        self.documents = documents
        self.embedding_client = QianfanEmbedding(api_key)
        self.doc_embeddings = None

    def build_index(self):
        """Build embedding index for all documents"""
        print("Building embedding index...")
        texts = [doc.content for doc in self.documents]
        self.doc_embeddings = self.embedding_client.embed(texts)
        print(f"Index built for {len(self.documents)} documents")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        """Retrieve top-k documents using semantic similarity"""
        if self.doc_embeddings is None:
            self.build_index()

        import numpy as np

        query_emb = np.array(self.embedding_client.embed_single(query))
        doc_embeddings = np.array(self.doc_embeddings)

        # Calculate cosine similarity
        similarities = np.dot(doc_embeddings, query_emb) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_emb)
        )

        # Get top-k
        top_indices = similarities.argsort()[-top_k:][::-1]

        return [(self.documents[i], float(similarities[i])) for i in top_indices]


# ============================================
# Query Router
# ============================================

class QueryRouter:
    """
    Routes queries based on type to adjust retrieval strategy

    Detection:
    - Keyword-heavy: Has technical terms, numbers, special characters
    - Conceptual: General explanatory questions
    """

    def __init__(self):
        pass

    def detect_query_type(self, query: str) -> str:
        """Detect query type: 'keyword' or 'semantic'"""
        import re

        # Check for technical terms (uppercase, numbers, special chars)
        if re.search(r'\b[A-Z]{2,}|\d+\.\d+|[A-Z]-[A-Z]|\w+-\w+', query):
            return 'keyword'

        # Check for specific ML terminology patterns
        keyword_patterns = [
            r'\b(svm|knn|cnn|rnn|lstm|api|csv|json)\b',
            r'\b(Fama-French|Qwen|BERT|GPT)\b',
            r'\d+[%]'
        ]

        for pattern in keyword_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return 'keyword'

        return 'semantic'

    def get_retrieval_weights(self, query: str) -> Tuple[float, float]:
        """
        Get weights for BM25 and Dense retrieval

        Returns:
            (bm25_weight, dense_weight)
        """
        query_type = self.detect_query_type(query)

        if query_type == 'keyword':
            return 0.7, 0.4  # BM25 dominant
        else:
            return 0.4, 0.8  # Dense dominant


# ============================================
# Complete RAG Pipeline
# ============================================

class RAGPipeline:
    """
    Complete Retrieval-Augmented Generation Pipeline

    Flow:
    1. Query Router → Detect query type
    2. Hybrid Retrieval → BM25 + Dense with adaptive weights
    3. Reranking → Qwen3-Reranker-8B
    4. Answer Generation → Using retrieved context
    """

    def __init__(
        self,
        documents: List[Document],
        qianfan_api_key: str,
        glm_api_key: str
    ):
        self.documents = documents

        # Initialize components
        self.bm25_retriever = BM25Retriever(documents)
        self.dense_retriever = DenseRetriever(documents, qianfan_api_key)
        self.reranker = QianfanReranker(qianfan_api_key)
        self.router = QueryRouter()

        # GLM for answer generation
        self.glm_api_key = glm_api_key
        self.glm_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

        # Build dense index
        self.dense_retriever.build_index()

    def hybrid_retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """Hybrid retrieval with adaptive weights"""
        bm25_weight, dense_weight = self.router.get_retrieval_weights(query)

        # Calculate actual k values
        bm25_k = int(top_k * bm25_weight) + 1
        dense_k = int(top_k * dense_weight) + 1

        # Retrieve from both
        bm25_results = self.bm25_retriever.retrieve(query, top_k=bm25_k)
        dense_results = self.dense_retriever.retrieve(query, top_k=dense_k)

        # Merge and deduplicate by doc_id
        merged = {}
        for doc, score in bm25_results:
            merged[doc.doc_id] = (doc, score * bm25_weight)

        for doc, score in dense_results:
            if doc.doc_id in merged:
                merged[doc.doc_id] = (doc, merged[doc.doc_id][1] + score * dense_weight)
            else:
                merged[doc.doc_id] = (doc, score * dense_weight)

        # Sort by combined score
        results = sorted(merged.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, score in results[:top_k]]

    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Document]:
        """Rerank documents using Qianfan API"""
        if not documents:
            return []

        doc_texts = [doc.content for doc in documents]
        rerank_results = self.reranker.rerank(query, doc_texts, top_k=top_k)

        # Return reranked documents
        return [documents[idx] for idx, score in rerank_results]

    def generate_answer(self, query: str, context_docs: List[Document]) -> str:
        """Generate answer using GLM-4.7"""
        context = "\n\n".join([
            f"[{i+1}] {doc.content}"
            for i, doc in enumerate(context_docs)
        ])

        prompt = f"""Based on the following course materials, answer the question:

{context}

Question: {query}

Provide a clear, accurate answer based on the materials above. If the answer is not in the materials, say "The answer is not available in the provided materials."
"""

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.glm_api_key}"
        }

        payload = {
            "model": "glm-4.7",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 1024
        }

        response = requests.post(self.glm_url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()

        return result["choices"][0]["message"]["content"]

    def query(self, query: str, top_k: int = 10, rerank_k: int = 5) -> Dict:
        """
        Complete RAG query

        Args:
            query: User query
            top_k: Number of documents to retrieve
            rerank_k: Number of documents to keep after reranking

        Returns:
            Dictionary with query, context, and answer
        """
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        # Step 1: Query routing
        query_type = self.router.detect_query_type(query)
        print(f"[1/4] Query Type: {query_type}")

        # Step 2: Hybrid retrieval
        print(f"[2/4] Hybrid Retrieval...")
        retrieved_docs = self.hybrid_retrieve(query, top_k=top_k)
        print(f"     Retrieved {len(retrieved_docs)} documents")

        # Step 3: Reranking
        print(f"[3/4] Reranking...")
        reranked_docs = self.rerank(query, retrieved_docs, top_k=rerank_k)
        print(f"     Top {len(reranked_docs)} after reranking")

        # Step 4: Answer generation
        print(f"[4/4] Generating answer...")
        answer = self.generate_answer(query, reranked_docs)

        return {
            "query": query,
            "query_type": query_type,
            "retrieved_docs": [doc.content for doc in retrieved_docs],
            "reranked_docs": [doc.content for doc in reranked_docs],
            "answer": answer
        }


# ============================================
# Utility Functions
# ============================================

def load_chunks_from_json(json_path: str) -> List[Document]:
    """Load chunks from parser output"""
    with open(json_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)

    documents = []
    for chunk_data in chunks_data:
        doc = Document(
            content=chunk_data["content"],
            doc_id=chunk_data["chunk_id"],
            metadata=chunk_data.get("metadata", {})
        )
        documents.append(doc)

    return documents


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    import argparse

    QIANFAN_API_KEY = "bce-v3/ALTAK-dgZMQj7E5tByoRofFKlbM/e852481aaab5ebf3ffe6f2a50589e6e41646c127"
    GLM_API_KEY = "bfacda9a355fd3e9557063bc90aa8a1e.1zJ3JaydaKIkZ6Kq"

    parser = argparse.ArgumentParser(description="Run RAG pipeline")
    parser.add_argument("--chunks", default="data/parsed/enhanced_chunks.json", help="Path to chunks JSON")
    parser.add_argument("--query", help="Query to process")

    args = parser.parse_args()

    # Load documents
    print(f"Loading documents from {args.chunks}...")
    documents = load_chunks_from_json(args.chunks)
    print(f"Loaded {len(documents)} documents")

    # Initialize pipeline
    print("\nInitializing RAG pipeline...")
    pipeline = RAGPipeline(documents, QIANFAN_API_KEY, GLM_API_KEY)
    print("Pipeline ready!\n")

    if args.query:
        # Single query mode
        result = pipeline.query(args.query)
        print(f"\n{'='*60}")
        print("ANSWER:")
        print(result["answer"])
    else:
        # Interactive mode
        print("Interactive mode (type 'quit' to exit)")
        while True:
            query = input("\nQuery> ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break

            if query:
                result = pipeline.query(query)
                print(f"\n{'='*60}")
                print("ANSWER:")
                print(result["answer"])
