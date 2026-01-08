#!/usr/bin/env python3
"""
Interactive Query Testing Module

Run with: python interactive_query.py

Features:
- Test any query against 3 RAG systems: Naive, Enhanced, Agentic
- Side-by-side comparison
- Detailed Chain of Thought for Agentic RAG
"""

import os
import json
import time
from typing import Dict, List

from src.rag.pipeline import RAGPipeline, Document
from src.agent import create_react_agent


def load_documents() -> List[Document]:
    """Load enhanced chunks for all systems"""
    print("[1/3] Loading documents...")
    with open("data/parsed/enhanced_chunks.json") as f:
        chunks = json.load(f)

    # Map chunk_id -> doc_id for Document class
    docs = []
    for c in chunks:
        docs.append(Document(
            content=c["content"],
            doc_id=c.get("chunk_id", c.get("doc_id", "")),
            metadata=c.get("metadata", {})
        ))

    print(f"      Loaded {len(docs)} enhanced chunks")
    return docs


def load_baseline_documents() -> List[Document]:
    """Load baseline chunks for Naive RAG"""
    with open("data/parsed/baseline_chunks.json") as f:
        chunks = json.load(f)

    docs = []
    for c in chunks:
        docs.append(Document(
            content=c["content"],
            doc_id=c.get("chunk_id", c.get("doc_id", "")),
            metadata=c.get("metadata", {})
        ))
    return docs


def query_naive_rag(query: str, documents: List[Document], glm_key: str) -> Dict:
    """Naive RAG: Dense retrieval only"""
    pipeline = RAGPipeline(
        documents,
        qianfan_key=os.getenv("QIANFAN_API_KEY"),
        glm_key=glm_key
    )

    start = time.time()
    result = pipeline.run(query, mode="dense_only")
    elapsed = time.time() - start

    # Build sources from retrieved_chunks
    sources = []
    for chunk in result.get("retrieved_chunks", [])[:3]:
        sources.append({
            "content": chunk.get("content", ""),
            "score": 0.0,  # dense_only mode doesn't have scores in run()
            "source_type": chunk.get("source_type", "unknown")
        })

    return {
        "system": "Naive RAG",
        "answer": result.get("answer", ""),
        "sources": sources,
        "time": elapsed
    }


def query_enhanced_rag(query: str, documents: List[Document], glm_key: str) -> Dict:
    """Enhanced RAG: Hybrid + Reranker"""
    pipeline = RAGPipeline(
        documents,
        qianfan_key=os.getenv("QIANFAN_API_KEY"),
        glm_key=glm_key
    )

    start = time.time()
    result = pipeline.run(query, mode="hybrid")
    elapsed = time.time() - start

    # Build sources from retrieved_chunks and top_scores
    sources = []
    chunks = result.get("retrieved_chunks", [])[:3]
    scores = result.get("top_scores", [])
    for i, chunk in enumerate(chunks):
        score = float(scores[i]) if i < len(scores) else 0.0
        sources.append({
            "content": chunk.get("content", ""),
            "score": score,
            "source_type": chunk.get("source_type", "unknown")
        })

    return {
        "system": "Enhanced RAG",
        "answer": result.get("answer", ""),
        "sources": sources,
        "time": elapsed
    }


def query_agentic_rag(query: str, documents: List[Document], glm_key: str, verbose: bool = True) -> Dict:
    """Agentic RAG: ReAct Loop + Query Rewriting + Memory"""
    pipeline = RAGPipeline(
        documents,
        qianfan_key=os.getenv("QIANFAN_API_KEY"),
        glm_key=glm_key
    )

    agent = create_react_agent(
        pipeline=pipeline,
        glm_key=glm_key,
        enable_memory=True,
        enable_react=True
    )

    start = time.time()
    result = agent.query(
        query,
        mode="hybrid",
        use_react=True,
        use_memory=True,
        save_to_memory=False,
        verbose=verbose
    )
    elapsed = time.time() - start

    return {
        "system": "Agentic RAG",
        "answer": result.get("answer", ""),
        "sources": result.get("sources", [])[:3],
        "time": elapsed,
        "react_steps": result.get("react_steps", 0),
        "thoughts": result.get("thoughts", []),
        "query_was_rewritten": result.get("query_was_rewritten", False),
        "rewritten_query": result.get("rewritten_query", ""),
        "tools_used": result.get("tools_used", [])
    }


def print_result(result: Dict, show_cot: bool = False):
    """Pretty print a single result"""
    print(f"\n{'='*60}")
    print(f"📊 {result['system']} ({result['time']:.1f}s)")
    print('='*60)

    answer = result['answer']
    if len(answer) > 500:
        print(f"\n📝 Answer ({len(answer)} chars):")
        print(answer[:500] + "...")
    else:
        print(f"\n📝 Answer ({len(answer)} chars):")
        print(answer)

    if result.get('sources'):
        print(f"\n🔍 Top Sources:")
        for i, src in enumerate(result['sources'][:3], 1):
            score = src.get('score', 0)
            content = src.get('content', '')[:80]
            print(f"   {i}. [{score:.2f}] {content}...")

    # Chain of Thought for Agentic RAG
    if show_cot and result.get('thoughts'):
        print(f"\n🧠 Chain of Thought ({result.get('react_steps', 0)} steps):")
        for thought in result['thoughts']:
            print(f"   - {thought}")

    if result.get('query_was_rewritten'):
        print(f"\n🔄 Query Rewritten: \"{result['rewritten_query']}\"")

    if result.get('tools_used'):
        print(f"\n🛠️  Tools Used: {', '.join(result['tools_used'])}")


def compare_systems(query: str, documents: List[Document], glm_key: str, show_cot: bool = True):
    """Run query on all 3 systems and compare"""
    print(f"\n{'='*60}")
    print(f"🎯 Query: \"{query}\"")
    print('='*60)

    results = []

    # Naive RAG
    print("\n⚪ Running Naive RAG...")
    try:
        naive = query_naive_rag(query, load_baseline_documents(), glm_key)
        results.append(naive)
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Enhanced RAG
    print("\n🟡 Running Enhanced RAG...")
    try:
        enhanced = query_enhanced_rag(query, documents, glm_key)
        results.append(enhanced)
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Agentic RAG
    print("\n🔴 Running Agentic RAG...")
    try:
        agentic = query_agentic_rag(query, documents, glm_key, verbose=True)
        results.append(agentic)
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Print results
    print("\n" + "="*60)
    print("📊 RESULTS")
    print("="*60)

    for result in results:
        print_result(result, show_cot=show_cot)

    # Summary
    print(f"\n{'='*60}")
    print("📈 SUMMARY")
    print('='*60)
    for result in results:
        name = result['system']
        chars = len(result['answer'])
        time_sec = result['time']
        print(f"   {name}: {chars} chars, {time_sec:.1f}s")


def main():
    """Interactive loop"""
    print("""
╔════════════════════════════════════════════════════════════╗
║     Interactive Query Testing for HKU G.RAG               ║
╚════════════════════════════════════════════════════════════╝
    """)

    # Check API keys
    glm_key = os.getenv("GLM_API_KEY")
    if not glm_key:
        print("❌ GLM_API_KEY not set!")
        return

    # Load documents once
    documents = load_documents()

    print(f"\n[2/3] Systems ready: Naive | Enhanced | Agentic")

    print("""
[3/3] Commands:
  compare <query>    - Run query on all 3 systems
  naive <query>      - Run Naive RAG only
  enhanced <query>   - Run Enhanced RAG only
  agentic <query>    - Run Agentic RAG with CoT
  quit or exit       - Exit
    """)

    while True:
        try:
            user_input = input("\n➤ ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Bye!")
                break

            # Parse command
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            query = parts[1] if len(parts) > 1 else ""

            if not query:
                print("❌ Please provide a query")
                continue

            # Execute
            if cmd == 'compare':
                compare_systems(query, documents, glm_key, show_cot=True)
            elif cmd == 'naive':
                result = query_naive_rag(query, load_baseline_documents(), glm_key)
                print_result(result)
            elif cmd == 'enhanced':
                result = query_enhanced_rag(query, documents, glm_key)
                print_result(result)
            elif cmd == 'agentic':
                result = query_agentic_rag(query, documents, glm_key, verbose=True)
                print_result(result, show_cot=True)
            else:
                print(f"❌ Unknown command: {cmd}")
                print("   Available: compare, naive, enhanced, agentic, quit")

        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()
