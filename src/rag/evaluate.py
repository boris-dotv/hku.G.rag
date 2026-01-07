import json
import os
import sys
sys.path.append('src/rag')

from pipeline import RAGPipeline, Document

def load_chunks(path: str) -> list:
    """Load and convert chunks to Document format"""
    with open(path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    # Convert to Document format
    documents = []
    for chunk in chunks:
        # Handle both old and new format
        if 'metadata' in chunk and 'source_type' in chunk['metadata']:
            # New format
            documents.append(Document(
                content=chunk['content'],
                doc_id=chunk['chunk_id'],
                metadata=chunk['metadata']
            ))
        else:
            # Old format
            documents.append(Document(
                content=chunk['content'],
                doc_id=chunk['chunk_id'],
                metadata=chunk
            ))
    return documents

def main():
    # API Keys
    QIANFAN_KEY = "bce-v3/ALTAK-dgZMQj7E5tByoRofFKlbM/e852481aaab5ebf3ffe6f2a50589e6e41646c127"
    GLM_KEY = "bfacda9a355fd3e9557063bc90aa8a1e.1zJ3JaydaKIkZ6Kq"

    # Load chunks
    print("Loading chunks...")
    naive_docs = load_chunks("data/parsed/baseline_chunks.json")
    enhanced_docs = load_chunks("data/parsed/enhanced_chunks.json")
    print(f"  Naive Parse: {len(naive_docs)} chunks")
    print(f"  Enhanced Parse: {len(enhanced_docs)} chunks")

    # Initialize 3 Systems
    print("\nInitializing pipelines...")

    # System 1: Naive Parse + Dense Only
    sys1_pipe = RAGPipeline(naive_docs, QIANFAN_KEY, GLM_KEY)

    # System 2: Enhanced Parse + Dense Only
    sys2_pipe = RAGPipeline(enhanced_docs, QIANFAN_KEY, GLM_KEY)

    # System 3: Enhanced Parse + Hybrid + Rerank
    sys3_pipe = RAGPipeline(enhanced_docs, QIANFAN_KEY, GLM_KEY)

    print("  ✓ Pipelines ready")

    # Test queries
    queries = [
        "What is the difference between machine learning and deep learning?",
        "How to calculate F1 Score from confusion matrix?",
        "What techniques can prevent overfitting in machine learning?",
        "Explain the bagging method in ensemble learning",
        "How does a decision tree make split decisions?"
    ]

    results = []

    print("\n" + "="*80)
    print("RUNNING EVALUATION - 5 Queries × 3 Systems")
    print("="*80)

    for i, q in enumerate(queries, 1):
        print(f"\n{'='*80}")
        print(f"Query {i}/5: {q}")
        print('='*80)

        # System 1: Naive Parse + Naive RAG
        print(f"  [1/3] Naive Parse + Naive RAG...")
        res1 = sys1_pipe.run(q, mode='dense_only')
        print(f"      Top scores: {res1['top_scores']}")

        # System 2: Enhanced Parse + Naive RAG
        print(f"  [2/3] Enhanced Parse + Naive RAG...")
        res2 = sys2_pipe.run(q, mode='dense_only')
        print(f"      Top scores: {res2['top_scores']}")

        # System 3: Enhanced Parse + Enhanced RAG
        print(f"  [3/3] Enhanced Parse + Enhanced RAG...")
        res3 = sys3_pipe.run(q, mode='hybrid')
        print(f"      Top scores: {res3['top_scores']}")

        results.append({
            "query": q,
            "systems": [
                {"name": "Naive Parse + Naive RAG", "result": res1},
                {"name": "Enhanced Parse + Naive RAG", "result": res2},
                {"name": "Enhanced Parse + Enhanced RAG", "result": res3}
            ]
        })

    # Save
    output = {
        "comparison": "5 queries x 3 systems (Fixed RRF Bug)",
        "systems": [
            "Naive Parse + Naive RAG (Dense Only - Cosine Similarity)",
            "Enhanced Parse + Naive RAG (Dense Only - Cosine Similarity)",
            "Enhanced Parse + Enhanced RAG (RRF + BM25 + Dense + Reranker)"
        ],
        "chunk_counts": {
            "naive_parse": len(naive_docs),
            "enhanced_parse": len(enhanced_docs)
        },
        "pipeline_config": {
            "embedding": "qwen3-embedding-4b",
            "reranker": "qwen3-reranker-8b"
        },
        "results": results
    }

    os.makedirs("data/evaluation", exist_ok=True)
    with open("data/evaluation/final_15_answers_fixed.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE!")
    print("="*80)
    print("Results saved to: data/evaluation/final_15_answers_fixed.json")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for i, res in enumerate(results, 1):
        print(f"\nQuery {i}: {res['query']}")
        for sys in res["systems"]:
            print(f"  {sys['name']:45} | Top: {sys['result']['top_scores'][0]}")

if __name__ == "__main__":
    main()
