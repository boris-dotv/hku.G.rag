"""
Generate Complete RAG Comparison

重新生成完整对比：6 queries × 3 systems
Agentic RAG 包含完整 CoT
"""

import sys
import os
import json
import time
from datetime import datetime
sys.path.append('src')

from rag.pipeline import RAGPipeline, Document
from agent import create_react_agent


def load_chunks(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        chunks = json.load(f)

    documents = []
    for chunk in chunks:
        if 'metadata' in chunk and 'source_type' in chunk['metadata']:
            documents.append(Document(
                content=chunk['content'],
                doc_id=chunk['chunk_id'],
                metadata=chunk['metadata']
            ))
        else:
            documents.append(Document(
                content=chunk['content'],
                doc_id=chunk['chunk_id'],
                metadata=chunk
            ))
    return documents


# 6 个测试查询
TEST_QUERIES = [
    {
        "id": 1,
        "category": "Table Extraction",
        "query": "What are the main differences between deep learning and machine learning in terms of hardware dependency?"
    },
    {
        "id": 2,
        "category": "Cross-Page Context",
        "query": "Explain the process of splitting a dataset into training and test sets using both serial and random methods."
    },
    {
        "id": 3,
        "category": "Overfitting Definition",
        "query": "What is overfitting in machine learning?"
    },
    {
        "id": 4,
        "category": "Overfitting Prevention",  # 第二轮
        "query": "How can it be prevented?"
    },
    {
        "id": 5,
        "category": "Multi-Hop Reasoning",
        "query": "What are the types of machine learning and which one uses labeled data?"
    },
    {
        "id": 6,
        "category": "Code + Concept",
        "query": "Show me how to use scikit-learn to split a dataset into training and test sets."
    }
]


def main():
    print("="*70)
    print("  Generate Complete RAG Comparison")
    print("  6 Queries × 3 Systems (Agentic with CoT)")
    print("="*70)
    print(f"\n  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    GLM_KEY = "bfacda9a355fd3e9557063bc90aa8a1e.1zJ3JaydaKIkZ6Kq"
    QIANFAN_KEY = "bce-v3/ALTAK-dgZMQj7E5tByoRofFKlbM/e852481aaab5ebf3ffe6f2a50589e6e41646c127"

    # 加载文档
    print("\n[1/3] Loading documents...")
    baseline_docs = load_chunks("data/parsed/baseline_chunks.json")
    enhanced_docs = load_chunks("data/parsed/enhanced_chunks.json")
    print(f"      Baseline: {len(baseline_docs)} chunks")
    print(f"      Enhanced: {len(enhanced_docs)} chunks")

    # 初始化系统
    print("\n[2/3] Initializing systems...")
    pipeline_naive = RAGPipeline(baseline_docs, QIANFAN_KEY, GLM_KEY)
    pipeline_enhanced = RAGPipeline(enhanced_docs, QIANFAN_KEY, GLM_KEY)
    agent = create_react_agent(pipeline_enhanced, GLM_KEY, enable_memory=True, enable_react=True)
    print("      All systems ready")

    # 运行对比
    print("\n[3/3] Running comparison...")
    print("      (This will take 10-15 minutes...)")

    all_results = []

    for test_case in TEST_QUERIES:
        query_id = test_case["id"]
        category = test_case["category"]
        query = test_case["query"]

        print(f"\n{'='*70}")
        print(f"  Query {query_id}: {category}")
        print(f"{'='*70}")
        print(f"  Q: {query}")

        query_result = {
            "query_id": query_id,
            "category": category,
            "query": query,
            "results": []
        }

        # System 1: Naive RAG
        print(f"\n  [1/3] Naive RAG...")
        start = time.time()
        try:
            result1 = pipeline_naive.run(query, mode="dense_only")
            result1["system"] = "Naive Parse + Naive RAG"
            result1["time"] = f"{time.time() - start:.2f}s"
            query_result["results"].append(result1)
            print(f"      Done: {len(result1['answer'])} chars")
        except Exception as e:
            print(f"      Error: {e}")
            query_result["results"].append({"system": "Naive Parse + Naive RAG", "error": str(e)})

        # System 2: Enhanced RAG
        print(f"  [2/3] Enhanced RAG...")
        start = time.time()
        try:
            result2 = pipeline_enhanced.run(query, mode="hybrid")
            result2["system"] = "Enhanced Parse + Enhanced RAG"
            result2["time"] = f"{time.time() - start:.2f}s"
            query_result["results"].append(result2)
            print(f"      Done: {len(result2['answer'])} chars")
        except Exception as e:
            print(f"      Error: {e}")
            query_result["results"].append({"system": "Enhanced Parse + Enhanced RAG", "error": str(e)})

        # System 3: Agentic RAG (with CoT)
        print(f"  [3/3] Agentic RAG...")
        start = time.time()
        try:
            result3 = agent.query(
                query,
                mode='hybrid',
                use_react=True,
                use_memory=True,  # Enable memory for query rewriting
                save_to_memory=True,  # Save to build conversation history
                verbose=True  # 获取 CoT
            )
            result3["system"] = "Enhanced Parse + Agentic RAG"
            result3["time"] = f"{time.time() - start:.2f}s (with CoT)"
            query_result["results"].append(result3)
            print(f"      Done: {len(result3['answer'])} chars, {len(result3.get('thoughts', []))} thoughts")
        except Exception as e:
            print(f"      Error: {e}")
            query_result["results"].append({"system": "Enhanced Parse + Agentic RAG", "error": str(e)})

        all_results.append(query_result)
        # Don't clear memory - we want conversation history for query rewriting!

    # 保存结果
    output = {
        "timestamp": datetime.now().isoformat(),
        "total_queries": len(TEST_QUERIES),
        "total_results": len(all_results) * 3,
        "systems": ["Naive Parse + Naive RAG", "Enhanced Parse + Enhanced RAG", "Enhanced Parse + Agentic RAG"],
        "results": all_results
    }

    # 备份旧文件
    old_backup = "data/evaluation/comprehensive_comparison_old.json"
    if os.path.exists("data/evaluation/comprehensive_comparison.json"):
        import shutil
        shutil.copy("data/evaluation/comprehensive_comparison.json", old_backup)
        print(f"\n  ✅ Old backup: {old_backup}")

    # 保存新结果
    output_path = "data/evaluation/comprehensive_comparison.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  ✅ Saved: {output_path}")
    print(f"{'='*70}")

    # 验证
    print("\n  Verification:")
    for result in all_results:
        query_id = result["query_id"]
        category = result["category"]
        print(f"\n  Query {query_id} ({category}):")
        for sys_result in result["results"]:
            system = sys_result["system"]
            if "error" in sys_result:
                print(f"    {system}: ERROR")
            else:
                answer_len = len(sys_result.get("answer", ""))
                thoughts_len = len(sys_result.get("thoughts", []))
                cot_status = "✅" if thoughts_len > 0 else "❌"
                print(f"    {system}: {answer_len} chars, CoT: {cot_status}")

    print(f"\n{'='*70}")
    print("  Complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
