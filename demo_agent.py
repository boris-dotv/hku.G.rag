"""
Agentic RAG Demo - 带记忆的 RAG 系统

演示记忆系统的使用：
1. 短期记忆：记住最近的对话
2. 长期记忆：跨会话持久化
3. 上下文整合：结合历史回答问题
"""

import sys
sys.path.append('src')

from agent import AgenticRAG
from rag.pipeline import Document
import json
import os


def load_chunks(path: str):
    """加载文档块"""
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


def main():
    print("="*70)
    print("Agentic RAG Demo - Memory System")
    print("="*70)

    # API Keys
    QIANFAN_KEY = os.environ.get("QIANFAN_KEY",
        "bce-v3/ALTAK-dgZMQj7E5tByoRofFKlbM/e852481aaab5ebf3ffe6f2a50589e6e41646c127")
    GLM_KEY = os.environ.get("GLM_KEY",
        "bfacda9a355fd3e9557063bc90aa8a1e.1zJ3JaydaKIkZ6Kq")

    # 加载文档
    print("\n[1/3] Loading documents...")
    documents = load_chunks("data/parsed/enhanced_chunks.json")
    print(f"      Loaded {len(documents)} chunks")

    # 初始化 Pipeline
    print("\n[2/3] Initializing RAG Pipeline...")
    from rag.pipeline import RAGPipeline
    pipeline = RAGPipeline(documents, QIANFAN_KEY, GLM_KEY)
    print("      Pipeline ready")

    # 创建 Agent（带记忆）
    print("\n[3/3] Creating Agentic RAG with Memory...")
    agent = AgenticRAG(
        pipeline=pipeline,
        enable_memory=True,
        short_term_size=10,
        long_term_path="data/memory/memories.json"
    )
    print("      Agent ready")

    # 显示初始记忆状态
    print("\n" + "="*70)
    print("Initial Memory Stats:")
    print("="*70)
    stats = agent.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 模拟对话
    print("\n" + "="*70)
    print("Starting Conversation (with Memory)")
    print("="*70)

    conversations = [
        {
            "query": "What is machine learning?",
            "importance": 0.8  # 高重要性，存入长期记忆
        },
        {
            "query": "How does it differ from traditional programming?",
            "importance": 0.7  # 高重要性
        },
        {
            "query": "What are the main types of machine learning?",
            "importance": 0.7
        },
        {
            # 这个问题应该能从记忆中找到上下文
            "query": "Can you explain more about the first type you mentioned?",
            "importance": 0.6
        }
    ]

    for i, conv in enumerate(conversations, 1):
        query = conv["query"]
        importance = conv["importance"]

        print(f"\n{'─'*70}")
        print(f"[Round {i}] User: {query}")
        print(f"{'─'*70}")

        result = agent.query(
            query,
            mode="hybrid",
            use_memory=True,
            save_to_memory=True,
            importance=importance
        )

        # 显示答案
        answer = result["answer"]
        if len(answer) > 300:
            answer = answer[:300] + "..."
        print(f"\nAgent: {answer}")

        # 显示记忆使用情况
        memory_used = result["memory_used"]
        if memory_used["short_term"] > 0 or memory_used["long_term"] > 0:
            print(f"\n📝 Memory Used:")
            print(f"   Short-term: {memory_used['short_term']} recent turns")
            print(f"   Long-term: {memory_used['long_term']} relevant memories")

    # 显示最终记忆状态
    print("\n" + "="*70)
    print("Final Memory Stats:")
    print("="*70)
    stats = agent.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # 显示对话历史
    print("\n" + "="*70)
    print("Conversation History:")
    print("="*70)
    history = agent.get_conversation_history(k=10)
    for i, turn in enumerate(history, 1):
        print(f"\n[{i}] Q: {turn['query']}")
        print(f"    A: {turn['answer'][:100]}...")
        print(f"    Tools: {turn['tools']}")

    print("\n" + "="*70)
    print("Demo Complete!")
    print("="*70)
    print("\n💡 Memory data is persisted in: data/memory/memories.json")
    print("   You can run this demo again and the agent will remember previous conversations!")


if __name__ == "__main__":
    main()
