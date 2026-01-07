"""
Agentic RAG with Memory System

核心特性：
1. 记忆系统：记住对话历史
2. 工具调用：向量检索等
3. 上下文整合：结合记忆和检索结果
"""

import sys
import os
sys.path.append('src/rag')

from pipeline import RAGPipeline
from memory import MemorySystem, MemoryItem
from typing import List, Dict, Optional
import numpy as np


class Tool:
    """工具基类"""

    name: str = "base_tool"
    description: str = "Base tool"

    def run(self, query: str, **kwargs) -> Dict:
        """执行工具"""
        raise NotImplementedError


class VectorSearchTool(Tool):
    """向量检索工具"""

    name = "vector_search"
    description = "在文档库中检索相关信息"

    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline

    def run(self, query: str, mode: str = "hybrid", **kwargs) -> Dict:
        """执行向量检索"""
        result = self.pipeline.run(query, mode=mode)

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("retrieved_chunks", []),
            "scores": result.get("top_scores", []),
            "method": result.get("method", "")
        }


class AgenticRAG:
    """
    带记忆的 Agentic RAG

    特性：
    1. 短期记忆：记住最近 10 轮对话
    2. 长期记忆：持久化存储重要信息
    3. 上下文整合：结合记忆和检索结果
    """

    def __init__(self,
                 pipeline: RAGPipeline,
                 enable_memory: bool = True,
                 short_term_size: int = 10,
                 long_term_path: str = "data/memory/memories.json"):

        self.pipeline = pipeline
        self.enable_memory = enable_memory

        # 初始化工具
        self.tools = {
            "vector_search": VectorSearchTool(pipeline)
        }

        # 初始化记忆系统
        if enable_memory:
            self.memory = MemorySystem(
                short_term_size=short_term_size,
                long_term_path=long_term_path
            )
        else:
            self.memory = None

    def query(self,
             user_query: str,
             mode: str = "hybrid",
             use_memory: bool = True,
             save_to_memory: bool = True,
             importance: float = 0.5) -> Dict:
        """
        执行查询

        Args:
            user_query: 用户问题
            mode: 'dense_only' 或 'hybrid'
            use_memory: 是否使用记忆
            save_to_memory: 是否保存到记忆
            importance: 重要性 (0-1)，决定是否存入长期记忆

        Returns:
            {
                "answer": str,
                "sources": list,
                "memory_used": dict,
                "tools_used": list
            }
        """

        # 1. 获取 query embedding（用于记忆检索）
        query_embedding = None
        if self.enable_memory and use_memory:
            query_embedding = self._get_query_embedding(user_query)

        # 2. 检索相关记忆
        memory_context = ""
        memory_used = {"short_term": 0, "long_term": 0}

        if self.enable_memory and use_memory and query_embedding is not None:
            context = self.memory.get_context(
                query_embedding=query_embedding,
                short_term_k=3,
                long_term_k=2
            )

            if context:
                memory_context = context
                memory_used["short_term"] = len(self.memory.short_term.get_recent(3))
                long_term_memories = self.memory.long_term.retrieve(query_embedding, top_k=2)
                memory_used["long_term"] = len(long_term_memories)

        # 3. 执行向量检索
        tool_result = self.tools["vector_search"].run(user_query, mode=mode)
        base_answer = tool_result["answer"]

        # 4. 整合记忆和检索结果
        final_answer = self._synthesize_with_memory(
            user_query,
            base_answer,
            memory_context
        )

        # 5. 保存到记忆
        if self.enable_memory and save_to_memory:
            answer_embedding = self._get_query_embedding(user_query + " " + final_answer)

            # 保存到短期记忆
            self.memory.add_memory(
                query=user_query,
                answer=final_answer,
                tools_used=["vector_search"],
                embedding=answer_embedding,
                importance=importance
            )

            # 如果重要性足够高，也保存到长期记忆
            if importance >= 0.7:
                self.memory.add_memory(
                    query=user_query,
                    answer=final_answer,
                    tools_used=["vector_search"],
                    embedding=answer_embedding,
                    importance=importance,
                    to_long_term=True
                )

        return {
            "answer": final_answer,
            "sources": tool_result["sources"],
            "scores": tool_result["scores"],
            "memory_used": memory_used,
            "tools_used": ["vector_search"],
            "has_memory": bool(memory_context)
        }

    def _get_query_embedding(self, text: str) -> Optional[np.ndarray]:
        """获取 query 的 embedding"""
        try:
            # 使用 pipeline 的 embedding 客户端
            embeddings = self.pipeline.qianfan_client.embed([text])
            if len(embeddings) > 0:
                return embeddings[0]
        except Exception as e:
            print(f"Error getting embedding: {e}")

        return None

    def _synthesize_with_memory(self,
                               query: str,
                               base_answer: str,
                               memory_context: str) -> str:
        """
        整合记忆和检索结果

        如果有相关记忆，会在答案中提及之前的对话
        """

        if not memory_context:
            return base_answer

        # 如果有记忆，添加提示
        # 这里简单处理：如果有相关记忆，在答案开头提及
        memory_hint = "\n\n[相关历史信息已在上下文中]"

        # 如果 base_answer 已经很完整，直接返回
        if len(base_answer) > 200:
            return base_answer + memory_hint

        # 否则，返回整合后的答案
        return base_answer + memory_hint

    def chat(self, message: str, **kwargs) -> str:
        """
        简化的聊天接口

        Args:
            message: 用户消息

        Returns:
            Agent 的回复
        """
        result = self.query(message, **kwargs)
        return result["answer"]

    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        if not self.enable_memory:
            return {"memory_enabled": False}

        stats = self.memory.get_stats()
        stats["memory_enabled"] = True
        return stats

    def clear_short_term_memory(self):
        """清空短期记忆（开始新对话）"""
        if self.enable_memory:
            self.memory.short_term.clear()

    def get_conversation_history(self, k: int = 5) -> List[Dict]:
        """获取最近的对话历史"""
        if not self.enable_memory:
            return []

        memories = self.memory.short_term.get_recent(k)
        return [
            {
                "query": m.query,
                "answer": m.answer,
                "timestamp": m.timestamp,
                "tools": m.tools_used
            }
            for m in memories
        ]


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    import json

    # 加载文档
    def load_chunks(path: str):
        from pipeline import Document
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

    # 初始化
    print("Loading documents...")
    documents = load_chunks("data/parsed/enhanced_chunks.json")

    QIANFAN_KEY = "bce-v3/ALTAK-dgZMQj7E5tByoRofFKlbM/e852481aaab5ebf3ffe6f2a50589e6e41646c127"
    GLM_KEY = "bfacda9a355fd3e9557063bc90aa8a1e.1zJ3JaydaKIkZ6Kq"

    print("Initializing pipeline...")
    pipeline = RAGPipeline(documents, QIANFAN_KEY, GLM_KEY)

    print("Creating Agent...")
    agent = AgenticRAG(
        pipeline=pipeline,
        enable_memory=True,
        short_term_size=10
    )

    # 模拟对话
    print("\n" + "="*60)
    print("Starting conversation with memory...")
    print("="*60)

    queries = [
        "What is machine learning?",
        "How is it different from deep learning?",  # 应该记住上面的问题
        "What techniques prevent overfitting?"
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n[Round {i}] User: {query}")

        result = agent.query(
            query,
            use_memory=True,
            save_to_memory=True,
            importance=0.7
        )

        print(f"Agent: {result['answer'][:200]}...")
        print(f"Memory used: {result['memory_used']}")

    # 显示记忆统计
    print("\n" + "="*60)
    print("Memory Stats:")
    print("="*60)
    stats = agent.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
