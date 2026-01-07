"""
Memory System for Agentic RAG

Provides:
1. Short-term memory: Recent conversation history
2. Long-term memory: Persistent vector storage of important information
3. Memory retrieval: Find relevant memories based on query
"""

import time
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


@dataclass
class MemoryItem:
    """单条记忆"""
    query: str                    # 用户问题
    answer: str                   # Agent 回答
    tools_used: List[str]         # 使用的工具
    timestamp: float              # 时间戳
    session_id: str = ""          # 会话 ID
    importance: float = 0.5       # 重要性 (0-1)
    embedding: Optional[np.ndarray] = None  # 向量（用于检索）

    def to_dict(self) -> dict:
        """转换为字典"""
        data = asdict(self)
        if self.embedding is not None:
            data["embedding"] = self.embedding.tolist()
        else:
            data["embedding"] = None
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'MemoryItem':
        """从字典创建"""
        if data.get("embedding"):
            data["embedding"] = np.array(data["embedding"])
        return cls(**data)


class ShortTermMemory:
    """短期记忆：存储最近的对话历史"""

    def __init__(self, max_items: int = 10):
        self.max_items = max_items
        self.memories: List[MemoryItem] = []

    def add(self, query: str, answer: str, tools_used: List[str] = None,
            importance: float = 0.5) -> MemoryItem:
        """添加记忆"""
        memory = MemoryItem(
            query=query,
            answer=answer,
            tools_used=tools_used or [],
            timestamp=time.time(),
            importance=importance
        )
        self.memories.append(memory)

        # 保持最大数量
        if len(self.memories) > self.max_items:
            self.memories.pop(0)

        return memory

    def get_recent(self, k: int = 5) -> List[MemoryItem]:
        """获取最近的 k 条记忆"""
        return self.memories[-k:]

    def get_all(self) -> List[MemoryItem]:
        """获取所有记忆"""
        return self.memories

    def clear(self):
        """清空记忆"""
        self.memories = []

    def get_context_string(self, k: int = 3) -> str:
        """获取格式化的上下文字符串"""
        recent = self.get_recent(k)
        if not recent:
            return ""

        context_parts = []
        for i, memory in enumerate(recent, 1):
            context_parts.append(
                f"[Round {i}] Q: {memory.query}\nA: {memory.answer}"
            )

        return "=== Previous Conversation ===\n" + "\n".join(context_parts)


class LongTermMemory:
    """长期记忆：持久化存储重要信息"""

    def __init__(self, storage_path: str = "data/memory/memories.json"):
        self.storage_path = storage_path
        self.memories: List[MemoryItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self._load()

    def _load(self):
        """从文件加载记忆"""
        import os
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.memories = [MemoryItem.from_dict(item) for item in data]

                # 提取 embeddings
                embeddings = [m.embedding for m in self.memories if m.embedding is not None]
                if embeddings:
                    self.embeddings = np.array(embeddings)
                else:
                    self.embeddings = None

                print(f"Loaded {len(self.memories)} long-term memories")
            except Exception as e:
                print(f"Error loading memories: {e}")
                self.memories = []

    def _save(self):
        """保存记忆到文件"""
        import os
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

        data = [m.to_dict() for m in self.memories]
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, memory: MemoryItem):
        """添加记忆（需要 embedding）"""
        self.memories.append(memory)
        self._save()

        # 更新 embeddings
        if memory.embedding is not None:
            if self.embeddings is None:
                self.embeddings = np.array([memory.embedding])
            else:
                self.embeddings = np.vstack([self.embeddings, memory.embedding])

    def add_with_embedding(self, query: str, answer: str, embedding: np.ndarray,
                          tools_used: List[str] = None, importance: float = 0.5):
        """添加带 embedding 的记忆"""
        memory = MemoryItem(
            query=query,
            answer=answer,
            tools_used=tools_used or [],
            timestamp=time.time(),
            importance=importance,
            embedding=embedding
        )
        self.add(memory)
        return memory

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3,
                 min_importance: float = 0.0) -> List[Tuple[MemoryItem, float]]:
        """检索相关记忆"""
        if self.embeddings is None or len(self.memories) == 0:
            return []

        # 计算相似度
        similarities = np.dot(self.embeddings, query_embedding)

        # 过滤重要性
        valid_indices = [
            i for i, m in enumerate(self.memories)
            if m.importance >= min_importance
        ]

        if not valid_indices:
            return []

        # 排序并返回 top_k
        valid_indices = sorted(
            valid_indices,
            key=lambda i: similarities[i],
            reverse=True
        )[:top_k]

        return [
            (self.memories[i], float(similarities[i]))
            for i in valid_indices
        ]

    def get_all(self) -> List[MemoryItem]:
        """获取所有记忆"""
        return self.memories

    def clear(self):
        """清空所有记忆"""
        self.memories = []
        self.embeddings = None
        self._save()


class MemorySystem:
    """完整的记忆系统"""

    def __init__(self,
                 short_term_size: int = 10,
                 long_term_path: str = "data/memory/memories.json"):
        self.short_term = ShortTermMemory(max_items=short_term_size)
        self.long_term = LongTermMemory(storage_path=long_term_path)

    def add_memory(self, query: str, answer: str, tools_used: List[str] = None,
                  embedding: np.ndarray = None, importance: float = 0.5,
                  to_long_term: bool = False):
        """添加记忆"""
        # 总是添加到短期记忆
        self.short_term.add(query, answer, tools_used, importance)

        # 如果指定，也添加到长期记忆
        if to_long_term and embedding is not None:
            self.long_term.add_with_embedding(
                query, answer, embedding, tools_used, importance
            )

    def retrieve(self, query_embedding: np.ndarray, top_k: int = 3) -> Dict:
        """检索相关记忆（短期 + 长期）"""
        # 从长期记忆检索
        long_term_results = self.long_term.retrieve(query_embedding, top_k=top_k)

        # 从短期记忆检索（简单的关键词匹配）
        short_term_results = []
        for memory in self.short_term.get_all():
            # 简单的相关性判断
            relevance = self._simple_relevance(query_embedding, memory)
            if relevance > 0.1:  # 阈值
                short_term_results.append((memory, relevance))

        # 合并去重
        all_results = long_term_results + short_term_results

        # 按 relevance 排序
        all_results.sort(key=lambda x: x[1], reverse=True)

        return {
            "memories": [m for m, _ in all_results[:top_k]],
            "scores": [s for _, s in all_results[:top_k]]
        }

    def _simple_relevance(self, query_embedding: np.ndarray,
                        memory: MemoryItem) -> float:
        """简单相关性计算（用于短期记忆）"""
        if memory.embedding is None:
            # 如果没有 embedding，用时间衰减
            age = time.time() - memory.timestamp
            return max(0, 1 - age / 3600)  # 1小时内的记忆
        else:
            # 用 embedding 相似度
            return float(np.dot(query_embedding, memory.embedding))

    def get_context(self, query_embedding: np.ndarray = None,
                   short_term_k: int = 3, long_term_k: int = 2) -> str:
        """获取完整的上下文（用于 prompt）"""
        context_parts = []

        # 短期记忆上下文
        short_term_context = self.short_term.get_context_string(short_term_k)
        if short_term_context:
            context_parts.append(short_term_context)

        # 长期记忆上下文
        if query_embedding is not None:
            long_term_memories = self.long_term.retrieve(query_embedding, top_k=long_term_k)
            if long_term_memories:
                context_parts.append("\n=== Relevant Past Information ===")
                for memory, score in long_term_memories:
                    context_parts.append(
                        f"[{score:.2f}] Q: {memory.query}\nA: {memory.answer}"
                    )

        return "\n\n".join(context_parts)

    def promote_to_long_term(self, recent_index: int = -1,
                            embedding_func=None):
        """将短期记忆提升到长期记忆"""
        if recent_index == -1:
            # 提升最近的一条
            memory = self.short_term.memories[-1]
        else:
            memory = self.short_term.memories[recent_index]

        # 如果没有 embedding，需要生成
        if memory.embedding is None and embedding_func is not None:
            memory.embedding = embedding_func(memory.query + " " + memory.answer)

        if memory.embedding is not None:
            self.long_term.add(memory)
            return True
        return False

    def clear_short_term(self):
        """清空短期记忆"""
        self.short_term.clear()

    def get_stats(self) -> dict:
        """获取记忆统计"""
        return {
            "short_term_count": len(self.short_term.memories),
            "long_term_count": len(self.long_term.memories),
            "total_memories": len(self.short_term.memories) + len(self.long_term.memories)
        }


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 创建记忆系统
    memory = MemorySystem()

    # 添加一些测试记忆
    test_queries = [
        "什么是机器学习？",
        "如何计算 F1 分数？",
        "什么是过拟合？"
    ]

    test_answers = [
        "机器学习是人工智能的一个分支...",
        "F1 分数 = 2 * (precision * recall) / (precision + recall)",
        "过拟合是指模型在训练集上表现很好但泛化能力差..."
    ]

    # 模拟 embedding
    for i, (q, a) in enumerate(zip(test_queries, test_answers)):
        # 创建假的 embedding（实际应该用 embedding 模型）
        embedding = np.random.randn(2560)
        embedding /= np.linalg.norm(embedding)  # 归一化

        memory.add_memory(
            query=q,
            answer=a,
            tools_used=["vector_search"],
            embedding=embedding,
            importance=0.7,
            to_long_term=True
        )

    # 测试检索
    query_embedding = np.random.randn(2560)
    query_embedding /= np.linalg.norm(query_embedding)

    results = memory.retrieve(query_embedding, top_k=2)
    print("Retrieved memories:")
    for m, score in results["memories"]:
        print(f"  [{score:.3f}] {m.query}")

    # 获取上下文
    context = memory.get_context(query_embedding)
    print("\nContext:")
    print(context[:200] + "...")
