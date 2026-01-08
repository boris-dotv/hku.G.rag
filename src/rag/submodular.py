"""
Submodular Optimization for RAG - 多样性优化

解决 Reranker 返回重复文档的问题
目标：既保证相关性，又保证多样性

原理：
- 次模函数：f(S ∪ {x}) - f(S) 随 |S| 增大而递减
- 贪心算法：每次选择边际收益最大的元素
- 懒惰优化：用优先队列减少计算量
"""

import heapq
import numpy as np
from typing import List, Dict, Tuple, Callable
from dataclasses import dataclass


@dataclass
class Document:
    """文档对象"""
    doc_id: str
    content: str
    score: float = 0.0
    metadata: Dict = None


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """计算余弦相似度"""
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2) + 1e-8))


def get_similarity_matrix(documents: List[Document],
                          embeddings: np.ndarray = None) -> np.ndarray:
    """
    计算文档相似度矩阵

    Args:
        documents: 文档列表
        embeddings: 可选的预计算 embeddings (N x D)

    Returns:
        N x N 相似度矩阵
    """
    n = len(documents)

    if embeddings is not None:
        # 使用预计算的 embeddings
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarity_matrix = np.dot(normalized, normalized.T)
    else:
        # 使用简单的文本重叠（jieba 分词）
        import jieba
        from collections import Counter

        similarity_matrix = np.zeros((n, n))

        # 预计算分词
        token_sets = []
        for doc in documents:
            tokens = set(jieba.lcut(doc.content))
            token_sets.append(tokens)

        # 计算相似度
        for i in range(n):
            for j in range(i + 1, n):
                # Jaccard 相似度
                intersection = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                sim = intersection / union if union > 0 else 0

                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

    return similarity_matrix


def passage_set_gain(
    S: List[int],
    similarity_matrix: np.ndarray,
    relevance_scores: List[float],
    theta: float = 0.85,
    penalty: float = 0.7
) -> float:
    """
    计算文档集合的条件相关收益

    Args:
        S: 已选文档的索引列表
        similarity_matrix: 相似度矩阵
        relevance_scores: 相关性分数（来自 Reranker）
        theta: 相似度阈值（超过此值视为重复）
        penalty: 重复文档的惩罚系数

    Returns:
        集合的总收益
    """
    if not S:
        return 0.0

    score = 0.0
    for i in S:
        base_score = relevance_scores[i]

        # 检查是否与已选文档重复
        weight = 1.0
        for j in S:
            if i == j:
                continue

            if similarity_matrix[i][j] > theta:
                # 发现重复，应用惩罚
                weight = penalty
                break

        score += weight * base_score

    return score


def marginal_passage_set_gain(
    new_idx: int,
    selected: List[int],
    similarity_matrix: np.ndarray,
    relevance_scores: List[float],
    theta: float = 0.85,
    penalty: float = 0.7
) -> float:
    """
    计算添加新文档的边际收益

    Args:
        new_idx: 新文档的索引
        selected: 已选文档的索引列表
        similarity_matrix: 相似度矩阵
        relevance_scores: 相关性分数
        theta: 相似度阈值
        penalty: 惩罚系数

    Returns:
        边际收益 = f(S ∪ {new}) - f(S)
    """
    current_score = passage_set_gain(
        selected, similarity_matrix, relevance_scores, theta, penalty
    )
    new_score = passage_set_gain(
        selected + [new_idx], similarity_matrix, relevance_scores, theta, penalty
    )
    return new_score - current_score


def lazy_greedy_selection(
    similarity_matrix: np.ndarray,
    relevance_scores: List[float],
    k: int,
    compute_marginal_gain: Callable,
    theta: float = 0.85,
    penalty: float = 0.7
) -> List[int]:
    """
    懒惰贪心选择算法

    时间复杂度：O(n log n) 而非 O(nk)

    Args:
        similarity_matrix: 相似度矩阵
        relevance_scores: 相关性分数
        k: 要选择的文档数量
        compute_marginal_gain: 边际收益计算函数
        theta: 相似度阈值
        penalty: 惩罚系数

    Returns:
        选中的文档索引列表
    """
    n = len(relevance_scores)
    selected = []

    # 初始化优先队列
    # 每个元素: (-边际收益, 最后更新的迭代, 文档索引)
    pq = []

    for i in range(n):
        # 初始边际收益（此时 S 为空）
        gain = relevance_scores[i]  # 空集合的边际收益就是自身分数
        heapq.heappush(pq, (-gain, 0, i))

    # 贪心选择 k 次
    for iteration in range(k):
        if not pq:
            break

        while True:
            neg_gain, last_updated, best_idx = heapq.heappop(pq)

            # 如果这个收益是当前迭代计算的，就是最终答案
            if last_updated == iteration:
                selected.append(best_idx)
                break

            # 否则，重新计算边际收益
            current_gain = compute_marginal_gain(
                best_idx, selected, similarity_matrix,
                relevance_scores, theta, penalty
            )

            # 重新放回队列
            heapq.heappush(pq, (-current_gain, iteration, best_idx))

    return selected


def submodular_rerank(
    documents: List[Document],
    relevance_scores: List[float],
    top_k: int = 5,
    theta: float = 0.85,
    penalty: float = 0.7,
    embeddings: np.ndarray = None
) -> Tuple[List[Document], List[int]]:
    """
    次模优化重排序

    Args:
        documents: 候选文档列表（已按 Reranker 分数排序）
        relevance_scores: Reranker 的相关性分数
        top_k: 要选择的文档数量
        theta: 相似度阈值（默认 0.85）
        penalty: 重复惩罚系数（默认 0.7）
        embeddings: 可选的预计算 embeddings

    Returns:
        (优化后的文档列表, 选中的索引)
    """
    if len(documents) <= top_k:
        return documents, list(range(len(documents)))

    # 计算相似度矩阵
    sim_matrix = get_similarity_matrix(documents, embeddings)

    # 懒惰贪心选择
    selected_indices = lazy_greedy_selection(
        sim_matrix,
        relevance_scores,
        top_k,
        marginal_passage_set_gain,
        theta,
        penalty
    )

    # 返回选中的文档
    selected_docs = [documents[i] for i in selected_indices]

    return selected_docs, selected_indices


# ============================================
# 便捷接口
# ============================================

def diversify_rag_results(
    retrieved_chunks: List[Dict],
    top_k: int = 5,
    similarity_threshold: float = 0.85,
    diversity_penalty: float = 0.7
) -> List[Dict]:
    """
    对 RAG 检索结果进行多样性优化

    Args:
        retrieved_chunks: 检索到的 chunks（每个有 content 和 score）
        top_k: 要保留的文档数量
        similarity_threshold: 相似度阈值
        diversity_penalty: 重复惩罚

    Returns:
        优化后的 chunks
    """
    # 转换为 Document 对象
    documents = []
    scores = []

    for i, chunk in enumerate(retrieved_chunks):
        doc = Document(
            doc_id=chunk.get("chunk_id", f"chunk_{i}"),
            content=chunk.get("content", ""),
            score=chunk.get("score", 0.0),
            metadata=chunk.get("metadata", {})
        )
        documents.append(doc)
        scores.append(doc.score)

    # 次模优化
    selected_docs, selected_indices = submodular_rerank(
        documents,
        scores,
        top_k=top_k,
        theta=similarity_threshold,
        penalty=diversity_penalty
    )

    # 转换回原始格式
    result = []
    for i, doc in enumerate(selected_docs):
        result.append({
            "chunk_id": doc.doc_id,
            "content": doc.content,
            "score": doc.score,
            "metadata": doc.metadata,
            "rank": i + 1
        })

    return result


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    # 模拟重复文档问题
    test_documents = [
        Document("doc1", "Machine learning is a subset of AI.", 0.95),
        Document("doc2", "Machine learning is a subset of AI techniques.", 0.94),  # 几乎重复
        Document("doc3", "Deep learning uses neural networks.", 0.92),
        Document("doc4", "Deep learning is a type of machine learning.", 0.91),
        Document("doc5", "Overfitting occurs when a model memorizes training data.", 0.88),
        Document("doc6", "Overfitting is when models perform well on training but poorly on test.", 0.87),
    ]

    test_scores = [0.95, 0.94, 0.92, 0.91, 0.88, 0.87]

    print("原始排序（按相关性）:")
    for i, doc in enumerate(test_documents):
        print(f"  {i+1}. [{doc.score:.2f}] {doc.content[:50]}...")

    # 次模优化
    selected, indices = submodular_rerank(
        test_documents,
        test_scores,
        top_k=3,
        theta=0.85,
        penalty=0.7
    )

    print("\n次模优化后（相关性 + 多样性）:")
    for i, doc in enumerate(selected):
        print(f"  {i+1}. [{doc.score:.2f}] {doc.content[:50]}...")
