"""
Enhanced Agentic RAG with Query Rewriting and ReAct Loop

核心特性：
1. Query Rewriting: 基于对话历史重写查询（解决指代消解）
2. ReAct Loop: Thought → Action → Observation 循环
3. 记忆系统: 短期 + 长期记忆
4. 工具调用: 向量检索
"""

import sys
import os
sys.path.append('src/rag')

import json
import requests
import re
import time
import signal
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pipeline import RAGPipeline, RAGConfig
from .memory import MemorySystem, MemoryItem
import numpy as np


# ============================================
# Timeout Exception
# ============================================
class TimeoutError(Exception):
    """Exception raised when operation times out"""
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")


# ============================================
# Query Rewriter
# ============================================
class QueryRewriter:
    """查询重写器：基于对话历史解决指代消解"""

    def __init__(self, glm_key: str):
        self.glm_key = glm_key
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def rewrite(self, current_query: str, conversation_history: List[Dict]) -> str:
        """
        重写查询，解析代词和省略（激进策略：有历史就总是尝试重写）

        Args:
            current_query: 当前用户查询
            conversation_history: 对话历史

        Returns:
            重写后的查询
        """
        # 【核心修改】：只要历史不为空，就强制让 LLM 审视一次查询
        # 不要相信预判，让 LLM 决定是否需要修改
        if not conversation_history:
            return current_query

        # 构建历史上下文（使用最近一轮，因为Q4只依赖Q3）
        last_turn = conversation_history[-1]
        # _get_conversation_history() 返回字典，用 get() 访问
        context_query = last_turn.get('query', '')
        context_answer = last_turn.get('answer', '')[:300] if last_turn.get('answer') else ""  # 只用前300字

        # 【更强的 Prompt】：明确强调代词检测
        prompt = f"""You are a conversation context resolver. Your ONLY job is to resolve pronouns and implicit references.

Current Query: "{current_query}"

Context from PREVIOUS turn:
- Previous User Question: "{context_query}"
- Previous Answer: "{context_answer}..."

CRITICAL TASK:
1. Does the Current Query contain pronouns (it, they, this, that, its, their, them)?
2. If YES, you MUST replace the pronoun with the actual topic from the Context.
3. If NO, output the Current Query exactly as-is.

Examples:
- Current: "How can it be prevented?" + Context about "overfitting" → "How can overfitting be prevented?"
- Current: "What are its limitations?" + Context about "deep learning" → "What are the limitations of deep learning?"
- Current: "Explain the process" + Context about "data splitting" → "Explain the data splitting process"

Output ONLY the rewritten query. No explanations, no quotes:"""

        try:
            headers = {"Authorization": f"Bearer {self.glm_key}"}
            payload = {
                "model": RAGConfig.CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.1  # 降低温度让输出更确定
            }
            res = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()

            if "choices" in data:
                rewritten = data["choices"][0]["message"]["content"].strip()
                # 清理可能的引号和多余字符
                rewritten = rewritten.strip('"\'').strip()

                # 只要有改动，就使用重写后的版本
                if rewritten and rewritten.lower() != current_query.lower():
                    print(f"  [🔄 Query Rewriting] '{current_query}' → '{rewritten}'")
                    return rewritten
                else:
                    print(f"  [✓ Query Rewriting] No change needed for '{current_query}'")
                    return current_query

        except Exception as e:
            print(f"  [⚠️ Query Rewriting Error: {e}. Using original query.]")

        return current_query

    def _check_needs_rewriting(self, query: str) -> bool:
        """检查是否需要重写"""
        # 检查代词
        pronouns = ['it', 'its', 'they', 'them', 'their', 'this', 'that', 'these', 'those']
        query_lower = query.lower()

        # 检查是否有代词开头
        words = query_lower.split()
        if words and words[0] in pronouns:
            return True

        # 检查是否包含代词（更宽松的检查，支持 "How can it be prevented?"）
        if any(pronoun in query_lower for pronoun in pronouns):
            return True

        # 检查是否太短（可能是省略）
        if len(query.split()) <= 3 and not query.endswith('?'):
            return True

        return False

    def _format_history(self, history: List[Dict]) -> str:
        """格式化历史 - 增加上下文长度以更好地解析代词"""
        lines = []
        for i, turn in enumerate(history, 1):
            query = turn.get('query', '')
            answer = turn.get('answer', '')
            # Include more context (200 chars) and the full last query
            if i == len(history):  # Most recent query - include full answer
                lines.append(f"Q{i}: {query}")
                lines.append(f"A{i}: {answer[:300]}...")
            else:  # Older queries - shorter context
                lines.append(f"Q{i}: {query}")
                lines.append(f"A{i}: {answer[:150]}...")
        return '\n'.join(lines)


# ============================================
# Query Decomposer
# ============================================
class QueryDecomposer:
    """查询拆解器：使用LLM将复杂查询拆解为多个子查询"""

    def __init__(self, glm_key: str):
        self.glm_key = glm_key
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def decompose(self, query: str) -> Tuple[bool, List[str]]:
        """
        使用LLM判断并拆解复杂查询

        Returns:
            (should_decompose, sub_queries)
            - should_decompose: 是否需要拆解
            - sub_queries: 拆解后的子查询列表
        """
        prompt = f"""You are a query decomposition assistant. Analyze the user's query and determine if it needs to be broken down into multiple search queries.

User Query: {query}

Analyze the query:
1. If it compares two or more things (e.g., "difference between X and Y"), create separate queries for each
2. If it has multiple distinct questions, break them down
3. If it's simple and direct, keep it as-is

Output ONLY a JSON list of strings. For example:
- Simple query: ["original query"]
- Comparison: ["query about first topic", "query about second topic", "comparison query"]
- Multi-part: ["first part", "second part"]

JSON:"""

        try:
            headers = {"Authorization": f"Bearer {self.glm_key}"}
            payload = {
                "model": RAGConfig.CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300,
                "temperature": 0.3
            }
            res = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()

            if "choices" in data:
                response = data["choices"][0]["message"]["content"].strip()

                # Try to extract JSON from response (LLM might add extra text)
                if not response:
                    print(f"    [Query decomposition: Empty response. Using original query.]")
                    return False, [query]

                # Find JSON array in response
                json_start = response.find('[')
                json_end = response.rfind(']') + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    sub_queries = json.loads(json_str)
                else:
                    # No JSON array found, try parsing whole response
                    sub_queries = json.loads(response)

                # Validate that we got a list of strings
                if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                    # Check if decomposition is needed (more than 1 query or different from original)
                    if len(sub_queries) > 1:
                        return True, sub_queries
                    elif len(sub_queries) == 1 and sub_queries[0].lower() != query.lower():
                        # LLM rephrased the query, use it
                        return False, sub_queries
                    else:
                        return False, [query]
                else:
                    print(f"    [Query decomposition: Invalid format. Using original query.]")
                    return False, [query]

        except json.JSONDecodeError as e:
            print(f"    [Query decomposition JSON error: {e}. Using original query.]")
        except Exception as e:
            print(f"    [Query decomposition error: {e}. Using original query.]")

        # Fallback: don't decompose
        return False, [query]


# ============================================
# Enhanced Agent with ReAct
# ============================================
class ReActAgent:
    """
    ReAct Agent: 推理 + 行动

    循环:
    1. Thought: 分析当前情况，决定下一步
    2. Action: 执行工具或直接回答
    3. Observation: 观察结果，决定是否继续
    """

    def __init__(self,
                 pipeline: RAGPipeline,
                 glm_key: str,
                 enable_memory: bool = True,
                 enable_react: bool = True,
                 max_iterations: int = 3):
        self.pipeline = pipeline
        self.glm_key = glm_key
        self.enable_memory = enable_memory
        self.enable_react = enable_react
        self.max_iterations = max_iterations
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

        # 初始化组件
        self.query_rewriter = QueryRewriter(glm_key)
        self.query_decomposer = QueryDecomposer(glm_key)  # 新增：查询拆解器

        if enable_memory:
            self.memory = MemorySystem(
                short_term_size=10,
                long_term_path="data/memory/memories.json"
            )
        else:
            self.memory = None

        # 工具注册
        self.tools = {
            "vector_search": self._vector_search,
            "bm25_search": self._bm25_search,
        }

    def query(self,
             user_query: str,
             mode: str = "hybrid",
             use_react: bool = True,
             use_memory: bool = True,
             save_to_memory: bool = True,
             importance: float = 0.5,
             verbose: bool = True) -> Dict:
        """
        执行查询（带 ReAct 循环）

        Args:
            user_query: 用户查询
            mode: 检索模式
            use_react: 是否使用 ReAct 循环
            use_memory: 是否使用记忆
            save_to_memory: 是否保存到记忆
            importance: 重要性分数
            verbose: 是否打印思考过程

        Returns:
            查询结果
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"🤖 ReAct Agent Processing: {user_query}")
            print(f"{'='*70}")

        # ============ Step 1: Query Rewriting ============
        rewritten_query = user_query
        # Query rewriting can work independently of memory saving
        # It only needs conversation history for context
        if self.memory:
            history = self._get_conversation_history()
            rewritten_query = self.query_rewriter.rewrite(user_query, history)

        # ============ Step 2: ReAct Loop ============
        if use_react and self.enable_react:
            result = self._react_loop(rewritten_query, mode, verbose)
        else:
            # 直接查询
            result = self._direct_query(rewritten_query, mode)

        # ============ Step 3: Save to Memory ============
        if save_to_memory and self.memory:
            self._save_to_memory(
                user_query,
                rewritten_query,
                result["answer"],
                result.get("tools_used", []),
                importance
            )

        # 添加元数据
        result["original_query"] = user_query
        result["rewritten_query"] = rewritten_query
        result["query_was_rewritten"] = (user_query != rewritten_query)

        return result

    # ============================================
    # New Helper Methods for Enhanced ReAct
    # ============================================

    def _check_semantic_relevance(self, query: str, observation: Dict) -> Dict:
        """
        语义相关性校验：检查检索结果是否真的回答了问题

        不只看长度，而是用LLM判断语义是否相关
        """
        top_chunks = observation.get("sources", [])[:3]
        if not top_chunks:
            return {"is_satisfactory": False, "issue": "no_sources", "summary": "No sources"}

        chunks_text = "\n\n".join([f"- {c.get('content', '')[:200]}" for c in top_chunks])

        prompt = f"""You are a relevance checker. Determine if the retrieved content actually answers the question.

Question: {query}

Retrieved Content:
{chunks_text}

Strictly evaluate:
- If the content is about a completely different topic (e.g., aerodynamics for a data question), respond "No"
- If the content mentions keywords but doesn't actually answer, respond "Partial"
- Only respond "Yes" if it directly addresses the question

Answer (just one word: Yes/No/Partial):"""

        try:
            headers = {"Authorization": f"Bearer {self.glm_key}"}
            payload = {
                "model": RAGConfig.CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
                "temperature": 0.1
            }
            res = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()

            if "choices" in data:
                answer = data["choices"][0]["message"]["content"].strip().lower()

                if "no" in answer:
                    return {
                        "is_satisfactory": False,
                        "issue": "semantic_mismatch",
                        "summary": "Retrieved content is semantically irrelevant"
                    }
                elif "partial" in answer:
                    return {
                        "is_satisfactory": False,
                        "issue": "partial_match",
                        "summary": "Retrieved content only partially relevant"
                    }

        except Exception as e:
            print(f"    [Semantic check error: {e}. Continuing...]")

        return {"is_satisfactory": True, "summary": "Semantically relevant"}

    def _expand_query(self, original_query: str, context: Dict) -> str:
        """
        查询扩展：当检索失败时，生成更多样化的搜索词
        """
        # 从之前的observation中提取关键词
        keywords = set()
        if context["observations"]:
            last_obs = context["observations"][-1]
            sources = last_obs.get("sources", [])
            for source in sources[:2]:
                content = source.get("content", "")
                # 简单的n-gram关键词提取（取2-3词的短语）
                words = content.lower().split()
                for i in range(len(words) - 1):
                    if len(words[i]) > 3:  # 只取长词
                        keywords.add(words[i])
                        if i < len(words) - 1 and len(words[i+1]) > 3:
                            keywords.add(f"{words[i]} {words[i+1]}")

        # 构建扩展查询
        if keywords:
            top_keywords = list(keywords)[:5]
            return f"{original_query} {' '.join(top_keywords[:3])}"

        return original_query

    def _extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """从文本中提取关键词（简单实现）"""
        # 移除停用词
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'is', 'are', 'was', 'were'}
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) >= 4]

        # 统计词频
        from collections import Counter
        word_freq = Counter(keywords)

        return [w for w, _ in word_freq.most_common(top_n)]

    def _react_loop(self, query: str, mode: str, verbose: bool) -> Dict:
        """ReAct 循环：Thought → Action → Observation → Reflection

        支持查询拆解（Query Decomposition）来处理复杂问题
        支持上下文注入（Context Injection）来保持对话连续性
        Hard timeout: 120 seconds max to prevent infinite loops
        """

        context = {
            "query": query,
            "thoughts": [],
            "actions": [],
            "observations": [],
            "reflections": [],
            "final_answer": "",
            "sub_queries": [],  # 新增：记录子查询
            "previous_topic": ""  # 新增：记录上一轮的主题
        }

        # ============================================
        # Step -1: 获取上一轮的主题（上下文注入）
        # ============================================
        if self.memory:
            # 获取最近1轮的对话历史
            recent_history = self.memory.short_term.get_recent(1)
            if recent_history:
                prev_query = recent_history[0].query  # MemoryItem 用属性访问
                # 提取关键词作为主题
                keywords = self._extract_keywords(prev_query, top_n=3)
                if keywords:
                    context["previous_topic"] = f"Previous topic: {', '.join(keywords)}"
                    if verbose:
                        print(f"  [Context] {context['previous_topic']}")

        # ============================================
        # Step 0: 检查是否需要拆解查询（Query Decomposition）
        # ============================================
        should_decompose, sub_queries = self.query_decomposer.decompose(query)

        if should_decompose:
            if verbose:
                print(f"\n  [🔍 Complex query detected. Decomposing into {len(sub_queries)} sub-queries:]")
                for i, sq in enumerate(sub_queries, 1):
                    print(f"      {i}. {sq}")

            context["sub_queries"] = sub_queries

            # 对每个子查询执行检索
            for i, sub_query in enumerate(sub_queries, 1):
                if verbose:
                    print(f"\n  [Sub-query {i}/{len(sub_queries)}] Searching: '{sub_query}'")

                observation = self._vector_search(sub_query, mode=mode)

                # 为每个子查询的observation打上标签
                observation["sub_query"] = sub_query
                observation["sub_query_index"] = i

                context["observations"].append(observation)
                context["actions"].append("search")

                if verbose:
                    print(f"      📊 Found {len(observation.get('sources', []))} chunks")

            # 拆解后的查询直接生成最终答案（跳过标准ReAct循环）
            context["final_answer"] = self._generate_final_answer(query, context)

            return {
                "answer": context["final_answer"],
                "sources": self._extract_all_sources(context),
                "react_steps": len(sub_queries),
                "reflections_count": 0,
                "thoughts": [f"Decomposed query into {len(sub_queries)} sub-queries"],
                "tools_used": ["search"],
                "was_decomposed": True,
                "sub_queries": sub_queries
            }

        # ============================================
        # 标准ReAct循环（不需要拆解的情况）
        # ============================================
        def run_with_timeout():
            loop_start = time.time()

            for iteration in range(1, self.max_iterations + 1):
                # Check if we've exceeded the timeout
                elapsed = time.time() - loop_start
                if elapsed > 120:
                    print(f"    [⚠️  ReAct loop timeout after {elapsed:.1f}s, forcing answer...]")
                    break

                if verbose:
                    print(f"\n  [Iteration {iteration}/{self.max_iterations}]")

                # === Thought ===
                thought = self._generate_thought(query, context, iteration)
                context["thoughts"].append(thought)

                if verbose:
                    print(f"  🧠 Thought: {thought}")

                # === Action ===
                action_decision = self._decide_action(thought, context)

                if action_decision["action"] == "answer":
                    # 直接回答
                    answer = self._generate_final_answer(query, context)
                    context["final_answer"] = answer

                    if verbose:
                        print(f"  ✅ Decision: Generate final answer")

                    break

                elif action_decision["action"] == "search":
                    # 执行检索
                    search_query = action_decision.get("query", query)

                    if verbose:
                        print(f"  🔍 Action: Search for '{search_query}'")

                    observation = self._vector_search(search_query, mode=mode)
                    context["actions"].append("search")
                    context["observations"].append(observation)

                    if verbose:
                        print(f"  📊 Observation: Found {len(observation.get('sources', []))} chunks")

                    # === Reflection: 评估检索质量 ===
                    reflection = self._reflect_on_search(query, observation, iteration)
                    context["reflections"].append(reflection)

                    if verbose:
                        print(f"  🔍 Reflection: {reflection['summary']}")

                    # 如果反思认为质量不好，调整策略继续搜索
                    if not reflection["is_satisfactory"] and iteration < self.max_iterations:
                        if verbose:
                            print(f"  ⚠️  Quality issue: {reflection['issue']}")
                            print(f"  🔄 Adjusting strategy for next search...")

            # 如果循环结束还没有答案，生成一个
            if not context["final_answer"]:
                context["final_answer"] = self._generate_final_answer(query, context)

            return {
                "answer": context["final_answer"],
                "sources": self._extract_all_sources(context),
                "react_steps": len(context["thoughts"]),
                "reflections_count": len(context["reflections"]),
                "thoughts": context["thoughts"] if verbose else [],
                "tools_used": list(set(context["actions"])) if context["actions"] else []
            }

        # Run with timeout protection
        try:
            return run_with_timeout()
        except Exception as e:
            print(f"    [⚠️  ReAct loop error: {e}]")
            # Fallback: return whatever we have
            return {
                "answer": context.get("final_answer") or self._generate_final_answer(query, context),
                "sources": self._extract_all_sources(context),
                "react_steps": len(context["thoughts"]),
                "reflections_count": len(context["reflections"]),
                "thoughts": context["thoughts"] if verbose else [],
                "tools_used": list(set(context["actions"])) if context["actions"] else []
            }

    def _generate_thought(self, query: str, context: Dict, iteration: int) -> str:
        """生成思考：分析当前情况（支持Look-Back机制和上下文注入）"""

        previous_topic = context.get("previous_topic", "")

        # 第一次迭代：总是搜索
        if iteration == 1:
            if previous_topic:
                # 【上下文注入】：告诉 Agent 上一轮的主题
                return f"I need to search for information about '{query}'. Note: {previous_topic}"
            else:
                return f"I need to search for information about {query}"

        # 后续迭代：基于反思调整策略
        if context["reflections"]:
            last_reflection = context["reflections"][-1]

            # 如果上次搜索质量不好，调整策略
            if not last_reflection.get("is_satisfactory", True):
                issue = last_reflection.get("issue", "")

                # Look-Back 机制：检查查询是否有代词或模糊词
                if any(pronoun in query.lower() for pronoun in ["it", "they", "this", "that"]):
                    # 优先使用 previous_topic（来自对话历史）
                    if previous_topic:
                        # 从主题中提取关键词
                        topic_words = previous_topic.split(":")[-1].strip()
                        refined_query = f"{topic_words} {query}"
                        return f"The query '{query}' contains pronouns. {previous_topic}. I should search for '{refined_query}' instead."

                    # 如果没有 previous_topic，从之前的搜索结果中提取关键词
                    if context["observations"]:
                        last_obs = context["observations"][-1]
                        last_answer = last_obs.get("answer", "")
                        keywords = self._extract_keywords(last_answer, top_n=3)
                        if keywords:
                            # 修正查询词
                            refined_query = f"{keywords[0]} {query}"
                            return f"The query '{query}' was too vague. Based on context, I should search for '{refined_query}' instead."

                if issue == "semantic_mismatch":
                    # 语义不匹配：使用查询扩展，并考虑 previous_topic
                    if previous_topic:
                        return f"The previous search found irrelevant content. {previous_topic}. Let me try searching for '{previous_topic.split(':')[-1].strip()} {query}'"
                    else:
                        expanded_query = self._expand_query(query, context)
                        return f"The previous search found irrelevant content. Let me try an expanded query: '{expanded_query}'"

                elif issue == "retrieval_failed":
                    # 检索完全失败：尝试查询扩展
                    expanded_query = self._expand_query(query, context)
                    return f"The search found no relevant results (score < 0.1). Let me try with expanded terms: '{expanded_query}'"

                elif issue == "no_info_found":
                    # 说没找到信息，尝试换个关键词
                    return f"The search didn't find relevant info. Let me try a more specific search for {query}"

                elif issue == "answer_too_short":
                    # 答案太短，可能需要更多上下文
                    return f"The previous answer was too brief. Let me search with more context about {query}"

                elif issue == "content_truncated":
                    # 内容被截断，尝试获取完整信息
                    return f"The content was incomplete. Let me search for the full definition of {query}"

                elif issue == "incomplete_comparison":
                    # 【新增】对比不完整：强制要求更全面的搜索
                    return f"The search results seem incomplete for a comparison task (too few points found). I need to search specifically for a 'full comparison table' or more detailed differences regarding {query}."

                else:
                    # 其他问题，重新搜索
                    return f"Previous search had quality issues. Let me try searching for {query} with a different approach"

        # 检查最后一次搜索的答案质量
        if context["observations"]:
            last_obs = context["observations"][-1]
            last_answer = last_obs.get("answer", "")

            # 如果有答案且长度足够，可以考虑回答
            if len(last_answer) > 100:
                return "I have sufficient information from the search to answer"

        # 默认：继续搜索
        return f"I need to search for more specific information about {query}"

    def _decide_action(self, thought: str, context: Dict) -> Dict:
        """决定下一步动作"""

        thought_lower = thought.lower()

        # 检查是否已经有足够信息
        if "sufficient information" in thought_lower or "can answer" in thought_lower:
            return {"action": "answer"}

        # 检查是否需要搜索
        if "need to search" in thought_lower or "search for" in thought_lower:
            # 提取搜索关键词
            match = re.search(r'search for\s+(.+?)(?:\.|$)', thought_lower, re.IGNORECASE)
            if match:
                search_query = match.group(1).strip()
                return {"action": "search", "query": search_query}
            return {"action": "search", "query": context["query"]}

        # 第一次迭代默认搜索
        if not context["observations"]:
            return {"action": "search", "query": context["query"]}

        # 默认回答
        return {"action": "answer"}

    def _reflect_on_search(self, query: str, observation: Dict, iteration: int) -> Dict:
        """
        反思：评估检索质量

        检查：
        1. 检索分数是否太低（< 0.1 表示检索失败）
        2. 答案长度是否足够
        3. 是否有碎片标记（"...", "ove..." 等）
        4. 是否包含"没有足够信息"的提示

        Returns:
            {
                "is_satisfactory": bool,
                "summary": str,
                "issue": str (if not satisfactory)
            }
        """
        # 检查 0: 检索分数太低（检索失败）
        top_scores = observation.get("top_scores", [])
        if top_scores:
            try:
                # Parse scores (they might be strings like "0.8364")
                max_score = max(float(s) for s in top_scores)
                if max_score < 0.1:
                    return {
                        "is_satisfactory": False,
                        "summary": f"Retrieval failed: max score {max_score:.4f} < 0.1",
                        "issue": "retrieval_failed"
                    }
            except (ValueError, TypeError):
                pass  # If score parsing fails, continue with other checks

        answer = observation.get("answer", "")

        # 检查 1: 答案太短
        if len(answer) < 50:
            return {
                "is_satisfactory": False,
                "summary": "Answer too short",
                "issue": "answer_too_short"
            }

        # 检查 2: 答案包含"没有足够信息"
        negative_phrases = [
            "does not contain enough information",
            "does not have enough information",
            "no information available",
            "context does not contain"
        ]
        for phrase in negative_phrases:
            if phrase.lower() in answer.lower():
                return {
                    "is_satisfactory": False,
                    "summary": "Answer says no info found",
                    "issue": "no_info_found"
                }

        # 检查 3: 答案有碎片标记
        fragmentation_markers = [
            "...",  # 省略号
            "ove...",  # overfitting 被截断
            "defin",  # definition 被截断
            "prevent",  # prevent 被截断
            "This is called",  # 后面应该有内容但断了
        ]
        for marker in fragmentation_markers:
            if marker in answer:
                # 检查是否在句子末尾（真正的截断）
                if answer.endswith(marker) or answer.endswith(marker + "."):
                    return {
                        "is_satisfactory": False,
                        "summary": "Content appears truncated",
                        "issue": "content_truncated"
                    }

        # 检查 4: 答案质量基于来源数量
        sources = observation.get("sources", [])
        if len(sources) == 0:
            return {
                "is_satisfactory": False,
                "summary": "No sources retrieved",
                "issue": "no_sources"
            }

        # 检查 5: 语义相关性校验（新增）
        # 避免用空气动力学回答数据切分问题
        semantic_check = self._check_semantic_relevance(query, observation)
        if not semantic_check["is_satisfactory"]:
            return semantic_check

        # =========================================================
        # 检查 6: 针对对比/列表类问题的完整性检查 (Completeness Check)
        # =========================================================
        is_comparison = any(w in query.lower() for w in ["compare", "difference", "vs", "versus", "distinction", "list", "types of"])

        if is_comparison:
            # 统计结构化标记：Markdown表格(|), 列表项(-, *)
            # 如果内容虽然长，但只是一大段废话，没有分点，对于对比题来说也是不合格的
            structure_score = answer.count("|") + answer.count("\n-") + answer.count("\n*") + answer.count("\n1.")

            # 阈值设定：
            # 1. 如果包含表格符号 '|' 少于 4 个（说明连表头都没有），且列表项少于 3 个
            # 2. 并且还没达到最大迭代次数（给它重试的机会）
            if structure_score < 3 and iteration < self.max_iterations:
                return {
                    "is_satisfactory": False,
                    "summary": f"Potential incomplete comparison (score: {structure_score})",
                    "issue": "incomplete_comparison"  # 新的 issue 类型
                }

        # 通过所有检查
        return {
            "is_satisfactory": True,
            "summary": f"Good quality: {len(answer)} chars, {len(sources)} sources"
        }

    def _generate_final_answer(self, query: str, context: Dict) -> str:
        """
        生成最终答案（智能回退策略）

        保证：Agentic RAG 的效果至少不低于 Enhanced RAG
        支持带标签的上下文拼接（用于查询拆解场景）
        """
        # 收集所有检索到的上下文（带标签的拼接）
        all_context = []
        chunk_idx = 1
        was_decomposed = context.get("sub_queries") is not None and len(context.get("sub_queries", [])) > 1

        for obs in context["observations"]:
            if "sources" in obs:
                # 检查是否是拆解查询的子查询结果
                sub_query_label = obs.get("sub_query", "")
                if was_decomposed and sub_query_label:
                    # 带标签的上下文拼接（帮助LLM理解来源）
                    all_context.append(f"\n## Results for: {sub_query_label}\n")

                # Use up to 5 chunks from each observation for more context
                for source in obs["sources"][:5]:
                    content = source.get("content", source)
                    # Clean up the content (remove excessive whitespace)
                    content = " ".join(content.split())[:500]  # Truncate very long chunks

                    if was_decomposed and sub_query_label:
                        # 带标签的chunk
                        all_context.append(f"[{chunk_idx}] {content}")
                    else:
                        # 普通chunk
                        all_context.append(f"[{chunk_idx}] {content}")
                    chunk_idx += 1

        context_str = "\n".join(all_context) if all_context else "No context available"

        # 【核心修改】检测对比类问题，注入强制完整性的指令
        is_comparison_query = any(w in query.lower() for w in ["compare", "difference", "vs", "versus", "distinction"])

        # 改进的Prompt - 支持对比类问题
        if was_decomposed:
            # 拆解查询的专用prompt
            prompt = f"""You are a helpful assistant. The user's question was broken down into multiple sub-queries, and the results from each are provided below.

User Question: {query}

The question was decomposed into:
{chr(10).join(f'- {sq}' for sq in context.get('sub_queries', []))}

Context from each sub-query:
{context_str}

Instructions:
1. Synthesize information from ALL sub-queries to answer the original question
2. For comparison questions, clearly state the differences between each aspect
3. Use all relevant information from the context
4. Be thorough and complete

Answer:"""
        else:
            # 标准prompt - 针对对比类问题加强指令
            if is_comparison_query:
                # 对比类问题的强化 prompt
                prompt = f"""Based on the following retrieved context, answer the user's question.

User Question: {query}

Context:
{context_str}

Instructions:
1. Answer the question using ONLY the provided context.
2. If the context doesn't contain enough information, say so.
3. Be concise and direct.
4. Include specific details from the context when relevant.

CRITICAL RULES FOR COMPARISON/LISTS:
- If the user asks to COMPARE items (e.g., "difference", "vs"), you MUST list ALL differences found in the context.
- Do NOT summarize or pick just one point. Be comprehensive.
- If the context contains a TABLE (marked by '|'), please reconstruct the table in your answer or list every row clearly.

Answer:"""
            else:
                # 普通问题的标准 prompt
                prompt = f"""You are a helpful assistant. Based on the following retrieved context, answer the user's question thoroughly.

User Question: {query}

Context:
{context_str}

Instructions:
1. Use the provided context to answer the question
2. If the context contains relevant code examples, include them
3. If the context doesn't contain enough information to fully answer, still provide what you can from the context
4. Be clear and concise, but complete
5. Don't say "the context doesn't mention" - just use what's available

Answer:"""

        # Try Agent generation with increased token limit
        agent_answer = ""
        try:
            headers = {"Authorization": f"Bearer {self.glm_key}"}
            payload = {
                "model": RAGConfig.CHAT_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 2000,  # Increased from 1000 to allow longer answers
                "temperature": 0.7
            }
            res = requests.post(self.api_url, headers=headers, json=payload, timeout=60)  # 60s timeout for generation
            res.raise_for_status()
            data = res.json()

            if "choices" in data:
                agent_answer = data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            print(f"    [Agent generation error: {e}]")

        # ============================================================
        # 智能回退逻辑：保证不低于 Enhanced RAG 的质量
        # ============================================================

        # Define what's unsatisfactory
        is_agent_unsatisfactory = (
            not agent_answer or                                    # Empty answer
            len(agent_answer) < 50 or                               # Too short (lowered from 100)
            "does not contain enough information" in agent_answer.lower() or  # Refusal
            "context does not contain" in agent_answer.lower()      # Refusal (new check)
        )

        if is_agent_unsatisfactory:
            print(f"    [⚠️  Agent answer unsatisfactory ({len(agent_answer)} chars). Using Pipeline fallback...")

            # Check if we have Enhanced RAG (Pipeline) answer available
            if context["observations"]:
                # Get the best pipeline answer (last observation is most recent)
                pipeline_answer = context["observations"][-1].get("answer", "")

                # Fallback if pipeline answer is reasonable (lowered threshold from 150 to 50)
                if len(pipeline_answer) > 50:
                    print(f"    [✅ Fallback to Pipeline: {len(pipeline_answer)} chars]")
                    return pipeline_answer

        # If Agent answer is satisfactory, use it
        if agent_answer:
            return agent_answer

        # Final fallback
        return "Sorry, I couldn't find relevant information to answer this question."

    def _direct_query(self, query: str, mode: str) -> Dict:
        """直接查询（不使用 ReAct）"""
        result = self.pipeline.run(query, mode=mode)

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("retrieved_chunks", []),
            "react_steps": 0,
            "thoughts": [],
            "tools_used": ["vector_search"]
        }

    def _vector_search(self, query: str, mode: str = "hybrid") -> Dict:
        """向量检索工具"""
        result = self.pipeline.run(query, mode=mode)

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("retrieved_chunks", []),
            "method": result.get("method", "")
        }

    def _bm25_search(self, query: str, **kwargs) -> Dict:
        """BM25 检索工具"""
        result = self.pipeline.run(query, mode="bm25_only")

        return {
            "answer": result.get("answer", ""),
            "sources": result.get("retrieved_chunks", []),
            "method": result.get("method", "")
        }

    def _extract_all_sources(self, context: Dict) -> List:
        """提取所有来源"""
        all_sources = []
        seen = set()

        for obs in context["observations"]:
            if "sources" in obs:
                for source in obs["sources"]:
                    # 去重
                    source_id = source.get("chunk_id", id(source))
                    if source_id not in seen:
                        seen.add(source_id)
                        all_sources.append(source)

        return all_sources

    def _get_conversation_history(self, k: int = 3) -> List[Dict]:
        """获取对话历史"""
        if not self.memory:
            return []

        memories = self.memory.short_term.get_recent(k)
        return [
            {
                "query": m.query,
                "answer": m.answer
            }
            for m in memories
        ]

    def _save_to_memory(self, original_query: str, rewritten_query: str,
                       answer: str, tools_used: List[str], importance: float):
        """保存到记忆"""
        # 获取 embedding
        try:
            embedding = self.pipeline.qianfan_client.embed([rewritten_query + " " + answer])
            if len(embedding) > 0:
                answer_embedding = embedding[0]
            else:
                answer_embedding = None
        except:
            answer_embedding = None

        # 保存到短期记忆
        self.memory.add_memory(
            query=original_query,  # 保存原始查询
            answer=answer,
            tools_used=tools_used,
            embedding=answer_embedding,
            importance=importance
        )

        # 高重要性存入长期记忆
        if importance >= 0.7 and answer_embedding is not None:
            self.memory.add_memory(
                query=original_query,
                answer=answer,
                tools_used=tools_used,
                embedding=answer_embedding,
                importance=importance,
                to_long_term=True
            )

    # ========== 便捷方法 ==========

    def chat(self, message: str, **kwargs) -> str:
        """简化的聊天接口"""
        result = self.query(message, verbose=False, **kwargs)
        return result["answer"]

    def get_memory_stats(self) -> Dict:
        """获取记忆统计"""
        if not self.memory:
            return {"memory_enabled": False}

        stats = self.memory.get_stats()
        stats["memory_enabled"] = True
        return stats

    def clear_memory(self):
        """清空短期记忆"""
        if self.memory:
            self.memory.short_term.clear()

    def get_conversation_history(self, k: int = 5) -> List[Dict]:
        """获取对话历史"""
        if not self.memory:
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
# 便捷接口
# ============================================
def create_react_agent(pipeline: RAGPipeline, glm_key: str,
                      enable_memory: bool = True,
                      enable_react: bool = True) -> ReActAgent:
    """创建 ReAct Agent 的便捷函数"""
    return ReActAgent(
        pipeline=pipeline,
        glm_key=glm_key,
        enable_memory=enable_memory,
        enable_react=enable_react
    )
