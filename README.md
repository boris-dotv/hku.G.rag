# HKU G.RAG: Agentic RAG with Multi-Stage Reasoning

**Production RAG system with ReAct Loop, Query Rewriting, and Comprehensive Evaluation**

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

</div>

---

## Overview

**HKU G.RAG** is an advanced Retrieval-Augmented Generation system featuring **Agentic RAG** with multi-stage reasoning. It combines multi-modal PDF parsing, hybrid retrieval, and intelligent agents that can:

- **Query Rewriting**: Resolve pronouns and implicit references from conversation context
- **ReAct Loop**: Thought → Action → Observation → Reflection reasoning cycle
- **Query Decomposition**: Break complex queries into sub-queries using LLM
- **Look-Back Mechanism**: Extract keywords from history to refine failed searches
- **Semantic Relevance Check**: Detect when retrieval results are semantically irrelevant
- **Smart Fallback**: Guarantee quality floor (Agentic RAG ≥ Enhanced RAG)
- **Submodular Optimization**: Remove redundant chunks for diversity

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Modal Parsing** | Camelot (tables) + ParseBlock (sliding windows) + Code detection |
| **Hybrid Retrieval** | BM25 + Dense embedding with RRF fusion |
| **Reranker** | Qwen3-Reranker-8B via Baidu Qianfan API |
| **ReAct Agent** | Multi-stage reasoning with reflection and self-correction |
| **Query Rewriting** | LLM-based pronoun resolution for multi-turn conversations |
| **Query Decomposition** | Break complex/comparison queries into sub-queries |
| **Semantic Check** | Detect irrelevant retrieval (e.g., aerodynamics for data questions) |
| **Smart Fallback** | Guaranteed answer quality with pipeline fallback |
| **Memory System** | Short-term (10 turns) + Long-term (vector DB) |
| **Submodular Optimization** | Remove redundant chunks using similarity clustering |

### Performance Comparison

| System | Query Rewriting | Multi-Hop | Code Support | Quality Floor |
|--------|----------------|-----------|--------------|---------------|
| **Naive RAG** | ❌ | ❌ | Basic | Low |
| **Enhanced RAG** | ❌ | Limited | Good | Medium |
| **Agentic RAG** | ✅ | ✅ | Excellent | High (≥ Enhanced) |

**Parsing Performance**:
- Chunks: 555 → 1,956 (+252%)
- Tables extracted: 442
- Code blocks detected: Yes
- Cross-page context: 297 sliding windows

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Agentic RAG Features](#agentic-rag-features)
- [Evaluation Results](#evaluation-results)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)

---

## Installation

### Prerequisites

- Python 3.10+
- API Keys:
  - Baidu Qianfan (embedding & reranker)
  - GLM-4 (answer generation & agent reasoning)

```bash
# Clone repository
git clone https://github.com/boris-dotv/hku.G.rag.git
cd hku.G.rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### API Configuration

```bash
# Required
export QIANFAN_API_KEY="your-qianfan-key"
export GLM_API_KEY="your-glm-key"
```

---

## Quick Start

### 1. Run Agentic RAG Demo

```python
from src.agent import create_react_agent
from src.rag.pipeline import RAGPipeline, Document
import json

# Load documents
with open("data/parsed/enhanced_chunks.json") as f:
    chunks = json.load(f)
documents = [Document(**c) for c in chunks]

# Initialize pipeline
pipeline = RAGPipeline(documents,
                       qianfan_key="your-key",
                       glm_key="your-key")

# Create ReAct Agent with all features
agent = create_react_agent(
    pipeline=pipeline,
    glm_key="your-glm-key",
    enable_memory=True,
    enable_react=True
)

# Simple query
result = agent.query("What is overfitting?")
print(result["answer"])

# Follow-up with pronoun (automatic rewriting)
result = agent.query("How can it be prevented?")
# Agent rewrites to: "How can overfitting be prevented?"
print(result["answer"])
```

### 2. Run Comprehensive Evaluation

```bash
# Compare 3 systems × 6 queries with full Chain of Thought
python generate_full_comparison.py

# Output: data/evaluation/comprehensive_comparison.json
# Contains: Naive RAG vs Enhanced RAG vs Agentic RAG
```

---

## Agentic RAG Features

### Query Rewriting

Automatically resolves pronouns and implicit references:

```python
# Q3: "What is overfitting?"
# Q4: "How can it be prevented?"

# Agent rewrites Q4 → "How can overfitting be prevented?"
```

**Implementation**: `QueryRewriter` in `src/agent/react_agent.py`

### ReAct Loop

Multi-stage reasoning with reflection:

```
Iteration 1:
  Thought: I need to search for information about overfitting prevention
  Action: Search (vector retrieval)
  Observation: Found 5 chunks
  Reflection: Good quality: 500 chars, 5 sources

Iteration 2:
  Thought: I have sufficient information
  Decision: Generate final answer
```

### Query Decomposition

Break complex queries into sub-queries:

```python
# Input: "Difference between DL and ML hardware?"
# Agent decomposes into:
#   1. "Hardware dependency of Deep Learning"
#   2. "Hardware dependency of Machine Learning"
#   3. "Comparison between DL and ML hardware"
```

### Look-Back Mechanism

When search fails, extract keywords from conversation history:

```python
# Failed search for: "How can it be prevented?"
# Look-Back: Extract "overfitting" from previous turn
# Refined search: "overfitting How can it be prevented?"
```

### Semantic Relevance Check

Detect irrelevant retrieval using LLM:

```python
# Prevents answering "aerodynamics" questions when user asks "data splitting"
# Checks: Does retrieved content actually answer the question?
```

### Smart Fallback

Guarantees Agentic RAG ≥ Enhanced RAG quality:

```python
# If Agent generation fails or is too short
# Automatically fallback to Enhanced RAG's answer
```

---

## Evaluation Results

### Comprehensive Comparison (6 Queries × 3 Systems)

Run `python generate_full_comparison.py` to generate:

**Systems Compared**:
1. Naive Parse + Naive RAG (dense retrieval only)
2. Enhanced Parse + Enhanced RAG (hybrid + reranker)
3. Enhanced Parse + Agentic RAG (ReAct + all features)

**Output**: `data/evaluation/comprehensive_comparison.json`

**Results Include**:
- Answer quality and length
- Chain of Thought (thoughts, reflections, react_steps)
- Query rewriting status
- Sources and scores
- Execution time

### Key Findings

| Query | Challenge | Naive | Enhanced | Agentic |
|-------|-----------|-------|----------|---------|
| Q1 (Table Extraction) | Complex comparison | 194 chars | 194 chars | 244 chars ✅ |
| Q2 (Cross-Page) | Multi-page context | 1362 chars | 2033 chars | 1175 chars ✅ |
| Q3 (Overfitting) | Definition | 439 chars | Failed | 1192 chars ✅ |
| Q4 (Prevention) | Pronoun resolution ("it") | 377 chars | 299 chars | **Target** 🔧 |
| Q5 (Multi-hop) | Multi-reasoning | Failed | 210 chars | 579 chars ✅ |
| Q6 (Code) | Code generation | 912 chars | 866 chars | **Full code** ✅ |

**🔧 = Requires query rewriting to work correctly**

---

## Project Structure

```
hku.G.rag/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
├── generate_full_comparison.py        # Comprehensive evaluation
├── demo_agent.py                      # Simple demo
│
├── data/
│   ├── cache/                        # Vector embeddings (auto-generated)
│   ├── memory/                       # Memory storage (gitignored)
│   ├── parsed/                       # Parsed chunks
│   │   ├── enhanced_chunks.json      # 1,956 chunks (tables + code + sliding windows)
│   │   └── baseline_chunks.json      # 555 chunks (baseline)
│   └── evaluation/                   # Evaluation results
│       └── comprehensive_comparison.json
│
├── src/
│   ├── agent/                        # Agentic RAG
│   │   ├── __init__.py               # Agent factory
│   │   ├── react_agent.py            # ReAct Agent with all features ✨ NEW
│   │   ├── agent.py                  # Legacy AgenticRAG
│   │   └── memory.py                 # Memory system (short-term + long-term)
│   ├── parser/
│   │   └── enhanced_parser.py        # Multi-modal PDF parser
│   └── rag/
│       ├── pipeline.py               # RAG pipeline (RRF + Reranker)
│       ├── submodular.py             # Diversity optimization ✨ NEW
│       └── ragas_comparison.py       # Ragas evaluation ✨ NEW
│
└── v2/                                # Reference implementations
```

---

## API Reference

### ReActAgent

```python
from src.agent import create_react_agent

agent = create_react_agent(
    pipeline: RAGPipeline,
    glm_key: str,
    enable_memory: bool = True,    # Enable memory system
    enable_react: bool = True,     # Enable ReAct loop
    max_iterations: int = 3        # Max ReAct iterations
)

# Query with full features
result = agent.query(
    user_query: str,
    mode: str = "hybrid",          # "dense_only" or "hybrid"
    use_react: bool = True,         # Use ReAct loop
    use_memory: bool = True,        # Use memory for query rewriting
    save_to_memory: bool = True,    # Save to memory
    verbose: bool = False          # Show Chain of Thought
)

# Returns:
# {
#     "answer": str,
#     "sources": List[Dict],
#     "react_steps": int,
#     "reflections_count": int,
#     "thoughts": List[str],          # Chain of Thought
#     "query_was_rewritten": bool,
#     "rewritten_query": str,
#     "tools_used": List[str]
# }
```

### Query Decomposition

```python
from src.agent.react_agent import QueryDecomposer

decomposer = QueryDecomposer(glm_key="your-key")

should_decompose, sub_queries = decomposer.decompose(
    "What are the differences between DL and ML?"
)

# Returns: (True, [
#     "Hardware dependency of Deep Learning",
#     "Hardware dependency of Machine Learning",
#     "Comparison between DL and ML"
# ])
```

### Submodular Optimization

```python
from src.rag.submodular import diversify_rag_results

# Remove redundant chunks
diverse_chunks = diversify_rag_results(
    retrieved_chunks=results["retrieved_chunks"],
    top_k=5,
    similarity_threshold=0.85  # Merge chunks with >85% similarity
)
```

---

## Dependencies

```
# Core
numpy>=1.24.0
requests>=2.31.0
jieba>=0.42.0

# PDF parsing
PyPDF2>=3.0.0
pdfplumber>=0.10.0
camelot-py[cv]>=0.11.0
opencv-python>=4.8.0

# Evaluation (optional)
ragas>=0.1.0
langchain-openai>=0.1.0
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@software{hku_g_rag,
  title={HKU G.RAG: Agentic RAG with Multi-Stage Reasoning},
  author={Your Name},
  year={2025},
  url={https://github.com/boris-dotv/hku.G.rag}
}
```

---

## License

Apache License 2.0

---

## Acknowledgments

- **Qwen Team** for Qwen3-Embedding and Qwen3-Reranker models
- **Baidu Qianfan** for API access
- **GLM (Zhipu AI)** for GLM-4 API
- **Ragas** for evaluation framework
- **ReAct Paper** for reasoning framework inspiration

---

## Contact

For questions and feedback, please open an issue on GitHub.
