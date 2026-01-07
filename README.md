# HKU G.RAG: Production RAG Pipeline with Multi-Modal Parsing

**A lightweight, API-based RAG system for academic courseware with hybrid retrieval and comprehensive evaluation**

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)

</div>

---

## Overview

**HKU G.RAG** is a production-ready Retrieval-Augmented Generation system designed for university courseware. It combines multi-modal PDF parsing, hybrid retrieval (BM25 + Dense), and LLM-based evaluation using the Ragas framework.

### Key Features

- **Multi-Modal PDF Parsing**: Camelot (tables) + ParseBlock (sliding windows) + Code detection
- **Hybrid Retrieval**: BM25 + Dense embedding with Reciprocal Rank Fusion (RRF)
- **Reranker**: Qwen3-Reranker-8B via Baidu Qianfan API
- **Vector Caching**: Hash-based persistence for fast restart
- **Ragas Evaluation**: LLM-based metrics (Context Recall, Faithfulness, etc.)

### Performance

| Metric | Naive Parse | Enhanced Parse | Improvement |
|--------|-------------|----------------|-------------|
| Chunks | 555 | 1,956 | +252% |
| Avg Chunk Length | 267 chars | 512+ chars | +92% |
| Table Extraction | ❌ | ✅ (442 tables) | - |
| Code Blocks | ❌ | ✅ (detected) | - |
| Cross-Page Context | ❌ | ✅ (297 sliding windows) | - |

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Evaluation](#evaluation)
- [API Reference](#api-reference)
- [Configuration](#configuration)

---

## Installation

### Prerequisites

- Python 3.10+
- API Keys:
  - Baidu Qianfan (for embedding & reranker)
  - GLM-4 (for answer generation)
  - Optional: DeepSeek/OpenAI (for Ragas evaluation)

### Setup

```bash
# Clone repository
git clone https://github.com/your-username/hKu.G.rag.git
cd hKu.G.rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### API Configuration

Set environment variables:

```bash
# Required for RAG pipeline
export QIANFAN_API_KEY="your-qianfan-key"
export GLM_API_KEY="your-glm-key"

# Optional: For Ragas evaluation
export RAGAS_EVAL_LLM_API_KEY="your-deepseek-key"
export RAGAS_EVAL_LLM_BASE_URL="https://api.deepseek.com"
export RAGAS_EVAL_LLM_MODEL="deepseek-chat"
```

---

## Quick Start

### 1. Parse PDF Documents

```bash
# Enhanced parsing (recommended)
python -m src.parser.enhanced_parser \
    --input data/slides/course.pdf \
    --output data/parsed/enhanced_chunks.json

# Baseline parsing (for comparison)
python -m src.parser.baseline_parser \
    --input data/slides/course.pdf \
    --output data/parsed/baseline_chunks.json
```

### 2. Run RAG Pipeline

```bash
# Basic evaluation (5 queries × 3 systems)
python src/rag/evaluate.py

# Ragas-based evaluation (comprehensive)
python src/rag/ragas_eval.py
```

### 3. Interactive Query

```python
from src.rag.pipeline import RAGPipeline, Document
import json

# Load documents
with open("data/parsed/enhanced_chunks.json") as f:
    chunks = json.load(f)
documents = [Document(**c) for c in chunks]

# Initialize pipeline
pipeline = RAGPipeline(
    documents=documents,
    qianfan_key="your-key",
    glm_key="your-key"
)

# Query
result = pipeline.run("What is overfitting in machine learning?")
print(result["answer"])
```

---

## Project Structure

```
hKu.G.rag/
├── README.md                          # This file
├── requirements.txt                   # Dependencies
│
├── data/
│   ├── cache/                        # Vector embeddings (auto-generated)
│   │   └── vectors_*.npy
│   ├── parsed/                       # Parsed chunks
│   │   ├── enhanced_chunks.json      # Enhanced parser output (1,956 chunks)
│   │   └── baseline_chunks.json      # Baseline parser output (555 chunks)
│   └── evaluation/                   # Evaluation results
│       ├── final_15_answers_fixed.json
│       └── ragas_results.json
│
├── src/
│   ├── parser/
│   │   └── enhanced_parser.py        # Multi-modal PDF parser
│   ├── rag/
│   │   ├── pipeline.py               # RAG pipeline (RRF + Reranker)
│   │   ├── evaluate.py               # Basic evaluation (5 queries)
│   │   └── ragas_eval.py             # Ragas-based evaluation
│   └── evaluation/                   # (deprecated, use src/rag/)
│
├── v2/                                # Reference implementations
└── models/                            # Local model storage (gitignored)
```

---

## Evaluation

### Basic Evaluation

The `evaluate.py` script runs 5 test queries across 3 system configurations:

1. **Naive Parse + Naive RAG**: Baseline chunks, dense retrieval only
2. **Enhanced Parse + Naive RAG**: Enhanced chunks, dense retrieval only
3. **Enhanced Parse + Enhanced RAG**: Enhanced chunks, hybrid retrieval + reranker

```bash
python src/rag/evaluate.py
```

**Output**: `data/evaluation/final_15_answers_fixed.json`

### Ragas Evaluation

For comprehensive, LLM-based evaluation using the Ragas framework:

```bash
# Set evaluation LLM (DeepSeek recommended)
export RAGAS_EVAL_LLM_API_KEY="your-deepseek-key"
export RAGAS_EVAL_LLM_BASE_URL="https://api.deepseek.com"
export RAGAS_EVAL_LLM_MODEL="deepseek-chat"

# Run evaluation
python src/rag/ragas_eval.py
```

**Metrics**:
- **Context Precision**: How relevant is the retrieved context?
- **Context Recall**: Does the context cover the reference answer?
- **Faithfulness**: Is the generated answer faithful to the context?
- **Answer Relevancy**: Is the answer relevant to the query?

**Output**: `data/evaluation/ragas_results.json`

---

## API Reference

### RAGPipeline

```python
class RAGPipeline:
    def __init__(self, documents: List[Document], qianfan_key: str, glm_key: str):
        """
        Initialize RAG pipeline.

        Args:
            documents: List of Document objects
            qianfan_key: Baidu Qianfan API key (embedding & reranker)
            glm_key: GLM-4 API key (generation)
        """

    def run(self, query: str, mode: str = 'hybrid') -> Dict:
        """
        Run RAG pipeline.

        Args:
            query: User query
            mode: 'dense_only' or 'hybrid'
                  - 'dense_only': Cosine similarity only
                  - 'hybrid': BM25 + Dense + Reranker

        Returns:
            {
                "method": str,
                "top_scores": List[str],
                "retrieved_chunks": List[Dict],
                "answer": str
            }
        """
```

### Document

```python
@dataclass
class Document:
    content: str          # Text content
    doc_id: str          # Unique identifier
    metadata: Dict = None # Additional metadata (source_type, page_numbers, etc.)
```

---

## Configuration

### RAGConfig

```python
class RAGConfig:
    # Models
    EMBEDDING_MODEL = "qwen3-embedding-4b"    # Qianfan embedding model
    RERANK_MODEL = "qwen3-reranker-8b"         # Qianfan reranker model
    CHAT_MODEL = "glm-4"                       # GLM generation model

    # Caching
    CACHE_DIR = "data/cache"
    VECTOR_CACHE_PATH = "data/cache/vectors.npy"
```

### BM25 Parameters

```python
class BM25Retriever:
    k1 = 1.5    # Term saturation parameter
    b = 0.75    # Length normalization parameter
```

### RRF Parameters

```python
# In hybrid_retrieve()
k_const = 60  # RRF constant (higher = smoother rank fusion)

# Dynamic weighting based on query type
if is_keyword_query:
    bm25_weight, dense_weight = 0.7, 0.3
else:
    bm25_weight, dense_weight = 0.3, 0.7
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
  title={HKU G.RAG: Production RAG Pipeline with Multi-Modal Parsing},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/hKu.G.rag}
}
```

---

## License

Apache License 2.0

---

## Acknowledgments

- **Qwen Team** for Qwen3-Embedding and Qwen3-Reranker models
- **Baidu Qianfan** for API access to embedding and reranking services
- **GLM (Zhipu AI)** for the API access to generation
- **Ragas** for the evaluation framework

---

## Contact

For questions and feedback, please open an issue on GitHub.
