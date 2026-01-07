"""
Ragas-based RAG Evaluation

This script uses the Ragas framework to evaluate RAG systems with LLM-based metrics:
- Context Recall: Measures retrieval quality
- Context Precision: Measures retrieval relevance
- Faithfulness: Measures answer grounding in context
- Answer Relevancy: Measures answer quality

Requires:
    pip install ragas langchain-openai
"""

import json
import os
import sys
sys.path.append('src/rag')

from pipeline import RAGPipeline, Document
from typing import List, Dict
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision
)
from langchain_openai import ChatOpenAI

# ============================================
# Configuration
# ============================================
QIANFAN_KEY = os.environ.get("QIANFAN_KEY", "bce-v3/ALTAK-dgZMQj7E5tByoRofFKlbM/e852481aaab5ebf3ffe6f2a50589e6e41646c127")
GLM_KEY = os.environ.get("GLM_KEY", "bfacda9a355fd3e9557063bc90aa8a1e.1zJ3JaydaKIkZ6Kq")

# For Ragas evaluation (using OpenAI-compatible LLM)
# You can use DeepSeek, Qianfan, or any OpenAI-compatible API
RAGAS_EVAL_LLM_API_KEY = os.environ.get("RAGAS_EVAL_LLM_API_KEY", "")
RAGAS_EVAL_LLM_BASE_URL = os.environ.get("RAGAS_EVAL_LLM_BASE_URL", "https://api.deepseek.com/v1")
RAGAS_EVAL_LLM_MODEL = os.environ.get("RAGAS_EVAL_LLM_MODEL", "deepseek-chat")

# ============================================
# Test Queries with Reference Answers
# ============================================
TEST_QUERIES = [
    {
        "query": "What is the difference between machine learning and deep learning?",
        "reference": "Machine learning is a broader subset of AI that uses various algorithms to learn from data, while deep learning is a specialized subset of machine learning that uses multi-layered neural networks. Deep learning typically requires more data and computational power but can automatically learn feature representations, whereas machine learning often requires manual feature engineering."
    },
    {
        "query": "How to calculate F1 Score from confusion matrix?",
        "reference": "F1 Score is calculated as: F1 = 2 * (Precision * Recall) / (Precision + Recall). From confusion matrix: Precision = TP / (TP + FP), Recall = TP / (TP + FN), where TP=True Positives, FP=False Positives, FN=False Negatives."
    },
    {
        "query": "What techniques can prevent overfitting in machine learning?",
        "reference": "Common techniques to prevent overfitting include: 1) Regularization (L1/L2), 2) Dropout (for neural networks), 3) Cross-validation, 4) Early stopping, 5) Data augmentation, 6) Reducing model complexity, 7) Ensemble methods, 8) Increasing training data."
    },
    {
        "query": "Explain the bagging method in ensemble learning",
        "reference": "Bagging (Bootstrap Aggregating) is an ensemble method that trains multiple models on different random subsets of training data (with replacement). Each model votes, and the majority prediction is selected. Random Forest is a popular example of bagging with decision trees."
    },
    {
        "query": "How does a decision tree make split decisions?",
        "reference": "Decision trees make split decisions using algorithms like ID3 (Information Gain), C4.5 (Gain Ratio), or CART (Gini Index). They evaluate each possible split point based on impurity measures and select the split that best separates classes or reduces variance."
    },
    {
        "query": "What is the purpose of cross-validation?",
        "reference": "Cross-validation is used to assess model performance and prevent overfitting by splitting data into training and validation sets multiple times. K-fold cross-validation divides data into K subsets, using K-1 for training and 1 for validation, rotating through all K combinations."
    },
    {
        "query": "Explain the bias-variance tradeoff",
        "reference": "The bias-variance tradeoff describes the balance between underfitting (high bias) and overfitting (high variance). High bias models are too simple and miss patterns, while high variance models are too complex and memorize noise. Optimal model complexity balances both to minimize total error."
    },
    {
        "query": "What is gradient descent in machine learning?",
        "reference": "Gradient descent is an optimization algorithm that minimizes a loss function by iteratively moving parameters in the direction of the negative gradient. Variants include batch gradient descent, stochastic gradient descent (SGD), and mini-batch gradient descent."
    },
    {
        "query": "How does support vector machine (SVM) work?",
        "reference": "SVM finds the optimal hyperplane that maximizes the margin between different classes. It uses kernel functions to transform data into higher dimensions for non-linear separation. Support vectors are the data points closest to the decision boundary."
    },
    {
        "query": "What is the difference between supervised and unsupervised learning?",
        "reference": "Supervised learning uses labeled data to learn input-output mappings (classification, regression), while unsupervised learning finds patterns in unlabeled data (clustering, dimensionality reduction). Supervised learning has explicit feedback through labels, while unsupervised learning discovers hidden structures."
    }
]

# ============================================
# Data Loading
# ============================================
def load_chunks(path: str) -> List[Document]:
    """Load and convert chunks to Document format"""
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


# ============================================
# RAG System Evaluation
# ============================================
def evaluate_rag_system(
    pipeline: RAGPipeline,
    test_queries: List[Dict],
    mode: str = 'hybrid',
    system_name: str = "RAG System"
) -> Dict:
    """
    Evaluate a RAG system using Ragas framework

    Args:
        pipeline: RAGPipeline instance
        test_queries: List of {"query": str, "reference": str}
        mode: 'dense_only' or 'hybrid'
        system_name: Name of the system being evaluated

    Returns:
        Ragas evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {system_name}")
    print(f"{'='*80}")

    # Collect evaluation data
    eval_data = []

    for item in test_queries:
        query = item["query"]
        reference = item["reference"]

        # Run RAG pipeline
        result = pipeline.run(query, mode=mode)

        # Extract context from retrieved chunks
        contexts = [[chunk["content"]] for chunk in result["retrieved_chunks"]]

        eval_data.append({
            "user_input": query,
            "retrieved_contexts": contexts,
            "response": result["answer"],
            "reference": reference
        })

        print(f"  Query: {query[:60]}...")
        print(f"    Top score: {result['top_scores'][0]}")

    # Create Dataset for Ragas
    dataset = Dataset.from_list(eval_data)

    # Initialize evaluator
    llm = ChatOpenAI(
        model=RAGAS_EVAL_LLM_MODEL,
        api_key=RAGAS_EVAL_LLM_API_KEY,
        base_url=RAGAS_EVAL_LLM_BASE_URL,
        temperature=0.0
    )

    # Run evaluation
    print(f"\n  Running Ragas evaluation...")
    results = evaluate(
        dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ],
        llm=llm
    )

    return results


# ============================================
# Main Evaluation Function
# ============================================
def main():
    print("="*80)
    print("RAGAS-BASED RAG EVALUATION")
    print("="*80)
    print(f"\nTest Queries: {len(TEST_QUERIES)}")
    print(f"Evaluation Model: {RAGAS_EVAL_LLM_MODEL}")

    # Load chunks
    print("\nLoading chunks...")
    naive_docs = load_chunks("data/parsed/baseline_chunks.json")
    enhanced_docs = load_chunks("data/parsed/enhanced_chunks.json")
    print(f"  Naive Parse: {len(naive_docs)} chunks")
    print(f"  Enhanced Parse: {len(enhanced_docs)} chunks")

    # Initialize pipelines
    print("\nInitializing pipelines...")
    sys1_pipe = RAGPipeline(naive_docs, QIANFAN_KEY, GLM_KEY)
    sys2_pipe = RAGPipeline(enhanced_docs, QIANFAN_KEY, GLM_KEY)
    sys3_pipe = RAGPipeline(enhanced_docs, QIANFAN_KEY, GLM_KEY)
    print("  ✓ Pipelines ready")

    # Evaluate each system
    all_results = {}

    # System 1: Naive Parse + Naive RAG
    all_results["system1"] = evaluate_rag_system(
        sys1_pipe,
        TEST_QUERIES,
        mode='dense_only',
        system_name="Naive Parse + Naive RAG"
    )

    # System 2: Enhanced Parse + Naive RAG
    all_results["system2"] = evaluate_rag_system(
        sys2_pipe,
        TEST_QUERIES,
        mode='dense_only',
        system_name="Enhanced Parse + Naive RAG"
    )

    # System 3: Enhanced Parse + Enhanced RAG
    all_results["system3"] = evaluate_rag_system(
        sys3_pipe,
        TEST_QUERIES,
        mode='hybrid',
        system_name="Enhanced Parse + Enhanced RAG"
    )

    # Print results
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)

    system_names = [
        "Naive Parse + Naive RAG",
        "Enhanced Parse + Naive RAG",
        "Enhanced Parse + Enhanced RAG"
    ]

    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    print(f"\n{'System':<40} {'Context Precision':<18} {'Context Recall':<16} {'Faithfulness':<14} {'Answer Relevancy'}")
    print("-" * 110)

    for i, (sys_key, sys_name) in enumerate(zip(["system1", "system2", "system3"], system_names)):
        result = all_results[sys_key]
        scores = [result.get(metric, 0) for metric in metrics]
        print(f"{sys_name:<40} {scores[0]:<18.4f} {scores[1]:<16.4f} {scores[2]:<14.4f} {scores[3]:.4f}")

    # Save results
    output = {
        "evaluation_type": "Ragas-based Evaluation",
        "test_queries": len(TEST_QUERIES),
        "evaluator_model": RAGAS_EVAL_LLM_MODEL,
        "systems": system_names,
        "metrics": metrics,
        "results": {
            sys_key: {
                metric: all_results[sys_key].get(metric, 0)
                for metric in metrics
            }
            for sys_key in ["system1", "system2", "system3"]
        }
    }

    os.makedirs("data/evaluation", exist_ok=True)
    with open("data/evaluation/ragas_results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: data/evaluation/ragas_results.json")
    print("\n" + "="*80)


if __name__ == "__main__":
    # Check if Ragas is installed
    try:
        import ragas
    except ImportError:
        print("Error: Ragas is not installed.")
        print("Please install: pip install ragas")
        print("Also install: pip install langchain-openai")
        sys.exit(1)

    # Check if API key is set
    if not RAGAS_EVAL_LLM_API_KEY:
        print("Error: RAGAS_EVAL_LLM_API_KEY is not set.")
        print("Please set the environment variable:")
        print("  export RAGAS_EVAL_LLM_API_KEY='your-api-key'")
        print("Or use DeepSeek:")
        print("  export RAGAS_EVAL_LLM_API_KEY='your-deepseek-key'")
        print("  export RAGAS_EVAL_LLM_BASE_URL='https://api.deepseek.com'")
        print("  export RAGAS_EVAL_LLM_MODEL='deepseek-chat'")
        sys.exit(1)

    main()
