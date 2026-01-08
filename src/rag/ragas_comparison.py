"""
Ragas Evaluation for RAG System Comparison

Evaluates 3 systems using Ragas metrics:
- Context Precision: 检索上下文的相关性
- Context Recall: 上下文是否覆盖参考答案
- Faithfulness: 答案是否忠实于上下文
- Answer Relevancy: 答案与问题的相关性
"""

import sys
import os
import json
from datetime import datetime
from typing import List, Dict

sys.path.append('src')

# Ragas imports
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)
from ragas.dataset import Dataset
from langchain_openai import ChatOpenAI


def load_comparison_results(path: str) -> Dict:
    """加载对比结果"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def prepare_ragas_data(comparison_results: Dict) -> Dict[str, List[Dict]]:
    """
    将对比结果转换为 Ragas 格式

    Returns:
        {
            "naive": [{"question": ..., "answer": ..., "contexts": [...]}],
            "enhanced": [...],
            "agentic": [...]
        }
    """
    systems_data = {
        "naive": [],
        "enhanced": [],
        "agentic": []
    }

    for result in comparison_results["results"]:
        query = result["query"]

        for sys_result in result["results"]:
            system_name = sys_result["system"]
            answer = sys_result.get("answer", "")

            # 提取 contexts
            contexts = []
            for chunk in sys_result.get("retrieved_chunks", []):
                content = chunk.get("content", "")
                contexts.append(content)

            # 映射到标准名称
            if "Naive Parse" in system_name:
                systems_data["naive"].append({
                    "question": query,
                    "answer": answer,
                    "contexts": contexts
                })
            elif "Enhanced Parse + Enhanced RAG" in system_name:
                systems_data["enhanced"].append({
                    "question": query,
                    "answer": answer,
                    "contexts": contexts
                })
            elif "Agentic RAG" in system_name:
                systems_data["agentic"].append({
                    "question": query,
                    "answer": answer,
                    "contexts": contexts
                })

    return systems_data


def run_ragas_evaluation(
    data: List[Dict],
    system_name: str,
    eval_llm_api_key: str,
    eval_llm_base_url: str = "https://api.deepseek.com",
    eval_llm_model: str = "deepseek-chat"
) -> Dict:
    """
    运行 Ragas 评估

    Args:
        data: 评估数据 [{"question", "answer", "contexts"}]
        system_name: 系统名称
        eval_llm_api_key: 评估 LLM API key
        eval_llm_base_url: 评估 LLM base URL
        eval_llm_model: 评估 LLM 模型

    Returns:
        评估结果
    """
    if not data:
        return {"error": "No data to evaluate"}

    print(f"\n{'='*60}")
    print(f"  Evaluating: {system_name}")
    print(f"{'='*60}")
    print(f"  Queries: {len(data)}")

    # 设置评估 LLM
    os.environ["OPENAI_API_KEY"] = eval_llm_api_key
    os.environ["OPENAI_API_BASE"] = eval_llm_base_url

    # 创建 Ragas Dataset
    try:
        dataset = Dataset.from_list(data)

        # 运行评估
        results = evaluate(
            dataset,
            metrics=[
                context_precision,
                context_recall,
                faithfulness,
                answer_relevancy
            ],
            llm=ChatOpenAI(
                api_key=eval_llm_api_key,
                base_url=eval_llm_base_url,
                model=eval_llm_model,
                temperature=0.1
            )
        )

        # 转换结果为字典
        results_dict = results.to_pandas().to_dict('records')

        # 计算平均分数
        scores = {}
        for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
            values = [r.get(metric, 0) for r in results_dict if not isinstance(r.get(metric), str)]
            if values:
                scores[metric] = {
                    "mean": sum(values) / len(values),
                    "values": values
                }

        return {
            "system": system_name,
            "num_queries": len(data),
            "scores": scores,
            "details": results_dict
        }

    except Exception as e:
        return {
            "system": system_name,
            "error": str(e)
        }


def save_evaluation_results(all_results: Dict, output_path: str):
    """保存评估结果"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to: {output_path}")


def print_summary(all_results: Dict):
    """打印评估摘要"""
    print("\n" + "="*70)
    print("  Ragas Evaluation Summary")
    print("="*70)

    for system_result in all_results["systems"]:
        system_name = system_result["system"]
        print(f"\n【{system_name}】")

        if "error" in system_result:
            print(f"  Error: {system_result['error']}")
            continue

        scores = system_result.get("scores", {})

        print(f"  Context Precision:  {scores.get('context_precision', {}).get('mean', 0):.3f}")
        print(f"  Context Recall:     {scores.get('context_recall', {}).get('mean', 0):.3f}")
        print(f"  Faithfulness:       {scores.get('faithfulness', {}).get('mean', 0):.3f}")
        print(f"  Answer Relevancy:   {scores.get('answer_relevancy', {}).get('mean', 0):.3f}")

    # 对比
    print("\n" + "="*70)
    print("  Comparison")
    print("="*70)

    metrics = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]

    for metric in metrics:
        print(f"\n{metric.replace('_', ' ').title()}:")
        for system_result in all_results["systems"]:
            system_name = system_result["system"]
            if "scores" in system_result:
                score = system_result["scores"].get(metric, {}).get("mean", 0)
                print(f"  {system_name}: {score:.3f}")


def main():
    print("="*70)
    print("  Ragas Evaluation for RAG Systems")
    print("="*70)
    print(f"\n  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # API Keys
    eval_llm_api_key = os.environ.get("RAGAS_EVAL_LLM_API_KEY")
    eval_llm_base_url = os.environ.get("RAGAS_EVAL_LLM_BASE_URL", "https://api.deepseek.com")
    eval_llm_model = os.environ.get("RAGAS_EVAL_LLM_MODEL", "deepseek-chat")

    if not eval_llm_api_key:
        print("\n⚠️  Warning: RAGAS_EVAL_LLM_API_KEY not set")
        print("   Using default key (may not work)")
        eval_llm_api_key = "sk-your-deepseek-key"

    print(f"\n  Evaluation LLM: {eval_llm_model}")
    print(f"  Base URL: {eval_llm_base_url}")

    # 加载对比结果
    comparison_path = "data/evaluation/comprehensive_comparison.json"
    print(f"\n[1/3] Loading comparison results from: {comparison_path}")

    if not os.path.exists(comparison_path):
        print(f"  Error: File not found: {comparison_path}")
        print("  Please run comprehensive_comparison.py first")
        return

    comparison_results = load_comparison_results(comparison_path)
    print(f"  Loaded {len(comparison_results['results'])} query results")

    # 准备 Ragas 数据
    print("\n[2/3] Preparing data for Ragas evaluation...")
    systems_data = prepare_ragas_data(comparison_results)

    for name, data in systems_data.items():
        print(f"  {name}: {len(data)} queries")

    # 运行评估
    print("\n[3/3] Running Ragas evaluation...")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "evaluation_config": {
            "llm_model": eval_llm_model,
            "base_url": eval_llm_base_url
        },
        "systems": []
    }

    for system_name, data in systems_data.items():
        result = run_ragas_evaluation(
            data,
            system_name.upper(),
            eval_llm_api_key,
            eval_llm_base_url,
            eval_llm_model
        )
        all_results["systems"].append(result)

    # 保存结果
    output_path = "data/evaluation/ragas_comparison_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    save_evaluation_results(all_results, output_path)

    # 打印摘要
    print_summary(all_results)

    print("\n" + "="*70)
    print("  Evaluation Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
