"""
PDF Parsing Benchmark for Chunk Quality Evaluation

Evaluates parsing quality along three dimensions:
1. Table Preservation: How well tables are extracted
2. Semantic Coherence: Are chunks semantically complete?
3. No Fragmentation: Are concepts split across chunks?
4. Content Coverage: How much content is captured vs baseline

Target: Compare enhanced parser (Camelot + ParseBlock) vs baseline (pdfplumber only)
"""

import json
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import requests


@dataclass
class BenchmarkResult:
    """Results from parsing benchmark"""
    parser_name: str
    total_chunks: int
    table_preservation: float  # F1 score for table extraction
    semantic_coherence: float  # LLM-rated 1-10, normalized to 0-1
    no_fragmentation: float    # % of complete concepts
    content_coverage: float    # % of content captured vs baseline
    overall_score: float       # Weighted average


class LLMEvaluator:
    """
    Use LLM API to evaluate chunk quality

    This provides a more nuanced assessment than pure heuristic metrics
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def call_api(self, messages: List[Dict], temperature: float = 0.3, max_retries: int = 3) -> str:
        """Call GLM-4.7 API with retry logic"""
        import time

        for attempt in range(max_retries):
            try:
                payload = {
                    "model": "glm-4.7",
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 1024
                }

                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                }

                response = requests.post(self.api_url, headers=headers, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]

            except (requests.RequestException, requests.Timeout, KeyError) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    print(f"    API error, retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    print(f"    API failed after {max_retries} attempts: {e}")
                    raise

    def evaluate_semantic_coherence(self, chunk: str) -> Tuple[float, str]:
        """
        Evaluate if a chunk is semantically coherent

        Returns:
            (score_0_to_1, rationale)
        """
        prompt = f"""
Evaluate the semantic coherence of the following text chunk from an academic document.

A semantically coherent chunk should:
1. Present a complete thought or concept
2. Not end abruptly in the middle of an explanation
3. Have logical flow between sentences
4. Not contain unrelated content

Text chunk:
{chunk[:1500]}

Rate the semantic coherence on a scale of 1-10, where:
- 10: Perfectly coherent, complete thought
- 5: Somewhat coherent but incomplete
- 1: Incoherent, fragmented, or incomplete

Return your answer as JSON:
{{"score": <number 1-10>, "rationale": "<brief explanation>"}}
"""

        messages = [{"role": "user", "content": prompt}]
        response = self.call_api(messages)

        # Parse JSON response
        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            start = response.find("{")
            end = response.rfind("}") + 1
            result = json.loads(response[start:end])

            score_normalized = result.get("score", 5) / 10.0
            rationale = result.get("rationale", "")

            return score_normalized, rationale

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return 0.5, "Failed to evaluate"

    def evaluate_concept_completeness(self, chunk: str) -> Tuple[float, str]:
        """
        Evaluate if a chunk contains complete concepts or is fragmented

        Returns:
            (score_0_to_1, rationale)
        """
        prompt = f"""
Evaluate if the following text chunk contains complete concepts or appears to be fragmented.

Check for signs of fragmentation:
1. Ends mid-sentence or mid-explanation
2. References something that was likely explained earlier (missing context)
3. Contains bullet points that seem cut off
4. Has obvious transitions that don't make sense

Text chunk:
{chunk[:1500]}

Rate the completeness on a scale of 1-10, where:
- 10: Complete concepts, no fragmentation
- 5: Some minor fragmentation but mostly complete
- 1: Heavily fragmented, incomplete concepts

Return your answer as JSON:
{{"score": <number 1-10>, "rationale": "<brief explanation>"}}
"""

        messages = [{"role": "user", "content": prompt}]
        response = self.call_api(messages)

        try:
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].split("```")[0].strip()

            start = response.find("{")
            end = response.rfind("}") + 1
            result = json.loads(response[start:end])

            score_normalized = result.get("score", 5) / 10.0
            rationale = result.get("rationale", "")

            return score_normalized, rationale

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to parse LLM response: {e}")
            return 0.5, "Failed to evaluate"


class ParsingBenchmark:
    """
    Benchmark PDF parsing quality

    Compares enhanced parsing strategy against baseline
    """

    def __init__(self, api_key: str):
        self.llm_evaluator = LLMEvaluator(api_key)

    def compute_table_preservation(self, chunks: List[Dict]) -> float:
        """
        Compute table preservation score

        For now, we use a simple heuristic:
        - Count chunks that look like tables (have Markdown table syntax)
        - Higher score = more tables captured

        Future: Can compare against ground truth annotations
        """
        table_pattern = re.compile(r'^\|.*\|$', re.MULTILINE)

        table_count = 0
        for chunk in chunks:
            content = chunk.get("content", "")
            if table_pattern.search(content):
                table_count += 1

        # Normalize by total chunks (avoid div by zero)
        if not chunks:
            return 0.0

        return min(1.0, table_count / len(chunks) * 5)  # Scale factor

    def compute_content_coverage(self, enhanced_chunks: List[Dict], baseline_chunks: List[Dict]) -> float:
        """
        Compute content coverage: how much content is captured vs baseline

        We compare total character count as a simple proxy
        """
        enhanced_total = sum(len(c.get("content", "")) for c in enhanced_chunks)
        baseline_total = sum(len(c.get("content", "")) for c in baseline_chunks)

        if baseline_total == 0:
            return 1.0 if enhanced_total > 0 else 0.0

        # Coverage should be close to 1.0 if we capture similar amount of content
        # Can be > 1.0 if we capture more (due to overlapping windows)
        ratio = enhanced_total / baseline_total

        # Score is 1.0 if ratio is between 0.8 and 1.2, decreases otherwise
        if 0.8 <= ratio <= 1.2:
            return 1.0
        elif ratio < 0.8:
            return ratio  # Penalize under-coverage
        else:
            return max(0.0, 2.0 - ratio)  # Penalize over-coverage (redundancy)

    def run_evaluation(
        self,
        enhanced_chunks: List[Dict],
        baseline_chunks: List[Dict],
        sample_size: int = 20
    ) -> Tuple[BenchmarkResult, BenchmarkResult]:
        """
        Run full benchmark evaluation

        Args:
            enhanced_chunks: Chunks from enhanced parser (Camelot + ParseBlock)
            baseline_chunks: Chunks from baseline parser (pdfplumber only)
            sample_size: Number of chunks to sample for LLM evaluation

        Returns:
            (enhanced_result, baseline_result)
        """
        print("=" * 60)
        print("PDF PARSING BENCHMARK")
        print("=" * 60)

        # Sample chunks for LLM evaluation
        enhanced_sample = enhanced_chunks[:sample_size] if len(enhanced_chunks) > sample_size else enhanced_chunks
        baseline_sample = baseline_chunks[:sample_size] if len(baseline_chunks) > sample_size else baseline_chunks

        # Evaluate Enhanced Parser
        print("\n[1/2] Evaluating Enhanced Parser...")
        enhanced_table = self.compute_table_preservation(enhanced_chunks)
        enhanced_coverage = self.compute_content_coverage(enhanced_chunks, baseline_chunks)

        # LLM evaluations for sample
        print(f"  Evaluating {len(enhanced_sample)} chunks with LLM...")
        enhanced_coherence_scores = []
        enhanced_completeness_scores = []

        for i, chunk in enumerate(enhanced_sample):
            print(f"    Chunk {i+1}/{len(enhanced_sample)}...", end="\r")
            content = chunk.get("content", "")

            coherence, _ = self.llm_evaluator.evaluate_semantic_coherence(content)
            completeness, _ = self.llm_evaluator.evaluate_concept_completeness(content)

            enhanced_coherence_scores.append(coherence)
            enhanced_completeness_scores.append(completeness)

        enhanced_coherence = sum(enhanced_coherence_scores) / len(enhanced_coherence_scores)
        enhanced_completeness = sum(enhanced_completeness_scores) / len(enhanced_completeness_scores)

        enhanced_overall = (
            0.15 * enhanced_table +
            0.35 * enhanced_coherence +
            0.35 * enhanced_completeness +
            0.15 * enhanced_coverage
        )

        enhanced_result = BenchmarkResult(
            parser_name="Enhanced (Camelot + ParseBlock)",
            total_chunks=len(enhanced_chunks),
            table_preservation=enhanced_table,
            semantic_coherence=enhanced_coherence,
            no_fragmentation=enhanced_completeness,
            content_coverage=enhanced_coverage,
            overall_score=enhanced_overall
        )

        # Evaluate Baseline Parser
        print("\n[2/2] Evaluating Baseline Parser...")
        baseline_table = self.compute_table_preservation(baseline_chunks)
        baseline_coverage = self.compute_content_coverage(baseline_chunks, baseline_chunks)  # 1.0 by definition

        print(f"  Evaluating {len(baseline_sample)} chunks with LLM...")
        baseline_coherence_scores = []
        baseline_completeness_scores = []

        for i, chunk in enumerate(baseline_sample):
            print(f"    Chunk {i+1}/{len(baseline_sample)}...", end="\r")
            content = chunk.get("content", "")

            coherence, _ = self.llm_evaluator.evaluate_semantic_coherence(content)
            completeness, _ = self.llm_evaluator.evaluate_concept_completeness(content)

            baseline_coherence_scores.append(coherence)
            baseline_completeness_scores.append(completeness)

        baseline_coherence = sum(baseline_coherence_scores) / len(baseline_coherence_scores)
        baseline_completeness = sum(baseline_completeness_scores) / len(baseline_completeness_scores)

        baseline_overall = (
            0.15 * baseline_table +
            0.35 * baseline_coherence +
            0.35 * baseline_completeness +
            0.15 * baseline_coverage
        )

        baseline_result = BenchmarkResult(
            parser_name="Baseline (pdfplumber only)",
            total_chunks=len(baseline_chunks),
            table_preservation=baseline_table,
            semantic_coherence=baseline_coherence,
            no_fragmentation=baseline_completeness,
            content_coverage=baseline_coverage,
            overall_score=baseline_overall
        )

        return enhanced_result, baseline_result

    def print_results(self, enhanced: BenchmarkResult, baseline: BenchmarkResult):
        """Print benchmark results in a nice format"""
        print("\n" + "=" * 60)
        print("BENCHMARK RESULTS")
        print("=" * 60)

        print(f"\n{'Metric':<25} {'Enhanced':>12} {'Baseline':>12} {'Improvement':>12}")
        print("-" * 63)

        metrics = [
            ("Total Chunks", enhanced.total_chunks, baseline.total_chunks, ""),
            ("Table Preservation", f"{enhanced.table_preservation:.3f}", f"{baseline.table_preservation:.3f}",
             f"+{((enhanced.table_preservation - baseline.table_preservation) / max(baseline.table_preservation, 0.001) * 100):.1f}%" if baseline.table_preservation > 0 else "N/A"),
            ("Semantic Coherence", f"{enhanced.semantic_coherence:.3f}", f"{baseline.semantic_coherence:.3f}",
             f"+{((enhanced.semantic_coherence - baseline.semantic_coherence) / max(baseline.semantic_coherence, 0.001) * 100):.1f}%" if baseline.semantic_coherence > 0 else "N/A"),
            ("No Fragmentation", f"{enhanced.no_fragmentation:.3f}", f"{baseline.no_fragmentation:.3f}",
             f"+{((enhanced.no_fragmentation - baseline.no_fragmentation) / max(baseline.no_fragmentation, 0.001) * 100):.1f}%" if baseline.no_fragmentation > 0 else "N/A"),
            ("Content Coverage", f"{enhanced.content_coverage:.3f}", f"{baseline.content_coverage:.3f}",
             f"+{((enhanced.content_coverage - baseline.content_coverage) / max(baseline.content_coverage, 0.001) * 100):.1f}%" if baseline.content_coverage > 0 else "N/A"),
        ]

        for metric, enh, base, improvement in metrics:
            print(f"{metric:<25} {enh:>12} {base:>12} {improvement:>12}")

        print("-" * 63)
        print(f"{'OVERALL SCORE':<25} {f'{enhanced.overall_score:.3f}':>12} {f'{baseline.overall_score:.3f}':>12} "
              f"{f'+{((enhanced.overall_score - baseline.overall_score) / max(baseline.overall_score, 0.001) * 100):.1f}%':>12}")

        print("\n" + "=" * 60)

        # Calculate the target 27.61% improvement
        actual_improvement = ((enhanced.overall_score - baseline.overall_score) / max(baseline.overall_score, 0.001)) * 100
        print(f"\nTarget improvement: 27.61%")
        print(f"Actual improvement: {actual_improvement:.2f}%")
        print(f"Gap: {27.61 - actual_improvement:.2f}%")

        print("=" * 60)

    def save_results(self, enhanced: BenchmarkResult, baseline: BenchmarkResult, output_path: str):
        """Save results to JSON file"""
        results = {
            "enhanced": enhanced.__dict__,
            "baseline": baseline.__dict__,
            "improvement_percent": ((enhanced.overall_score - baseline.overall_score) / max(baseline.overall_score, 0.001)) * 100
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to {output_path}")


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    import argparse

    API_KEY = "bfacda9a355fd3e9557063bc90aa8a1e.1zJ3JaydaKIkZ6Kq"

    parser = argparse.ArgumentParser(description="Benchmark PDF parsing quality")
    parser.add_argument("--enhanced", default="data/parsed/enhanced_chunks.json", help="Enhanced parser output")
    parser.add_argument("--baseline", default="data/parsed/baseline_chunks.json", help="Baseline parser output")
    parser.add_argument("--output", default="data/parsed/benchmark_results.json", help="Results output file")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of chunks to evaluate with LLM")

    args = parser.parse_args()

    # Load chunks
    with open(args.enhanced, 'r', encoding='utf-8') as f:
        enhanced_chunks = json.load(f)

    with open(args.baseline, 'r', encoding='utf-8') as f:
        baseline_chunks = json.load(f)

    # Run benchmark
    bench = ParsingBenchmark(API_KEY)
    enhanced_result, baseline_result = bench.run_evaluation(
        enhanced_chunks,
        baseline_chunks,
        sample_size=args.sample_size
    )

    # Print and save results
    bench.print_results(enhanced_result, baseline_result)
    bench.save_results(enhanced_result, baseline_result, args.output)
