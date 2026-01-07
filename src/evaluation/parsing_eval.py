"""
PDF Parsing Benchmark - Rule-based Evaluation (No API Required)

Metrics:
1. Concept Coverage: % of key concepts found in chunks
2. Concept Completeness: % of concepts not split across chunks
3. Chunk Quality: Average chunk length, structure preservation
4. Cross-Page Concept Handling: How well cross-page concepts are captured
"""

import json
import re
from typing import List, Dict, Tuple
from pathlib import Path


class Concept:
    """A key concept to evaluate"""
    def __init__(self, concept_id: str, name: str, keywords: List[str], expected_pages: List[int]):
        self.concept_id = concept_id
        self.name = name
        self.keywords = keywords
        self.expected_pages = expected_pages


class ParsingBenchmark:
    """
    Evaluate PDF parsing quality without using external APIs

    Compares enhanced parsing (ParseBlock) vs baseline (pdfplumber only)
    """

    def __init__(self):
        # Define key concepts from the course
        self.concepts = [
            Concept(
                "pc001",
                "Machine Learning vs Deep Learning",
                ["Machine Learning", "Deep Learning", "feature engineering", "neural networks", "manual", "automatic"],
                [4, 5, 11]
            ),
            Concept(
                "pc002",
                "Types of Machine Learning",
                ["supervised", "unsupervised", "reinforcement", "classification", "regression"],
                [5, 51]
            ),
            Concept(
                "pc003",
                "Overfitting Definition and Signs",
                ["overfitting", "training accuracy", "test accuracy", "generalization"],
                [21, 22]
            ),
            Concept(
                "pc004",
                "Overfitting Solutions",
                ["regularization", "dropout", "cross-validation", "early stopping", "simplify"],
                [21, 51]
            ),
            Concept(
                "pc005",
                "Decision Trees",
                ["Decision Tree", "interpret", "overfitting", "prone"],
                [31]
            ),
            Concept(
                "pc006",
                "Random Forest",
                ["Random Forest", "ensemble", "multiple trees", "overfitting"],
                [31, 51]
            ),
            Concept(
                "pc007",
                "Training vs Test Data",
                ["training", "test", "split", "validation", "70%", "80%"],
                [5, 50]
            ),
            Concept(
                "pc008",
                "Regression Analysis",
                ["regression", "predictive", "forecasting", "predictor", "trend"],
                [51]
            ),
            Concept(
                "pc009",
                "Supervised Algorithms",
                ["Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest", "SVM"],
                [31]
            ),
            Concept(
                "pc010",
                "Data Imbalance",
                ["imbalance", "unbalanced", "majority", "minority", "bias"],
                [21, 22]
            ),
            # Extended concepts for more thorough evaluation
            Concept(
                "pc011",
                "Support Vector Machines",
                ["SVM", "Support Vector Machine", "hyperplane", "margin"],
                [31]
            ),
            Concept(
                "pc012",
                "Naive Bayes",
                ["Naive Bayes", "Bayes", "independence", "probability"],
                [31]
            ),
            Concept(
                "pc013",
                "K-Nearest Neighbors",
                ["KNN", "K-Nearest Neighbours", "instance", "distance"],
                [31]
            ),
            Concept(
                "pc014",
                "Cross-Validation",
                ["cross-validation", "k-fold", "folds", "rotation"],
                [5]
            ),
            Concept(
                "pc015",
                "Logistic Regression",
                ["Logistic Regression", "binary", "classification", "probability"],
                [31, 51]
            ),
        ]

    def load_chunks(self, json_path: str) -> List[Dict]:
        """Load chunks from JSON file"""
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _contains_keywords(self, text: str, keywords: List[str], threshold: float = 0.5) -> bool:
        """Check if text contains enough keywords"""
        text_lower = text.lower()
        matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        return matches >= len(keywords) * threshold

    def _find_concept_in_chunks(self, concept: Concept, chunks: List[Dict]) -> Dict:
        """Find how a concept is distributed across chunks"""
        matching_chunks = []

        for i, chunk in enumerate(chunks):
            content = chunk.get("content", "")
            if self._contains_keywords(content, concept.keywords):
                matching_chunks.append({
                    "chunk_idx": i,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "pages": chunk.get("page_numbers", []),
                    "content": content[:200]
                })

        return {
            "concept": concept.name,
            "matching_chunks": matching_chunks,
            "num_chunks": len(matching_chunks)
        }

    def _calculate_concept_completeness(self, concept_result: Dict) -> float:
        """
        Calculate if concept is complete (not fragmented)

        Returns:
            Score 0-1, where 1 means concept is in 1-2 chunks (good),
            0 means spread across many chunks (bad)
        """
        num_chunks = concept_result["num_chunks"]

        if num_chunks == 0:
            return 0.0  # Concept not found
        elif num_chunks == 1:
            return 1.0  # Perfect: single chunk
        elif num_chunks == 2:
            return 0.9  # Good: might be split into definition + example
        elif num_chunks <= 4:
            return 0.6  # Fair: some fragmentation
        else:
            return 0.3  # Poor: heavily fragmented

    def evaluate_parser(self, chunks_path: str) -> Dict:
        """Evaluate a single parser output"""
        chunks = self.load_chunks(chunks_path)

        results = {
            "parser_name": chunks_path,
            "total_chunks": len(chunks),
            "concepts": [],
            "metrics": {}
        }

        # Evaluate each concept
        concept_scores = []

        for concept in self.concepts:
            concept_result = self._find_concept_in_chunks(concept, chunks)
            completeness = self._calculate_concept_completeness(concept_result)

            concept_result["completeness"] = completeness
            results["concepts"].append(concept_result)
            concept_scores.append(completeness)

        # Calculate overall metrics
        results["metrics"] = {
            # Concept Coverage: % of concepts found
            "concept_coverage": sum(1 for s in concept_scores if s > 0) / len(concept_scores),

            # Concept Completeness: Average completeness score
            "concept_completeness": sum(concept_scores) / len(concept_scores),

            # Total Chunks
            "total_chunks": len(chunks),

            # Avg Chunk Length
            "avg_chunk_length": sum(len(c.get("content", "")) for c in chunks) / len(chunks),

            # Chunks with Cross-Page Content
            "cross_page_chunks": sum(1 for c in chunks if len(c.get("page_numbers", [])) > 1),

            # Chunks that are too short (< 100 chars) - potential fragmentation
            "short_chunks": sum(1 for c in chunks if len(c.get("content", "")) < 100),

            # Chunks that are too long (> 2000 chars) - potential multi-concept
            "long_chunks": sum(1 for c in chunks if len(c.get("content", "")) > 2000),
        }

        # Overall score
        results["metrics"]["overall_score"] = (
            0.3 * results["metrics"]["concept_coverage"] +
            0.4 * results["metrics"]["concept_completeness"] +
            0.15 * (1 - results["metrics"]["short_chunks"] / len(chunks)) +
            0.15 * (1 - results["metrics"]["long_chunks"] / len(chunks))
        )

        return results

    def compare_parsers(
        self,
        enhanced_path: str,
        baseline_path: str
    ) -> Tuple[Dict, Dict, Dict]:
        """Compare enhanced vs baseline parsing"""

        print("=" * 70)
        print("PDF PARSING BENCHMARK - Enhanced vs Baseline")
        print("=" * 70)

        # Evaluate enhanced
        print("\n[1/2] Evaluating Enhanced Parser...")
        enhanced_results = self.evaluate_parser(enhanced_path)

        # Evaluate baseline
        print("[2/2] Evaluating Baseline Parser...")
        baseline_results = self.evaluate_parser(baseline_path)

        # Calculate comparison
        comparison = {
            "improvement": {
                "concept_coverage": (
                    enhanced_results["metrics"]["concept_coverage"] -
                    baseline_results["metrics"]["concept_coverage"]
                ),
                "concept_completeness": (
                    enhanced_results["metrics"]["concept_completeness"] -
                    baseline_results["metrics"]["concept_completeness"]
                ),
                "overall_score": (
                    enhanced_results["metrics"]["overall_score"] -
                    baseline_results["metrics"]["overall_score"]
                ),
            },
            "baseline_chunks": baseline_results["metrics"]["total_chunks"],
            "enhanced_chunks": enhanced_results["metrics"]["total_chunks"],
            "chunk_reduction": (
                (baseline_results["metrics"]["total_chunks"] - enhanced_results["metrics"]["total_chunks"]) /
                baseline_results["metrics"]["total_chunks"]
            )
        }

        return enhanced_results, baseline_results, comparison

    def print_results(self, enhanced: Dict, baseline: Dict, comparison: Dict):
        """Print detailed comparison results"""

        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS")
        print("=" * 70)

        print(f"\n{'Metric':<30} {'Enhanced':>12} {'Baseline':>12} {'Improvement':>12}")
        print("-" * 70)

        metrics = [
            ("Total Chunks", enhanced["metrics"]["total_chunks"], baseline["metrics"]["total_chunks"],
             f"{comparison['chunk_reduction']*100:.1f}% reduction"),
            ("Concept Coverage", f"{enhanced['metrics']['concept_coverage']:.3f}",
             f"{baseline['metrics']['concept_coverage']:.3f}",
             f"+{comparison['improvement']['concept_coverage']*100:.1f}%"),
            ("Concept Completeness", f"{enhanced['metrics']['concept_completeness']:.3f}",
             f"{baseline['metrics']['concept_completeness']:.3f}",
             f"+{comparison['improvement']['concept_completeness']*100:.1f}%"),
            ("Cross-Page Chunks", enhanced["metrics"]["cross_page_chunks"],
             baseline["metrics"]["cross_page_chunks"],
             f"+{enhanced['metrics']['cross_page_chunks'] - baseline['metrics']['cross_page_chunks']}"),
            ("Short Chunks (<100)", enhanced["metrics"]["short_chunks"],
             baseline["metrics"]["short_chunks"],
             f"{enhanced['metrics']['short_chunks'] - baseline['metrics']['short_chunks']}"),
            ("Long Chunks (>2000)", enhanced["metrics"]["long_chunks"],
             baseline["metrics"]["long_chunks"],
             f"{enhanced['metrics']['long_chunks'] - baseline['metrics']['long_chunks']}"),
            ("Avg Chunk Length", f"{enhanced['metrics']['avg_chunk_length']:.0f}",
             f"{baseline['metrics']['avg_chunk_length']:.0f}",
             f"+{enhanced['metrics']['avg_chunk_length'] - baseline['metrics']['avg_chunk_length']:.0f}"),
        ]

        for metric, enh, base, diff in metrics:
            print(f"{metric:<30} {enh:>12} {base:>12} {diff:>12}")

        print("-" * 70)
        overall_enh = f"{enhanced['metrics']['overall_score']:.3f}"
        overall_base = f"{baseline['metrics']['overall_score']:.3f}"
        overall_imp = f"+{comparison['improvement']['overall_score']*100:.1f}%"
        print(f"{'OVERALL SCORE':<30} {overall_enh:>12} {overall_base:>12} {overall_imp:>12}")

        print("\n" + "=" * 70)

        # Concept-by-concept breakdown
        print("\nCONCEPT BREAKDOWN:")
        print("-" * 70)

        for i, (enh_concept, base_concept) in enumerate(zip(enhanced["concepts"], baseline["concepts"])):
            if i >= 15:  # Show first 15
                print(f"... and {len(enhanced['concepts']) - 15} more concepts")
                break

            name = enh_concept["concept"][:30]
            enh_chunks = enh_concept["num_chunks"]
            base_chunks = base_concept["num_chunks"]
            enh_comp = f"{enh_concept['completeness']:.2f}"
            base_comp = f"{base_concept['completeness']:.2f}"

            print(f"{name:<30} chunks:{enh_chunks:>3}/{base_chunks:<3} "
                  f"comp:{enh_comp:>4}/{base_comp:<4}")

        print("=" * 70)

    def save_results(self, enhanced: Dict, baseline: Dict, comparison: Dict, output_path: str):
        """Save results to JSON"""
        results = {
            "enhanced": enhanced,
            "baseline": baseline,
            "comparison": comparison
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        print(f"\nResults saved to {output_path}")


# ============================================
# CLI
# ============================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark PDF parsing quality (no API required)")
    parser.add_argument("--enhanced", default="data/parsed/enhanced_chunks.json",
                        help="Enhanced parser output")
    parser.add_argument("--baseline", default="data/parsed/baseline_chunks.json",
                        help="Baseline parser output")
    parser.add_argument("--output", default="data/parsed/parsing_benchmark_results.json",
                        help="Results output file")

    args = parser.parse_args()

    # Run benchmark
    bench = ParsingBenchmark()
    enhanced_results, baseline_results, comparison = bench.compare_parsers(
        args.enhanced,
        args.baseline
    )

    # Print and save results
    bench.print_results(enhanced_results, baseline_results, comparison)
    bench.save_results(enhanced_results, baseline_results, comparison, args.output)
