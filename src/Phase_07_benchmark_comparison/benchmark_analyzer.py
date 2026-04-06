"""Benchmark comparison and judge calibration analysis"""

import json
import os
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.Phase_04_quality_evaluation.pydantic_classes import DIYRepairWithQualityScores
from src.Phase_04_quality_evaluation.quality_judge import QualityJudge


class BenchmarkAnalyzer:
    """
    Validates judge calibration and compares benchmark vs generated quality
    """

    QUALITY_DIMENSIONS = [
        "answer_coherence",
        "step_actionability",
        "tool_realism",
        "safety_specificity",
        "tip_usefulness",
        "problem_answer_alignment",
        "appropriate_scope",
        "category_accuracy",
    ]

    CALIBRATION_THRESHOLD = 0.95  # Judge must pass >=95% of benchmark

    def __init__(self):
        pass

    def run_judge_on_benchmark(
        self, benchmark_items: List[Any]
    ) -> Tuple[List[DIYRepairWithQualityScores], Dict[str, Any]]:
        """
        Run Phase 04 quality judge on benchmark items

        Returns:
            (evaluated_items, statistics)
        """
        print(
            f"\n🔍 Evaluating {len(benchmark_items)} benchmark items with quality judge..."
        )

        # Convert to dict format for QualityJudge
        input_data = [
            {
                **item.model_dump(),
                "failure_judgment": {
                    "trace_id": item.id,
                    "incomplete_answer": 0,
                    "safety_violations": 0,
                    "unrealistic_tools": 0,
                    "overcomplicated_solution": 0,
                    "missing_context": 0,
                    "poor_quality_tips": 0,
                    "overall_failure": False,
                    "reasoning": "Benchmark item (no failure evaluation)",
                    "metadata": item.metadata.model_dump(),
                },
            }
            for item in benchmark_items
        ]

        # Use Phase 04 quality judge (handles dict→Pydantic conversion internally)
        judge = QualityJudge(input_data)
        evaluated_items, stats = judge.evaluate_dataset()

        for item in evaluated_items:
            if not item.quality_judgment.quality_pass:
                print(f"\n❌ FAILED ITEM: {item.id}")
                print(f"   Scores: {item.quality_judgment.quality_scores.model_dump()}")
                print(f"   Reasoning: {item.quality_judgment.reasoning}")

        return evaluated_items, stats

    def check_calibration(self, benchmark_stats: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if judge passes calibration threshold on benchmark

        Returns:
            (passed: bool, message: str)
        """
        pass_rate = (
            benchmark_stats["quality_pass_count"] / benchmark_stats["evaluated"]
            if benchmark_stats["evaluated"] > 0
            else 0
        )

        passed = pass_rate >= self.CALIBRATION_THRESHOLD
        message = (
            f"✅ Judge CALIBRATED: {pass_rate:.1%} of benchmark items pass "
            f"(threshold: {self.CALIBRATION_THRESHOLD:.0%})"
            if passed
            else f"❌ Judge MISCALIBRATED: {pass_rate:.1%} of benchmark items pass "
            f"(threshold: {self.CALIBRATION_THRESHOLD:.0%})\n"
            f"   ⚠️  Judge criteria need adjustment before trusting generated data evaluation"
        )

        return passed, message

    def compare_quality_gap(
        self,
        benchmark_stats: Dict[str, Any],
        generated_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare quality metrics between benchmark and generated dataset

        Returns:
            Dictionary with gap analysis per dimension
        """
        benchmark_dim_rates = benchmark_stats.get("dimension_pass_counts", {})
        generated_dim_rates = generated_stats.get("dimension_pass_counts", {})

        # Normalize to pass rates (0-1)
        benchmark_total = benchmark_stats.get("evaluated", 1)
        generated_total = generated_stats.get("evaluated", 1)

        gap_analysis = {
            "total_samples": {
                "benchmark": benchmark_total,
                "generated": generated_total,
            },
            "overall_pass_rates": {
                "benchmark": (
                    round(
                        benchmark_stats.get("quality_pass_count", 0) / benchmark_total,
                        3,
                    )
                    if benchmark_total > 0
                    else 0.0
                ),
                "generated": (
                    round(
                        generated_stats.get("quality_pass_count", 0) / generated_total,
                        3,
                    )
                    if generated_total > 0
                    else 0.0
                ),
            },
            "dimension_gap": {},
        }

        for dim in self.QUALITY_DIMENSIONS:
            benchmark_pass = benchmark_dim_rates.get(dim, 0)
            generated_pass = generated_dim_rates.get(dim, 0)

            benchmark_rate = (
                benchmark_pass / benchmark_total if benchmark_total > 0 else 0
            )
            generated_rate = (
                generated_pass / generated_total if generated_total > 0 else 0
            )

            gap = benchmark_rate - generated_rate  # Positive = benchmark better

            gap_analysis["dimension_gap"][dim] = {
                "benchmark_pass_rate": round(benchmark_rate, 3),
                "generated_pass_rate": round(generated_rate, 3),
                "gap": round(gap, 3),
            }

        return gap_analysis

    def create_gap_analysis_chart(self, gap_analysis: Dict[str, Any], output_file: str):
        """Create side-by-side comparison chart of benchmark vs generated quality"""
        dimensions = self.QUALITY_DIMENSIONS
        benchmark_rates = [
            gap_analysis["dimension_gap"][dim]["benchmark_pass_rate"]
            for dim in dimensions
        ]
        generated_rates = [
            gap_analysis["dimension_gap"][dim]["generated_pass_rate"]
            for dim in dimensions
        ]

        x = np.arange(len(dimensions))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 7))

        bars1 = ax.bar(
            x - width / 2,
            benchmark_rates,
            width,
            label="Benchmark (Reference)",
            color="#27ae60",
            edgecolor="black",
            linewidth=1.2,
        )
        bars2 = ax.bar(
            x + width / 2,
            generated_rates,
            width,
            label="Generated Dataset",
            color="#e74c3c",
            edgecolor="black",
            linewidth=1.2,
        )

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.01,
                    f"{height:.0%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

        ax.set_title(
            "Quality Gap Analysis: Benchmark vs Generated Dataset\nAll 8 Quality Dimensions",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Quality Dimension", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pass Rate", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=11, loc="upper right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"📊 Gap analysis chart saved to: {output_file}")
        plt.close()

    def generate_final_report(
        self,
        benchmark_stats: Dict[str, Any],
        generated_stats: Dict[str, Any],
        gap_analysis: Dict[str, Any],
        calibrated: bool,
        output_file: str,
    ) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        report = {
            "judge_calibration": {
                "passed": calibrated,
                "threshold": f"{self.CALIBRATION_THRESHOLD:.0%}",
                "benchmark_pass_rate": (
                    round(
                        benchmark_stats.get("quality_pass_count", 0)
                        / benchmark_stats["evaluated"],
                        3,
                    )
                    if benchmark_stats.get("evaluated", 0) > 0
                    else 0.0
                ),
                "benchmark_samples_evaluated": benchmark_stats.get("evaluated", 0),
            },
            "quality_gap_analysis": gap_analysis,
            "summary": {
                "benchmark_overall_pass_rate": gap_analysis["overall_pass_rates"][
                    "benchmark"
                ],
                "generated_overall_pass_rate": gap_analysis["overall_pass_rates"][
                    "generated"
                ],
                "overall_gap": round(
                    gap_analysis["overall_pass_rates"]["benchmark"]
                    - gap_analysis["overall_pass_rates"]["generated"],
                    3,
                ),
            },
            "worst_dimensions": sorted(
                [
                    {
                        "dimension": dim,
                        "gap": gap_analysis["dimension_gap"][dim]["gap"],
                        "benchmark_rate": gap_analysis["dimension_gap"][dim][
                            "benchmark_pass_rate"
                        ],
                        "generated_rate": gap_analysis["dimension_gap"][dim][
                            "generated_pass_rate"
                        ],
                    }
                    for dim in self.QUALITY_DIMENSIONS
                ],
                key=lambda x: x["gap"],
                reverse=True,
            )[:3],
        }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 Final report saved to: {output_file}")
        return report

    def compute_generated_stats(self, items: List[Any]) -> Dict[str, Any]:
        """
        Extract quality statistics from already-evaluated Phase 04 items
        (items that already have quality_judgment attached)
        """
        if not items:
            return {
                "total_samples": 0,
                "evaluated": 0,
                "quality_pass_count": 0,
                "dimension_pass_counts": {dim: 0 for dim in self.QUALITY_DIMENSIONS},
                "average_score": 0.0,
            }
        total = len(items)
        pass_count = 0
        dim_pass_counts = {dim: 0 for dim in self.QUALITY_DIMENSIONS}
        score_sum = 0.0

        quality_scores = [DIYRepairWithQualityScores(**s) for s in items]
        for item in quality_scores:
            if item.quality_judgment:
                qj = item.quality_judgment
                if qj.quality_pass:
                    pass_count += 1
                score_sum += qj.overall_score
                for dim in self.QUALITY_DIMENSIONS:
                    if getattr(qj.quality_scores, dim, 0) == 1:
                        dim_pass_counts[dim] += 1

        return {
            "total_samples": total,
            "evaluated": total,
            "quality_pass_count": pass_count,
            "dimension_pass_counts": dim_pass_counts,
            "average_score": score_sum / total if total > 0 else 0.0,
        }

    def print_final_summary(
        self,
        calibrated: bool,
        gap_analysis: Dict[str, Any],
        report: Dict[str, Any],
    ):
        """Print human-readable summary"""
        print("\n" + "=" * 80)
        print("🎯 PHASE 07: BENCHMARK COMPARISON & JUDGE CALIBRATION RESULTS")
        print("=" * 80)

        # Calibration result
        if calibrated:
            print("\n✅ Judge PASSED Calibration")
        else:
            print("\n❌ Judge FAILED Calibration")
            print("   ⚠️  Judge criteria need adjustment")

        # Overall quality gap
        print(f"\nOverall Quality Pass Rate Gap:")
        print(f"  Benchmark: {gap_analysis['overall_pass_rates']['benchmark']:.1%}")
        print(f"  Generated: {gap_analysis['overall_pass_rates']['generated']:.1%}")
        print(f"  Gap: {report['summary']['overall_gap']:.1%}")

        # Worst dimensions
        print("\n🔴 Dimensions with Largest Quality Gap (Benchmark better):")
        for item in report["worst_dimensions"]:
            print(
                f"  {item['dimension']}: {item['gap']:+.1%} "
                f"(Benchmark: {item['benchmark_rate']:.0%}, "
                f"Generated: {item['generated_rate']:.0%})"
            )

        print("\n" + "=" * 80)
