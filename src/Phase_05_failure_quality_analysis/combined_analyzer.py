"""Combined Failure & Quality Analysis

Correlates failure modes with quality dimensions to identify root causes
and patterns in the generated dataset.
"""

import importlib
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple
from pipeline_core.utils import import_dual
from Phase_04_quality_evaluation.pydantic_classes import DIYRepairWithQualityScores


class CombinedAnalyzer:
    """Analyzes failure modes and quality dimensions together"""

    FAILURE_MODES = [
        "incomplete_answer",
        "safety_violations",
        "unrealistic_tools",
        "overcomplicated_solution",
        "missing_context",
        "poor_quality_tips",
    ]

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

    def __init__(self, labeled_samples: List[Any]):
        """Initialize with Phase 04 output (DIYRepairWithQualityScores items)"""
        self.samples = [DIYRepairWithQualityScores(**s) for s in labeled_samples]
        self.df = self._create_dataframe()

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert labeled samples to DataFrame for analysis"""
        rows = []

        for item in self.samples:
            # Extract failure judgment data
            failure_judgment = item.failure_judgment
            quality_judgment = item.quality_judgment

            if "Evaluation error" in failure_judgment.reasoning:
                continue

            row = {
                "trace_id": item.id,
                "issue_type": item.metadata.issue_type,
                # Failure modes
                "incomplete_answer": failure_judgment.incomplete_answer or 0,
                "safety_violations": failure_judgment.safety_violations or 0,
                "unrealistic_tools": failure_judgment.unrealistic_tools or 0,
                "overcomplicated_solution": failure_judgment.overcomplicated_solution
                or 0,
                "missing_context": failure_judgment.missing_context or 0,
                "poor_quality_tips": failure_judgment.poor_quality_tips or 0,
                "overall_failure": 1 if failure_judgment.overall_failure else 0,
                # Quality dimensions
                "answer_coherence": quality_judgment.quality_scores.answer_coherence,
                "step_actionability": quality_judgment.quality_scores.step_actionability,
                "tool_realism": quality_judgment.quality_scores.tool_realism,
                "safety_specificity": quality_judgment.quality_scores.safety_specificity,
                "tip_usefulness": quality_judgment.quality_scores.tip_usefulness,
                "problem_answer_alignment": quality_judgment.quality_scores.problem_answer_alignment,
                "appropriate_scope": quality_judgment.quality_scores.appropriate_scope,
                "category_accuracy": quality_judgment.quality_scores.category_accuracy,
                "quality_pass": 1 if quality_judgment.quality_pass else 0,
                "overall_quality_score": quality_judgment.overall_score,
            }

            rows.append(row)

        return pd.DataFrame(rows)

    def create_failure_quality_correlation_heatmap(self, output_file: str):
        """
        Create correlation matrix between failure modes and quality dimensions
        Shows which failure modes co-occur with which quality dimension failures
        """
        if len(self.df) == 0:
            print("⚠️  No data to visualize")
            return

        # Create correlation matrix
        failure_cols = self.FAILURE_MODES
        quality_cols = self.QUALITY_DIMENSIONS

        # Extract subset of dataframe
        corr_data = self.df[failure_cols + quality_cols].copy()

        # Calculate correlation between failure modes and quality dimensions
        corr_matrix = corr_data[failure_cols].corrwith(corr_data[quality_cols], axis=0)

        # Reshape for heatmap (6 failure modes x 8 quality dimensions)
        heatmap_data = []
        for failure in failure_cols:
            row = []
            for quality in quality_cols:
                correlation = corr_data[[failure, quality]].corr().iloc[0, 1]
                row.append(correlation if not pd.isna(correlation) else 0)
            heatmap_data.append(row)

        heatmap_array = np.array(heatmap_data)

        # Create heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(
            heatmap_array,
            xticklabels=quality_cols,
            yticklabels=failure_cols,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            square=False,
            linewidths=1,
            linecolor="white",
            cbar_kws={"label": "Correlation Coefficient", "shrink": 0.8},
        )

        plt.title(
            "Failure Mode ↔ Quality Dimension Correlation\nWhich failure modes co-occur with quality failures?",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Quality Dimensions", fontsize=12, fontweight="bold")
        plt.ylabel("Failure Modes", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"📊 Failure-quality correlation heatmap saved to: {output_file}")
        plt.close()

    def identify_failure_quality_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Identify top failure mode ↔ quality dimension pairs that co-occur
        Returns list of (failure_mode, quality_dimension, correlation) tuples
        """
        pairs = []

        for failure in self.FAILURE_MODES:
            for quality in self.QUALITY_DIMENSIONS:
                correlation = self.df[[failure, quality]].corr().iloc[0, 1]
                if (
                    not pd.isna(correlation) and correlation > 0.1
                ):  # Only positive correlations
                    pairs.append((failure, quality, round(correlation, 3)))

        # Sort by strength
        pairs.sort(key=lambda x: x[2], reverse=True)
        return pairs[:10]  # Top 10 pairs

    def generate_combined_report(self, output_file: str) -> Dict[str, Any]:
        """Generate comprehensive combined analysis report"""
        # Calculate statistics
        total_samples = len(self.df)
        total_failures = self.df["overall_failure"].sum()
        failure_rate = total_failures / total_samples if total_samples > 0 else 0

        total_quality_pass = self.df["quality_pass"].sum()
        quality_pass_rate = (
            total_quality_pass / total_samples if total_samples > 0 else 0
        )

        # Worst category by failure
        worst_category = None
        worst_rate = 0
        if "issue_type" in self.df.columns:
            for category in self.df["issue_type"].unique():
                cat_failure_rate = self.df[self.df["issue_type"] == category][
                    "overall_failure"
                ].mean()
                if cat_failure_rate > worst_rate:
                    worst_rate = cat_failure_rate
                    worst_category = category

        # Worst quality dimension
        worst_quality_dim = None
        worst_quality_rate = 0
        for dim in self.QUALITY_DIMENSIONS:
            fail_rate = (
                1 - (self.df[dim].sum() / total_samples) if total_samples > 0 else 0
            )
            if fail_rate > worst_quality_rate:
                worst_quality_rate = fail_rate
                worst_quality_dim = dim

        # Top failure-quality pairs
        failure_quality_pairs = self.identify_failure_quality_pairs()

        report = {
            "total_samples": total_samples,
            "overall_failure_rate": round(failure_rate, 3),
            "overall_quality_pass_rate": round(quality_pass_rate, 3),
            "failed_samples": int(total_failures),
            "quality_pass_samples": int(total_quality_pass),
            "worst_category_by_failure": (
                {
                    "category": worst_category,
                    "failure_rate": round(worst_rate, 3),
                }
                if worst_category
                else None
            ),
            "worst_quality_dimension": (
                {
                    "dimension": worst_quality_dim,
                    "failure_rate": round(worst_quality_rate, 3),
                }
                if worst_quality_dim
                else None
            ),
            "top_failure_quality_pairs": [
                {"failure_mode": f, "quality_dimension": q, "correlation": corr}
                for f, q, corr in failure_quality_pairs
            ],
            "failure_mode_frequencies": {
                mode: int(self.df[mode].sum()) for mode in self.FAILURE_MODES
            },
            "quality_dimension_pass_rates": (
                {
                    dim: round(self.df[dim].sum() / total_samples, 3)
                    for dim in self.QUALITY_DIMENSIONS
                }
                if total_samples > 0
                else {}
            ),
        }

        # Save report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 Combined analysis report saved to: {output_file}")
        return report

    def print_analysis_summary(self):
        """Print human-readable analysis summary"""
        print("\n" + "=" * 80)
        print("📊 COMBINED FAILURE & QUALITY ANALYSIS SUMMARY")
        print("=" * 80)

        total = len(self.df)
        failures = self.df["overall_failure"].sum()
        quality_pass = self.df["quality_pass"].sum()

        print(f"\nTotal Samples: {total}")
        print(f"Overall Failure Rate: {failures}/{total} ({100*failures/total:.1f}%)")
        print(
            f"Overall Quality Pass Rate: {quality_pass}/{total} ({100*quality_pass/total:.1f}%)"
        )

        print("\n🔴 Top Failure Modes:")
        for mode in self.FAILURE_MODES:
            count = self.df[mode].sum()
            pct = 100 * count / total if total > 0 else 0
            print(f"  {mode}: {int(count)} ({pct:.1f}%)")

        print("\n🟡 Quality Dimension Pass Rates:")
        for dim in self.QUALITY_DIMENSIONS:
            pass_count = self.df[dim].sum()
            pct = 100 * pass_count / total if total > 0 else 0
            print(f"  {dim}: {int(pass_count)}/{total} ({pct:.1f}%)")

        print("\n🔗 Top Failure ↔ Quality Correlations:")
        pairs = self.identify_failure_quality_pairs()
        for failure, quality, corr in pairs[:5]:
            print(f"  {failure} ↔ {quality}: {corr}")

        print("\n" + "=" * 80)

    def run_full_analysis(self, output_dir: str) -> Dict[str, str]:
        """
        Run complete combined analysis
        Returns dictionary of all generated file paths
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("📈 PHASE 05: FAILURE & QUALITY ANALYSIS")
        print("=" * 80)

        # Generate correlation chart
        correlation_file = os.path.join(output_dir, "failure_quality_correlation.png")
        self.create_failure_quality_correlation_heatmap(correlation_file)

        # Generate combined report
        report_file = os.path.join(output_dir, "combined_analysis_report.json")
        self.generate_combined_report(report_file)

        # Print summary
        self.print_analysis_summary()

        output_files = {
            "failure_quality_correlation": correlation_file,
            "combined_report": report_file,
        }

        print("\n✅ Phase 05 Analysis Complete!")
        print(f"\n📁 Output files:")
        for name, path in output_files.items():
            print(f"   - {path}")

        return output_files
