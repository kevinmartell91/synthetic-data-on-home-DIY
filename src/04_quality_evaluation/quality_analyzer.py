"""
Quality Pattern Analyzer
Analyzes quality dimension patterns and creates visualizations
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless; avoids TkAgg/tkinter teardown crashes on CLI/WSL

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import List, Dict, Any
from .pydantic_classes import DIYRepairWithQualityScores


class QualityAnalyzer:
    """Analyzes quality patterns across the dataset"""

    def __init__(self, labeled_samples: List[DIYRepairWithQualityScores]):
        self.samples = labeled_samples
        self.quality_dimensions = [
            "answer_coherence",
            "step_actionability",
            "tool_realism",
            "safety_specificity",
            "tip_usefulness",
            "problem_answer_alignment",
            "appropriate_scope",
            "category_accuracy",
        ]

    def create_quality_heatmap(self, output_dir: str) -> str:
        """Create heatmap of quality scores across samples"""
        quality_matrix = []
        sample_ids = []

        for sample in self.samples:
            scores = [
                getattr(sample.quality_judgment.quality_scores, dim)
                for dim in self.quality_dimensions
            ]
            quality_matrix.append(scores)
            sample_ids.append(sample.id[:8])

        plt.figure(figsize=(12, max(8, len(self.samples) * 0.3)))
        sns.heatmap(
            quality_matrix,
            xticklabels=self.quality_dimensions,
            yticklabels=sample_ids,
            cmap="RdYlGn",
            cbar_kws={"label": "Pass (1) / Fail (0)"},
            linewidths=0.5,
        )
        plt.title("Quality Dimension Scores Across Samples")
        plt.xlabel("Quality Dimension")
        plt.ylabel("Sample ID")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        output_path = os.path.join(output_dir, "quality_heatmap.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def create_dimension_distribution(self, output_dir: str) -> str:
        """Create bar chart of pass rates per dimension"""
        pass_counts = {dim: 0 for dim in self.quality_dimensions}

        for sample in self.samples:
            for dim in self.quality_dimensions:
                if getattr(sample.quality_judgment.quality_scores, dim) == 1:
                    pass_counts[dim] += 1

        pass_rates = {
            dim: count / len(self.samples) * 100 for dim, count in pass_counts.items()
        }

        plt.figure(figsize=(12, 6))
        bars = plt.bar(pass_rates.keys(), pass_rates.values())
        plt.axhline(y=80, color="r", linestyle="--", label="80% Target")
        plt.xlabel("Quality Dimension")
        plt.ylabel("Pass Rate (%)")
        plt.title("Quality Dimension Pass Rates")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()

        output_path = os.path.join(output_dir, "quality_dimension_distribution.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        return output_path

    def create_quality_dimension_scores_chart(
        self, output_dir: str, baseline_data: Dict[str, float] = None
    ) -> str:
        """
        Create enhanced bar chart of quality dimension pass rates
        Colors each bar by pass rate using RdYlGn colormap
        Adds percentage labels on bars, supports before/after baseline comparison

        Args:
            output_dir: Directory to save chart
            baseline_data: Optional dict of {dimension: pass_rate_pct} for before/after comparison

        Returns:
            Path to saved chart file
        """
        # Calculate current pass rates
        pass_counts = {dim: 0 for dim in self.quality_dimensions}

        for sample in self.samples:
            for dim in self.quality_dimensions:
                if getattr(sample.quality_judgment.quality_scores, dim) == 1:
                    pass_counts[dim] += 1

        current_pass_rates = {
            dim: count / len(self.samples) * 100 for dim, count in pass_counts.items()
        }

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(self.quality_dimensions))
        width = 0.35

        if baseline_data is not None:
            # Before vs After comparison
            bars1 = ax.bar(
                x - width / 2,
                [baseline_data.get(d, 0) for d in self.quality_dimensions],
                width,
                label="Baseline",
                color="#e74c3c",
                edgecolor="black",
                linewidth=1.2,
            )
            bars2 = ax.bar(
                x + width / 2,
                [current_pass_rates[d] for d in self.quality_dimensions],
                width,
                label="Post-Correction",
                color="#27ae60",
                edgecolor="black",
                linewidth=1.2,
            )

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 1,
                        f"{height:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

            ax.legend(fontsize=11, loc="lower right")
            title = "Quality Dimension Pass Rates: Baseline vs. Post-Correction\nAll 8 Quality Dimensions"
        else:
            # Current rates only with color mapping
            normalized_rates = np.array(
                [current_pass_rates[d] / 100 for d in self.quality_dimensions]
            )
            colors = plt.cm.RdYlGn(normalized_rates)
            bars = ax.bar(
                x,
                [current_pass_rates[d] for d in self.quality_dimensions],
                color=colors,
                edgecolor="black",
                linewidth=1.2,
            )

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1,
                    f"{height:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            title = "Quality Dimension Pass Rates\nAll 8 Quality Dimensions"

        # Add 80% threshold line
        ax.axhline(
            y=80,
            color="red",
            linestyle="--",
            linewidth=2,
            label="80% Threshold",
            alpha=0.7,
        )

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Quality Dimensions", fontsize=12, fontweight="bold")
        ax.set_ylabel("Pass Rate (%)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(self.quality_dimensions, rotation=45, ha="right")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        output_path = os.path.join(output_dir, "quality_dimension_scores.png")
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"📊 Quality dimension scores chart saved to: {output_path}")
        plt.close()

        return output_path

    def generate_quality_report(self, output_path: str):
        """Generate comprehensive quality analysis report"""
        report = {
            "total_samples": len(self.samples),
            "overall_quality_pass_rate": sum(
                1 for s in self.samples if s.quality_judgment.quality_pass
            )
            / len(self.samples),
            "average_quality_score": sum(
                s.quality_judgment.overall_score for s in self.samples
            )
            / len(self.samples),
            "dimension_analysis": {},
        }

        for dim in self.quality_dimensions:
            pass_count = sum(
                1
                for s in self.samples
                if getattr(s.quality_judgment.quality_scores, dim) == 1
            )
            report["dimension_analysis"][dim] = {
                "pass_count": pass_count,
                "fail_count": len(self.samples) - pass_count,
                "pass_rate": pass_count / len(self.samples),
            }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        return output_path

    def run_full_analysis(self, output_dir: str) -> Dict[str, str]:
        """Run complete quality analysis and generate all outputs"""
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("📈 QUALITY PATTERN ANALYSIS")
        print("=" * 60)

        viz_files = {
            "quality_heatmap": self.create_quality_heatmap(output_dir),
            "dimension_distribution": self.create_dimension_distribution(output_dir),
            "quality_dimension_scores": self.create_quality_dimension_scores_chart(
                output_dir
            ),
        }

        report_file = os.path.join(output_dir, "quality_analysis_report.json")
        self.generate_quality_report(report_file)
        viz_files["quality_report"] = report_file

        print("\n✅ Quality Analysis Complete!")
        print(f"\n📁 Output files:")
        for name, path in viz_files.items():
            print(f"   - {path}")

        return viz_files
