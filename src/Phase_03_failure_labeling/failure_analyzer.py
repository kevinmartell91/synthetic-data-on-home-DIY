"""
Failure Mode Analysis Module
Creates heatmaps, identifies patterns, and correlations
"""

import json
import os

import matplotlib

matplotlib.use("Agg")  # headless; avoids TkAgg/tkinter teardown crashes on CLI/WSL

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from typing import Dict, Any, List, Tuple
from .pydantic_classes import DIYRepairWithFailureLabels


class FailureAnalyzer:
    """Analyzes failure mode patterns and correlations"""

    FAILURE_MODES = [
        "incomplete_answer",
        "safety_violations",
        "unrealistic_tools",
        "overcomplicated_solution",
        "missing_context",
        "poor_quality_tips",
    ]

    def __init__(self, labeled_samples: List[DIYRepairWithFailureLabels]):
        self.samples = labeled_samples
        self.df = self._create_dataframe()

    def _create_dataframe(self) -> pd.DataFrame:
        """Convert labeled samples to DataFrame for analysis"""
        rows = []

        for sample in self.samples:

            failure = sample.failure_judgment

            if "Evaluation error" in failure.reasoning:
                continue

            row = {
                "trace_id": sample.id,
                "issue_type": sample.metadata.issue_type,
                "response_type": sample.metadata.user_query,
                # Failure modes
                "incomplete_answer": failure.incomplete_answer,
                "safety_violations": failure.safety_violations,
                "unrealistic_tools": failure.unrealistic_tools,
                "overcomplicated_solution": failure.overcomplicated_solution,
                "missing_context": failure.missing_context,
                "poor_quality_tips": failure.poor_quality_tips,
                "overall_failure": 1 if failure.overall_failure else 0,
            }

            rows.append(row)

        return pd.DataFrame(rows)

    def create_failure_cooccurrence_heatmap(self, output_file: str):
        """
        Requirement 1: Failure Mode Co-occurrence
        A heatmap showing which failure modes tend to appear together
        Uses diverging color scale for correlation
        """
        if len(self.df) == 0:
            print("⚠️  No data to visualize")
            return

        # Calculate correlation matrix
        corr_matrix = self.df[self.FAILURE_MODES].corr()

        # Create figure
        plt.figure(figsize=(12, 10))

        # Create heatmap with diverging color scale
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",  # Diverging: red for positive correlation, blue for negative
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=1,
            linecolor="white",
            cbar_kws={"label": "Correlation Coefficient", "shrink": 0.8},
        )

        plt.title(
            "Failure Mode Co-occurrence Heatmap\nWhich Failure Modes Appear Together?",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Failure Modes", fontsize=12, fontweight="bold")
        plt.ylabel("Failure Modes", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        # Save with high quality
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"📊 Failure co-occurrence heatmap saved to: {output_file}")
        plt.close()

        return corr_matrix

    def create_failure_rates_by_category_chart(self, output_file: str):
        """
        Requirement 2: Failure Rates by Repair Category
        Bar chart comparing failure rates across repair categories
        Uses sequential color scale
        """
        if "issue_type" not in self.df.columns:
            print("⚠️  'issue_type' column not found")
            return

        # Calculate failure rate per category
        category_stats = []
        for category in self.df["issue_type"].unique():
            category_df = self.df[self.df["issue_type"] == category]
            total = len(category_df)
            failures = category_df["overall_failure"].sum()
            failure_rate = (failures / total * 100) if total > 0 else 0

            category_stats.append(
                {
                    "category": category,
                    "failure_rate": failure_rate,
                    "total_samples": total,
                    "failed_samples": int(failures),
                }
            )

        stats_df = pd.DataFrame(category_stats).sort_values(
            "failure_rate", ascending=False
        )

        # Create bar chart
        plt.figure(figsize=(12, 7))

        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(stats_df)))
        bars = plt.bar(
            stats_df["category"],
            stats_df["failure_rate"],
            color=colors,
            edgecolor="black",
            linewidth=1.2,
        )

        # Add value labels on bars
        for i, (bar, row) in enumerate(zip(bars, stats_df.itertuples())):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 1,
                f"{height:.1f}%\n({row.failed_samples}/{row.total_samples})",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        plt.title(
            "Failure Rates by Repair Category\nWhich Prompt Template Produces the Most Failures?",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        plt.xlabel("Repair Category", fontsize=12, fontweight="bold")
        plt.ylabel("Failure Rate (%)", fontsize=12, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, max(stats_df["failure_rate"]) * 1.2)
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.tight_layout()

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"📊 Failure rates by category chart saved to: {output_file}")
        plt.close()

        return stats_df

    def create_per_mode_failure_chart(
        self, output_file: str, baseline_data: pd.DataFrame = None
    ):
        """
        Requirement 3: Per-Mode Failure Trend (Before vs. After)
        Comparison chart showing each failure mode individually
        If baseline_data provided, shows before/after comparison
        """
        # Calculate current failure rates
        current_rates = {}
        for mode in self.FAILURE_MODES:
            rate = (self.df[mode].sum() / len(self.df) * 100) if len(self.df) > 0 else 0
            current_rates[mode] = rate

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(self.FAILURE_MODES))
        width = 0.35

        if baseline_data is not None:
            # Before vs After comparison
            baseline_rates = {}
            for mode in self.FAILURE_MODES:
                rate = (
                    (baseline_data[mode].sum() / len(baseline_data) * 100)
                    if len(baseline_data) > 0
                    else 0
                )
                baseline_rates[mode] = rate

            bars1 = ax.bar(
                x - width / 2,
                [baseline_rates[m] for m in self.FAILURE_MODES],
                width,
                label="Baseline",
                color="#e74c3c",
                edgecolor="black",
                linewidth=1.2,
            )
            bars2 = ax.bar(
                x + width / 2,
                [current_rates[m] for m in self.FAILURE_MODES],
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
                        height + 0.5,
                        f"{height:.1f}%",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                        fontweight="bold",
                    )

            ax.legend(fontsize=11, loc="upper right")
            title = "Per-Mode Failure Rates: Baseline vs. Post-Correction\nShowing Individual Failure Mode Improvements"
        else:
            # Current rates only
            colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(self.FAILURE_MODES)))
            bars = ax.bar(
                x,
                [current_rates[m] for m in self.FAILURE_MODES],
                color=colors,
                edgecolor="black",
                linewidth=1.2,
            )

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            title = "Per-Mode Failure Rates\nIndividual Failure Mode Analysis"

        ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
        ax.set_xlabel("Failure Modes", fontsize=12, fontweight="bold")
        ax.set_ylabel("Failure Rate (%)", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(self.FAILURE_MODES, rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_ylim(0, max(current_rates.values()) * 1.2 if current_rates else 100)

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"📊 Per-mode failure chart saved to: {output_file}")
        plt.close()

    def create_most_problematic_items_chart(self, output_file: str):
        """
        Requirement 4: Most Problematic Items
        Visualization highlighting items with 3+ simultaneous failure flags
        """
        # Calculate failure count per item
        self.df["failure_count"] = self.df[self.FAILURE_MODES].sum(axis=1)

        # Identify problematic items (3+ failures)
        problematic = self.df[self.df["failure_count"] >= 3].copy()
        problematic = problematic.sort_values("failure_count", ascending=False)

        if len(problematic) == 0:
            print("✅ No items with 3+ simultaneous failures found!")
            # Create empty chart with message
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(
                0.5,
                0.5,
                "No items with 3+ simultaneous failures\n✅ Good quality!",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
            )
            ax.axis("off")
            plt.tight_layout()
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            plt.savefig(output_file, dpi=300, bbox_inches="tight")
            plt.close()
            return

        # Create stacked bar chart showing which failures each item has
        fig, ax = plt.subplots(figsize=(14, max(8, len(problematic) * 0.4)))

        # Prepare data for stacked bars
        failure_data = problematic[self.FAILURE_MODES].values
        item_labels = problematic["trace_id"].values

        # Create stacked horizontal bars
        colors = ["#e74c3c", "#e67e22", "#f39c12", "#f1c40f", "#3498db", "#9b59b6"]
        left = np.zeros(len(problematic))

        for i, mode in enumerate(self.FAILURE_MODES):
            values = problematic[mode].values
            ax.barh(
                item_labels,
                values,
                left=left,
                label=mode,
                color=colors[i],
                edgecolor="black",
                linewidth=0.5,
            )
            left += values

        # Add total count labels
        for i, (idx, row) in enumerate(problematic.iterrows()):
            ax.text(
                row["failure_count"] + 0.1,
                i,
                f"{int(row['failure_count'])} failures",
                va="center",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_title(
            f"Most Problematic Items (3+ Simultaneous Failures)\n{len(problematic)} items identified as worst cases",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xlabel("Number of Failures", fontsize=12, fontweight="bold")
        ax.set_ylabel("Sample ID", fontsize=12, fontweight="bold")
        ax.legend(
            title="Failure Modes",
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=9,
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"📊 Most problematic items chart saved to: {output_file}")
        plt.close()

        return problematic

    def identify_common_failures(self) -> Dict[str, Any]:
        """Identifies most common failure types and patterns"""
        analysis = {
            "failure_counts": {},
            "failure_percentages": {},
            "most_common_failure": None,
            "least_common_failure": None,
            "samples_with_multiple_failures": 0,
            "average_failures_per_sample": 0,
        }

        # Count failures
        for mode in self.FAILURE_MODES:
            count = self.df[mode].sum()
            analysis["failure_counts"][mode] = int(count)
            analysis["failure_percentages"][mode] = round(count / len(self.df) * 100, 2)

        # Most/least common
        if analysis["failure_counts"]:
            analysis["most_common_failure"] = max(
                analysis["failure_counts"].items(), key=lambda x: x[1]
            )
            analysis["least_common_failure"] = min(
                analysis["failure_counts"].items(), key=lambda x: x[1]
            )

        # Multiple failures
        failure_sum = self.df[self.FAILURE_MODES].sum(axis=1)
        analysis["samples_with_multiple_failures"] = int((failure_sum > 1).sum())
        analysis["average_failures_per_sample"] = round(failure_sum.mean(), 2)

        return analysis

    def identify_correlations(
        self, threshold: float = 0.3
    ) -> List[Tuple[str, str, float]]:
        """
        Identifies significant correlations between failure modes

        Args:
            threshold: Minimum correlation coefficient to report

        Returns:
            List of (mode1, mode2, correlation) tuples
        """
        corr_matrix = self.df[self.FAILURE_MODES].corr()

        correlations = []
        for i, mode1 in enumerate(self.FAILURE_MODES):
            for j, mode2 in enumerate(self.FAILURE_MODES):
                if i < j:  # Avoid duplicates and self-correlation
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) >= threshold:
                        correlations.append((mode1, mode2, round(corr_value, 3)))

        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x[2]), reverse=True)

        return correlations

    def analyze_by_issue_type(self) -> pd.DataFrame:
        """Analyzes failure modes grouped by issue_type (repair category)"""
        if "issue_type" not in self.df.columns:
            print("⚠️  'issue_type' column not found")
            return None

        # Group by issue type and sum failures
        grouped = self.df.groupby("issue_type")[self.FAILURE_MODES].sum()

        return grouped

    def generate_analysis_report(self, output_file: str):
        """Generates comprehensive analysis report"""
        report = {
            "common_failures": self.identify_common_failures(),
            "correlations": [
                {"mode1": m1, "mode2": m2, "correlation": corr}
                for m1, m2, corr in self.identify_correlations()
            ],
            "issue_type_analysis": None,
        }

        # Add issue type analysis if available
        grouped = self.analyze_by_issue_type()
        if grouped is not None:
            report["issue_type_analysis"] = grouped.to_dict()

        # Save report
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 Analysis report saved to: {output_file}")
        return report

    def print_analysis_summary(self):
        """Print human-readable analysis summary"""
        print("\n" + "=" * 60)
        print("📊 FAILURE MODE ANALYSIS SUMMARY")
        print("=" * 60)

        # Common failures
        common = self.identify_common_failures()
        print(f"\nTotal samples analyzed: {len(self.df)}")
        print(f"Average failures per sample: {common['average_failures_per_sample']}")
        print(
            f"Samples with multiple failures: {common['samples_with_multiple_failures']}"
        )

        print("\n🔴 Most Common Failure Modes:")
        sorted_failures = sorted(
            common["failure_counts"].items(), key=lambda x: x[1], reverse=True
        )
        for mode, count in sorted_failures[:3]:
            pct = common["failure_percentages"][mode]
            print(f"  {mode}: {count} samples ({pct}%)")

        # Correlations
        print("\n🔗 Significant Correlations (|r| >= 0.3):")
        correlations = self.identify_correlations(threshold=0.3)
        if correlations:
            for mode1, mode2, corr in correlations[:5]:
                direction = "positive" if corr > 0 else "negative"
                print(f"  {mode1} ↔ {mode2}: {corr} ({direction})")
        else:
            print("  No significant correlations found")

        # Issue type analysis
        print("\n📦 Failure Modes by Repair Category:")
        grouped = self.analyze_by_issue_type()
        if grouped is not None:
            for issue_type in grouped.index:
                total_failures = grouped.loc[issue_type].sum()
                print(f"  {issue_type}: {int(total_failures)} total failures")

        print("\n" + "=" * 60)

    def create_all_visualizations(
        self, output_dir: str, baseline_data: pd.DataFrame = None
    ):
        """
        Creates all required visualizations per requirements document

        Args:
            output_dir: Directory to save visualization files
            baseline_data: Optional baseline DataFrame for before/after comparison
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n📊 Creating visualizations per requirements...")

        # 1. Failure Mode Co-occurrence (diverging color scale)
        cooccurrence_file = os.path.join(
            output_dir, "01_failure_cooccurrence_heatmap.png"
        )
        self.create_failure_cooccurrence_heatmap(cooccurrence_file)

        # 2. Failure Rates by Repair Category (sequential color scale)
        category_file = os.path.join(output_dir, "02_failure_rates_by_category.png")
        self.create_failure_rates_by_category_chart(category_file)

        # 3. Per-Mode Failure Trend (before vs after if baseline provided)
        per_mode_file = os.path.join(output_dir, "03_per_mode_failure_trend.png")
        self.create_per_mode_failure_chart(per_mode_file, baseline_data)

        # 4. Most Problematic Items (3+ failures)
        problematic_file = os.path.join(output_dir, "04_most_problematic_items.png")
        self.create_most_problematic_items_chart(problematic_file)

        return {
            "failure_cooccurrence": cooccurrence_file,
            "failure_by_category": category_file,
            "per_mode_trend": per_mode_file,
            "problematic_items": problematic_file,
        }

    def run_full_analysis(self, output_dir: str, baseline_data: pd.DataFrame = None):
        """
        Runs complete failure analysis: visualizations + report + summary

        Args:
            output_dir: Directory to save all output files
            baseline_data: Optional baseline DataFrame for comparison

        Returns:
            Dictionary with paths to all generated files
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "=" * 60)
        print("📈 FAILURE PATTERN ANALYSIS")
        print("=" * 60)

        # Create all required visualizations
        viz_files = self.create_all_visualizations(output_dir, baseline_data)

        # Generate comprehensive report
        report_file = os.path.join(output_dir, "failure_analysis_report.json")
        self.generate_analysis_report(report_file)

        # Print analysis summary
        self.print_analysis_summary()

        # Return all file paths
        output_files = {**viz_files, "analysis_report": report_file}

        print("\n✅ Failure Analysis Complete!")
        print(f"\n📁 Output files:")
        for name, path in output_files.items():
            print(f"   - {path}")

        return output_files
