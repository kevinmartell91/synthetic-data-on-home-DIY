"""
Failure Mode Analysis Module
Creates heatmaps, identifies patterns, and correlations
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Tuple


class FailureAnalyzer:
    """Analyzes failure mode patterns and correlations"""
    
    FAILURE_MODES = [
        "incomplete_answer",
        "safety_violations",
        "unrealistic_tools",
        "overcomplicated_solution",
        "missing_context",
        "poor_quality_tips"
    ]
    
    def __init__(self, labeled_df: pd.DataFrame):
        self.df = labeled_df
        self.failure_cols = [f"failure_{mode}" for mode in self.FAILURE_MODES]
        
        # Verify columns exist
        missing = [col for col in self.failure_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing failure columns: {missing}")
    
    def create_failure_heatmap(self, output_file: str = "failure_heatmap.png"):
        """
        Creates heatmap showing failure modes across samples
        Rows = samples, Columns = failure modes
        """
        # Extract failure mode data
        failure_data = self.df[self.failure_cols].copy()
        failure_data.columns = self.FAILURE_MODES  # Cleaner labels
        
        # Create figure
        plt.figure(figsize=(12, max(8, len(self.df) * 0.3)))
        
        # Create heatmap
        sns.heatmap(
            failure_data,
            cmap="RdYlGn_r",  # Red for failures, green for success
            cbar_kws={"label": "Failure (1) / Success (0)"},
            yticklabels=self.df["trace_id"].values,
            xticklabels=self.FAILURE_MODES,
            linewidths=0.5,
            linecolor='gray'
        )
        
        plt.title("Failure Mode Heatmap Across Samples", fontsize=16, fontweight='bold')
        plt.xlabel("Failure Modes", fontsize=12)
        plt.ylabel("Sample Trace ID", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Failure heatmap saved to: {output_file}")
        plt.close()
    
    def create_correlation_matrix(self, output_file: str = "failure_correlation.png"):
        """
        Creates correlation matrix showing relationships between failure modes
        E.g., does overcomplicated_solution correlate with unrealistic_tools?
        """
        # Calculate correlation matrix
        corr_matrix = self.df[self.failure_cols].corr()
        
        # Rename columns for cleaner display
        corr_matrix.columns = self.FAILURE_MODES
        corr_matrix.index = self.FAILURE_MODES
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            corr_matrix,
            annot=True,  # Show correlation values
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"label": "Correlation Coefficient"}
        )
        
        plt.title("Failure Mode Correlation Matrix", fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Correlation matrix saved to: {output_file}")
        plt.close()
        
        return corr_matrix
    
    def identify_common_failures(self) -> Dict[str, Any]:
        """
        Identifies most common failure types and patterns
        """
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
            col = f"failure_{mode}"
            count = self.df[col].sum()
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
        failure_sum = self.df[self.failure_cols].sum(axis=1)
        analysis["samples_with_multiple_failures"] = int((failure_sum > 1).sum())
        analysis["average_failures_per_sample"] = round(failure_sum.mean(), 2)
        
        return analysis
    
    def identify_correlations(self, threshold: float = 0.3) -> List[Tuple[str, str, float]]:
        """
        Identifies significant correlations between failure modes
        
        Args:
            threshold: Minimum correlation coefficient to report
        
        Returns:
            List of (mode1, mode2, correlation) tuples
        """
        corr_matrix = self.df[self.failure_cols].corr()
        
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
        """
        Analyzes failure modes grouped by issue_type
        """
        if "issue_type" not in self.df.columns:
            print("⚠️  'issue_type' column not found")
            return None
        
        # Group by issue type and sum failures
        grouped = self.df.groupby("issue_type")[self.failure_cols].sum()
        grouped.columns = self.FAILURE_MODES
        
        return grouped
    
    def create_issue_type_heatmap(self, output_file: str = "failure_by_issue_type.png"):
        """
        Creates heatmap of failure modes by issue type
        """
        grouped = self.analyze_by_issue_type()
        
        if grouped is None:
            return
        
        plt.figure(figsize=(12, 6))
        
        sns.heatmap(
            grouped,
            annot=True,
            fmt="g",
            cmap="YlOrRd",
            linewidths=1,
            cbar_kws={"label": "Failure Count"}
        )
        
        plt.title("Failure Modes by Issue Type", fontsize=16, fontweight='bold')
        plt.xlabel("Failure Modes", fontsize=12)
        plt.ylabel("Issue Type", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Issue type heatmap saved to: {output_file}")
        plt.close()
    
    def generate_analysis_report(self, output_file: str = "failure_analysis_report.json"):
        """
        Generates comprehensive analysis report
        """
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
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Analysis report saved to: {output_file}")
        return report
    
    def print_analysis_summary(self):
        """Print human-readable analysis summary"""
        print("\n" + "="*60)
        print("📊 FAILURE MODE ANALYSIS SUMMARY")
        print("="*60)
        
        # Common failures
        common = self.identify_common_failures()
        print(f"\nTotal samples analyzed: {len(self.df)}")
        print(f"Average failures per sample: {common['average_failures_per_sample']}")
        print(f"Samples with multiple failures: {common['samples_with_multiple_failures']}")
        
        print("\n🔴 Most Common Failure Modes:")
        sorted_failures = sorted(
            common['failure_counts'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for mode, count in sorted_failures[:3]:
            pct = common['failure_percentages'][mode]
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
        
        print("\n" + "="*60)


def main():
    """Standalone analysis script"""
    import sys
    
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pass_number = "03"
    folder_name = f"eval_pass_{pass_number}"
    labeled_csv = os.path.join(script_dir, folder_name, "labeled_dataset.csv")
    
    # Output files
    heatmap_file = os.path.join(script_dir, folder_name, "failure_heatmap.png")
    correlation_file = os.path.join(script_dir, folder_name, "failure_correlation.png")
    issue_type_file = os.path.join(script_dir, folder_name, "failure_by_issue_type.png")
    report_file = os.path.join(script_dir, folder_name, "failure_analysis_report.json")
    
    # Load labeled dataset
    print(f"📂 Loading labeled dataset from: {labeled_csv}")
    try:
        df = pd.read_csv(labeled_csv)
        print(f"✅ Loaded {len(df)} samples")
    except FileNotFoundError:
        print(f"❌ Labeled dataset not found: {labeled_csv}")
        print("   Run failure_labeler.py first!")
        sys.exit(1)
    
    # Initialize analyzer
    print("\n🔍 Initializing Failure Analyzer...")
    analyzer = FailureAnalyzer(df)
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    analyzer.create_failure_heatmap(heatmap_file)
    analyzer.create_correlation_matrix(correlation_file)
    analyzer.create_issue_type_heatmap(issue_type_file)
    
    # Generate report
    print("\n📋 Generating analysis report...")
    report = analyzer.generate_analysis_report(report_file)
    
    # Print summary
    analyzer.print_analysis_summary()
    
    print("\n✅ Analysis complete!")
    print(f"\n📁 Output files:")
    print(f"   - {heatmap_file}")
    print(f"   - {correlation_file}")
    print(f"   - {issue_type_file}")
    print(f"   - {report_file}")


if __name__ == "__main__":
    main()