"""
Phase 04: Quality Evaluation Runner
Evaluates each sample using LLM-as-Judge for 8 quality dimensions
"""

import os
from typing import Any
from pipeline_core.runner import PhaseRunner
from pipeline_core.io import save_report
from .quality_judge import QualityJudge
from .quality_analyzer import QualityAnalyzer


def run_phase(input_data: Any, output_path: str):
    """
    Phase 04: Quality Evaluation
    - Load failure-labeled samples from Phase 03
    - Evaluate each with LLM-as-Judge for 8 quality dimensions
    - Analyze quality patterns and create visualizations
    - Output samples with quality scores attached
    """
    print("Running Phase 04: Quality Evaluation")

    # Initialize judge and evaluate
    judge = QualityJudge(input_data)
    quality_samples, quality_stats = judge.evaluate_dataset()

    # Print summary
    judge.print_summary(quality_stats)

    # Save quality statistics
    analysis_path = f"{output_path}/analysis"
    save_report(quality_stats, analysis_path, "quality_statistics.json")

    # Run quality pattern analysis
    analyzer = QualityAnalyzer(quality_samples)
    analyzer.run_full_analysis(analysis_path)

    return quality_samples


if __name__ == "__main__":
    PhaseRunner(
        phase_name="04_quality_evaluation",
        input_phase="phase_03",
        output_phase="phase_04",
        run_fn=run_phase,
        run_fn_kwargs={"output_path": "data/phase_04"},
    ).run()
