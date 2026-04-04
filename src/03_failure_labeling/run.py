"""
Phase 03: Failure Labeling Runner
Evaluates each sample using LLM-as-Judge for 6 binary failure modes
"""

from typing import Any, List
from pipeline_core.runner import PhaseRunner
from pipeline_core.io import save_report
from .failure_judge import FailureJudge
from .failure_analyzer import FailureAnalyzer


def run_phase(input_data: Any, output_path: str):
    """
    Phase 03: Failure Labeling
    - Load validated samples from Phase 02
    - Evaluate each with LLM-as-Judge for 6 failure modes
    - Analyze patterns and create visualizations
    - Output samples with failure labels attached
    """
    print("Running Phase 03: Failure Labeling")

    # Initialize judge and evaluate
    judge = FailureJudge(input_data)
    labeled_samples, failure_stats = judge.evaluate_dataset()

    # Print summary
    judge.print_summary(failure_stats)

    # Save failure statistics
    analysis_path = f"{output_path}/analysis"
    save_report(failure_stats, analysis_path, "failure_statistics.json")
    print(f"\n💾 Saved failure statistics to {analysis_path}/failure_statistics.json")

    # Run failure pattern analysis
    analyzer = FailureAnalyzer(labeled_samples)
    analyzer.run_full_analysis(analysis_path)

    return labeled_samples


if __name__ == "__main__":
    PhaseRunner(
        phase_name="03_failure_labeling",
        input_phase="phase_02",
        output_phase="phase_03",
        run_fn=run_phase,
        run_fn_kwargs={"output_path": "data/phase_03"},
    ).run()
