"""
Phase 05: Failure & Quality Analysis Runner
Aggregates failure modes and quality dimensions for diagnostic insights
"""

import os
from typing import Any
from pipeline_core.runner import PhaseRunner
from pipeline_core.io import save_report
from .combined_analyzer import CombinedAnalyzer


def run_phase(input_data: Any, output_path: str):
    """
    Phase 05: Failure & Quality Analysis
    - Load items with failure labels and quality scores from Phase 04
    - Correlate failure modes with quality dimensions
    - Identify patterns, worst categories, and root causes
    - Output items unchanged (pass-through) for Phase 06
    """
    print("Running Phase 05: Failure & Quality Analysis")

    # Initialize analyzer and run full analysis
    analyzer = CombinedAnalyzer(input_data)
    analysis_files = analyzer.run_full_analysis(f"{output_path}/analysis")

    # Items pass through unchanged
    return None


if __name__ == "__main__":
    PhaseRunner(
        phase_name="05_analysis",
        input_phase="phase_04",
        output_phase="phase_05",
        run_fn=run_phase,
        run_fn_kwargs={"output_path": "data/phase_05"},
    ).run()
