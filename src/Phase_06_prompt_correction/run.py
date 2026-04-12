"""
Phase 06: Prompt Correction & Re-evaluation Runner
Implements 12_prompt_correction_strategy.md:
  1. Identify dominant failure modes
  2. Find correlated failures (shared root cause)
  3. Pinpoint responsible template(s) by category
  4. Write targeted corrections (only affected categories)
  5. Document every change traceably
  6. Save corrected prompts — re-run pipeline to measure improvement
"""

from typing import Any
from ..pipeline_core.runner import PhaseRunner
from ..pipeline_core.io import save_report
from .prompt_corrector import PromptCorrector
from .iteration_logger import IterationLogger
from .pydantic_classes import DIYRepairWithCorrectionMetadata, CorrectionMetadata


def run_phase(input_data: Any, output_path: str):
    corrector = PromptCorrector()

    # Step 1 — Load Phase 05 analysis report
    analysis_report = corrector.load_analysis_report(
        "src/data/phase_05/analysis/combined_analysis_report.json"
    )

    # Step 1 — Identify dominant failure modes by frequency
    priority_failures = corrector.identify_priority_failures(analysis_report)
    active = [(m, c) for m, c in priority_failures if c > 0]
    print(f"\nDominant failure modes: {', '.join(m for m, _ in active) or 'none'}")

    # Step 2 — Find correlated failures (shared root cause via Phase 05)
    correlated_failures = corrector.find_correlated_failures(analysis_report)
    if correlated_failures:
        for dim, modes in correlated_failures.items():
            print(f"Correlated via '{dim}': {', '.join(modes)}")

    # Step 3 — Pinpoint which categories produced failures
    category_failures = corrector.failures_by_category(input_data)
    print(f"Categories with failures: {list(category_failures.keys()) or 'none'}")

    # Step 1 — Extract baseline stats from real data
    baseline_stats = corrector.extract_failure_stats(input_data)

    # Step 4 & 5 — Generate targeted, traceable corrections
    corrected_prompts = corrector.generate_corrected_prompts(
        corrector.load_original_prompts(),
        category_failures,
        correlated_failures,
        priority_failures,
    )
    save_report(corrected_prompts, f"{output_path}/analysis", "corrected_prompts.json")

    # Step 6 — Generate correction report (plan for re-run, not simulation)
    report = corrector.generate_correction_report(
        baseline_stats,
        category_failures,
        correlated_failures,
        f"{output_path}/analysis/prompt_correction_report.json",
    )
    corrector.print_correction_summary(report, category_failures, correlated_failures)

    IterationLogger.append_to_iteration_log(
        iteration_num=1,
        analysis_summary=(
            f"Corrected {len(category_failures)} category(ies) based on "
            f"{len(corrector.corrections)} traceable changes. "
            f"Baseline failure rate: {baseline_stats['failure_rate']:.1%}. "
            f"Re-run pipeline to measure improvement."
        ),
        corrections=corrector.corrections,
        improvement_ratio=0.0,
        success=False,
    )

    output_items = []
    for item in input_data:
        item_category = item.get("metadata", {}).get("issue_type", "unknown")
        meta = (
            CorrectionMetadata(
                iteration_num=1,
                corrections_applied=corrector.corrections,
                baseline_failure_rate=round(baseline_stats["failure_rate"], 3),
                categories_affected=list(category_failures.keys()),
                corrected_prompts_path=f"{output_path}/analysis/corrected_prompts.json",
            )
            if item_category in category_failures
            else None
        )
        output_items.append(
            DIYRepairWithCorrectionMetadata(**item, correction_metadata=meta)
        )

    return output_items


if __name__ == "__main__":
    PhaseRunner(
        phase_name="06_prompt_correction",
        input_phase="phase_04",
        output_phase="phase_06",
        run_fn=run_phase,
        run_fn_kwargs={"output_path": "src/data/phase_06"},
    ).run()
