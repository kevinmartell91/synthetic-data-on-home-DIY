"""
Prompt Correction Logic — implements 12_prompt_correction_strategy.md

Steps:
  1. Identify dominant failure modes by rate
  2. Find correlated failures (shared root cause)
  3. Pinpoint responsible template(s) by category
  4. Write targeted corrections (only affected categories)
  5. Document every change traceably
  6. Produce corrected prompts ready for re-run
"""

import json
import os
from typing import Dict, Any, List, Tuple
from datetime import datetime
from braintrust import current_span, traced


FAILURE_MODE_HINTS = {
    "incomplete_answer": (
        "Every answer MUST include all steps needed to complete the repair from start to finish. "
        "Do not skip steps or assume prior knowledge. Each step must have a clear action verb."
    ),
    "safety_violations": (
        "You MUST name the specific hazard AND the specific precaution for this task "
        "(e.g. 'Turn off the circuit breaker at the panel before touching any wiring'). "
        "Generic warnings like 'be careful' are not acceptable."
    ),
    "unrealistic_tools": (
        "Only list tools a homeowner already owns or can buy at a hardware store for under $50. "
        "Do not list trade-only equipment or specialty tools."
    ),
    "overcomplicated_solution": (
        "Provide the simplest approach a capable homeowner can safely complete. "
        "If any step requires a licensed professional, say so explicitly rather than including it."
    ),
    "missing_context": (
        "Explain WHY each major step is necessary, not just what to do. "
        "Do not assume the reader has prior repair experience."
    ),
    "poor_quality_tips": (
        "Tips must be non-obvious and specific to this repair type. "
        "Do not repeat information already in the steps. Each tip must add new value."
    ),
}


class PromptCorrector:
    """
    Implements the prompt correction strategy:
    data-driven, per-category, fully traceable.
    """

    def __init__(self):
        self.corrections: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Step 1 — Load Phase 05 report
    # ------------------------------------------------------------------

    def load_analysis_report(self, report_path: str) -> Dict[str, Any]:
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️  Analysis report not found at {report_path}")
            return {}

    # ------------------------------------------------------------------
    # Step 1 — Identify dominant failure modes
    # ------------------------------------------------------------------

    def identify_priority_failures(
        self, report: Dict[str, Any]
    ) -> List[Tuple[str, int]]:
        """Return failure modes sorted by frequency (highest first)."""
        failure_counts = report.get("failure_mode_frequencies", {})
        return sorted(failure_counts.items(), key=lambda x: x[1], reverse=True)

    # ------------------------------------------------------------------
    # Step 2 — Find correlated failures from Phase 05 correlations
    # ------------------------------------------------------------------

    def find_correlated_failures(
        self, report: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Group failure modes that share a correlated quality dimension.
        Returns {quality_dimension: [failure_modes]} for dimensions
        where correlation > 0 for more than one failure mode.
        """
        pairs = report.get("top_failure_quality_pairs", [])
        dim_to_modes: Dict[str, List[str]] = {}
        for pair in pairs:
            mode = pair.get("failure_mode", "")
            dim = pair.get("quality_dimension", "")
            corr = pair.get("correlation", 0)
            if corr > 0 and mode and dim:
                dim_to_modes.setdefault(dim, [])
                if mode not in dim_to_modes[dim]:
                    dim_to_modes[dim].append(mode)
        # Only keep dimensions with multiple correlated modes (shared root cause)
        return {dim: modes for dim, modes in dim_to_modes.items() if len(modes) > 1}

    # ------------------------------------------------------------------
    # Step 3 — Pinpoint responsible categories
    # ------------------------------------------------------------------

    def failures_by_category(self, items: List[Dict]) -> Dict[str, Dict[str, int]]:
        """
        Returns {category: {failure_mode: count}} for every item in the dataset.
        Only categories with at least one failure are included.
        """
        modes = list(FAILURE_MODE_HINTS.keys())
        result: Dict[str, Dict[str, int]] = {}
        for item in items:
            cat = item.get("metadata", {}).get("issue_type", "unknown")
            fj = item.get("failure_judgment", {})
            if cat not in result:
                result[cat] = {m: 0 for m in modes}
                result[cat]["_total"] = 0
            result[cat]["_total"] += 1
            for mode in modes:
                result[cat][mode] += fj.get(mode, 0)
        # Filter to only categories that have at least one failure
        return {
            cat: counts
            for cat, counts in result.items()
            if any(counts[m] > 0 for m in modes)
        }

    # ------------------------------------------------------------------
    # Step 4 & 5 — Write targeted corrections, document every change
    # ------------------------------------------------------------------

    def load_original_prompts(self) -> Dict[str, str]:
        try:
            from src.Phase_01_generation._templates import templates
            return templates
        except Exception:
            return {
                "appliance_repair": "Generate a Q&A about appliance repair.",
                "plumbing_repair": "Generate a Q&A about plumbing repair.",
                "electrical_repair": "Generate a Q&A about electrical repair.",
                "hvac_repair": "Generate a Q&A about HVAC repair.",
                "general_home_repair": "Generate a Q&A about general home repair.",
            }

    def generate_corrected_prompts(
        self,
        original_prompts: Dict[str, str],
        category_failures: Dict[str, Dict[str, int]],
        correlated_failures: Dict[str, List[str]],
        priority_failures: List[Tuple[str, int]],
    ) -> Dict[str, str]:
        """
        Step 4: Modify ONLY categories that produced failures.
        Step 5: Record every change with reason and template affected.
        """
        corrected = {}
        modes = list(FAILURE_MODE_HINTS.keys())

        for category, original_prompt in original_prompts.items():
            if category not in category_failures:
                # No failures — leave template untouched
                corrected[category] = original_prompt
                continue

            cat_counts = category_failures[category]
            total = cat_counts.get("_total", 1)
            active_modes = [m for m in modes if cat_counts.get(m, 0) > 0]

            corrected_prompt = original_prompt

            # Add targeted instruction for each active failure mode
            for mode in active_modes:
                count = cat_counts[mode]
                rate = count / total
                hint = FAILURE_MODE_HINTS[mode]
                corrected_prompt += f"\n\n[CORRECTION — {mode}]: {hint}"

                self.corrections.append({
                    "change": f"Added explicit instruction targeting '{mode}'",
                    "reason": (
                        f"'{mode}' failure rate was {rate:.0%} in baseline "
                        f"({count}/{total} items in {category})"
                    ),
                    "template_affected": category,
                })

            # If correlated failures share a root cause, call it out explicitly
            for dim, corr_modes in correlated_failures.items():
                overlap = [m for m in corr_modes if m in active_modes]
                if len(overlap) > 1:
                    corrected_prompt += (
                        f"\n\n[ROOT CAUSE NOTE — {dim}]: The failures "
                        f"{', '.join(overlap)} are correlated. "
                        f"Improving specificity in '{dim}' addresses all of them simultaneously."
                    )
                    self.corrections.append({
                        "change": f"Added root-cause note linking {overlap} via '{dim}'",
                        "reason": (
                            f"Phase 05 correlation analysis shows these modes share "
                            f"'{dim}' as a common quality gap"
                        ),
                        "template_affected": category,
                    })

            corrected[category] = corrected_prompt

        return corrected

    # ------------------------------------------------------------------
    # Step 1 (baseline) — Extract real failure stats from dataset
    # ------------------------------------------------------------------

    def extract_failure_stats(self, items: List[Dict]) -> Dict[str, Any]:
        """Extract real failure statistics from Phase 04 output (list of dicts)."""
        modes = list(FAILURE_MODE_HINTS.keys())
        total = len(items)
        overall_failures = sum(
            1 for item in items
            if item.get("failure_judgment", {}).get("overall_failure", False)
        )
        mode_counts = {
            mode: sum(item.get("failure_judgment", {}).get(mode, 0) for item in items)
            for mode in modes
        }
        return {
            "total_evaluated": total,
            "total_failed": overall_failures,
            "failure_rate": overall_failures / total if total > 0 else 0,
            "failure_mode_counts": mode_counts,
        }

    # ------------------------------------------------------------------
    # Step 6 — Generate correction report (plan, not simulated outcome)
    # ------------------------------------------------------------------

    @traced(type="llm", name="prompt_correction_analysis", notrace_io=True)
    def generate_correction_report(
        self,
        baseline_stats: Dict[str, Any],
        category_failures: Dict[str, Dict[str, int]],
        correlated_failures: Dict[str, List[str]],
        output_path: str,
    ) -> Dict[str, Any]:
        """
        Produce a traceable correction plan.
        Improvement ratio is not simulated — it will be measured after re-run.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "baseline": {
                "total_evaluated": baseline_stats["total_evaluated"],
                "total_failed": baseline_stats["total_failed"],
                "failure_rate": round(baseline_stats["failure_rate"], 3),
                "failure_mode_counts": baseline_stats["failure_mode_counts"],
            },
            "categories_affected": list(category_failures.keys()),
            "categories_unchanged": [],
            "correlated_root_causes": {
                dim: {"shared_failure_modes": modes, "action": f"Improve '{dim}' specificity"}
                for dim, modes in correlated_failures.items()
            },
            "corrections_applied": self.corrections,
            "success_criteria": {
                "requirement": ">80% reduction in overall failure rate",
                "status": "PENDING — re-run pipeline with corrected prompts to measure",
            },
            "next_step": (
                "Re-run Phase 01 with corrected prompts, then Phases 02-04, "
                "and compare failure rates against baseline above."
            ),
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"📋 Correction report saved to: {output_path}")

        current_span().log(
            input={
                "total_evaluated": baseline_stats["total_evaluated"],
                "failure_rate": baseline_stats["failure_rate"],
                "categories_affected": list(category_failures.keys()),
            },
            output={
                "corrections_count": len(self.corrections),
                "categories_corrected": report["categories_affected"],
            },
            metadata={
                "phase": "06_prompt_correction",
                "baseline_failure_rate": round(baseline_stats["failure_rate"], 3),
                "corrections_applied": len(self.corrections),
            },
        )

        return report

    def print_correction_summary(
        self,
        report: Dict[str, Any],
        category_failures: Dict[str, Dict[str, int]],
        correlated_failures: Dict[str, List[str]],
    ):
        modes = list(FAILURE_MODE_HINTS.keys())
        print("\n" + "=" * 80)
        print("📊 PROMPT CORRECTION SUMMARY")
        print("=" * 80)

        b = report["baseline"]
        print(f"\nBaseline: {b['failure_rate']:.1%} failure rate "
              f"({b['total_failed']}/{b['total_evaluated']} items)")

        print(f"\nCategories requiring correction: {report['categories_affected'] or 'none'}")
        for cat, counts in category_failures.items():
            total = counts.get("_total", 1)
            active = {m: counts[m] for m in modes if counts[m] > 0}
            print(f"  {cat}: {active} (out of {total} items)")

        if correlated_failures:
            print("\nCorrelated root causes (shared quality gap):")
            for dim, corr_modes in correlated_failures.items():
                print(f"  {dim} → {', '.join(corr_modes)}")

        print(f"\nCorrections applied: {len(self.corrections)}")
        for c in self.corrections:
            print(f"  • [{c['template_affected']}] {c['change']}")
            print(f"    Reason: {c['reason']}")

        print(f"\n⏭  Next step: {report['next_step']}")
        print("=" * 80)
