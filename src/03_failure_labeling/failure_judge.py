"""
LLM-as-Judge for Failure Mode Detection
Evaluates each sample for 6 binary failure modes
"""

import json
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from pydantic_classes import DIYRepairSyntheticItem
from .pydantic_classes import FailureJudgment, DIYRepairWithFailureLabels
from pipeline_core.llm import chat as llm_chat
from pipeline_core.utils import try_parse_json
from braintrust import current_span, traced
from ._judge_prompt import judge_prompt


class FailureJudge:
    """LLM-as-Judge for binary failure mode detection only"""

    FAILURE_MODES = [
        "incomplete_answer",
        "safety_violations",
        "unrealistic_tools",
        "overcomplicated_solution",
        "missing_context",
        "poor_quality_tips",
    ]

    # Remove QUALITY_DIMENSIONS - that's Phase 4

    def __init__(self, input_data: List[Dict[str, Any]]):
        self.phase_name = "03_failure_labeling"
        self.judge_prompt = self._create_judge_prompt()
        self.samples = [
            DIYRepairSyntheticItem(**item)
            for item in input_data
            if item.get("error") is None
        ]
        print(f"📊 Loaded {len(self.samples)} samples from Phase 02")

    def _create_judge_prompt(self) -> str:
        """Create the failure detection prompt"""
        return judge_prompt

    @traced(type="llm", name="failure_judge_evaluation", notrace_io=True)
    def evaluate(
        self, sample: DIYRepairSyntheticItem, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[FailureJudgment], str]:
        """
        Evaluate one sample for failure modes only
        Returns: (FailureJudgment, raw_response)
        """
        if params is None:
            params = {
                "max_tokens": 500,
                "temperature": 0.2,
            }

        # Format prompt
        prompt = self.judge_prompt.format(
            question=sample.question or "N/A",
            answer=sample.answer or "N/A",
            equipment_problem=sample.equipment_problem or "N/A",
            tools_required=(
                ", ".join(sample.tools_required) if sample.tools_required else "N/A"
            ),
            steps=(
                "\n".join(f"{i+1}. {s}" for i, s in enumerate(sample.steps))
                if sample.steps
                else "N/A"
            ),
            safety_info=sample.safety_info or "N/A",
            tips=sample.tips or "N/A",
        )

        metadata = {
            "phase": self.phase_name,
            "sample_id": sample.id,
            "issue_type": sample.metadata.issue_type,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            # Call LLM judge
            # raw_response = llm_chat(
            #     prompt=prompt,
            #     params=params,
            #     metadata=metadata,
            # )
            raw_response = """
            {
  "incomplete_answer": 0,
  "metadata": {
    "issue_type": "general_home_repair",
    "user_query": "The cabinet door fell off because the hinge screws pulled out of the wood."
  },
  "missing_context": 0,
  "overall_failure": false,
  "overcomplicated_solution": 0,
  "poor_quality_tips": 0,
  "reasoning": "The answer is complete, addresses the problem directly, includes necessary safety precautions, and provides realistic tools and actionable steps that match the skill level of typical DIYers. Additionally, the tips provided are practical and improve the overall quality of the guidance.",
  "safety_violations": 0,
  "trace_id": "qa_004",
  "unrealistic_tools": 0
}
"""
            # Parse JSON
            ok, data = try_parse_json(raw_response)
            if not ok:
                print(f"ERROR - [{self.phase_name}] JSON parse failed for {sample.id}")
                return None, raw_response

            # Extract only failure mode fields (filter out quality fields)
            failure_data = {
                "trace_id": sample.id,
                "incomplete_answer": data.get("incomplete_answer", 0),
                "safety_violations": data.get("safety_violations", 0),
                "unrealistic_tools": data.get("unrealistic_tools", 0),
                "overcomplicated_solution": data.get("overcomplicated_solution", 0),
                "missing_context": data.get("missing_context", 0),
                "poor_quality_tips": data.get("poor_quality_tips", 0),
                "overall_failure": data.get("overall_failure", False),
                "reasoning": data.get("reasoning", ""),
            }

            # Validate with Pydantic
            judgment = FailureJudgment(**failure_data)

            # Log to Braintrust
            self._log_braintrust(sample, judgment, raw_response)

            return judgment, raw_response

        except Exception as e:
            print(f"ERROR - [{self.phase_name}] Error evaluating {sample.id}: {e}")
            return None, ""

    def _log_braintrust(
        self,
        sample: DIYRepairSyntheticItem,
        judgment: FailureJudgment,
        raw_response: str,
    ):
        """Log failure judgment to Braintrust"""
        try:
            failure_count = sum(
                [
                    judgment.incomplete_answer,
                    judgment.safety_violations,
                    judgment.unrealistic_tools,
                    judgment.overcomplicated_solution,
                    judgment.missing_context,
                    judgment.poor_quality_tips,
                ]
            )

            current_span().log(
                input={
                    "sample_id": sample.id,
                    "question": sample.question,
                    "issue_type": sample.metadata.issue_type,
                },
                output=judgment.model_dump(),
                metrics={
                    "failure_count": failure_count,
                    "overall_failure": 1 if judgment.overall_failure else 0,
                },
                metadata={
                    "sample_id": sample.id,
                },
            )
        except Exception as e:
            print(f"⚠️  Braintrust logging failed: {e}")

    def evaluate_dataset(
        self,
    ) -> Tuple[List, Dict[str, Any]]:
        """
        Evaluate all samples and return labeled samples with statistics
        Returns: (labeled_samples, failure_stats)

        labeled_samples is a list of items extended with failure judgments
        """
        labeled_samples = []
        failure_stats = {
            "total_evaluated": 0,
            "total_passed_failure": 0,
            "total_failed_failure": 0,
            "failure_mode_counts": {mode: 0 for mode in self.FAILURE_MODES},
            "evaluation_errors": 0,
        }

        print(f"\n🔍 Evaluating {len(self.samples)} samples with LLM-as-Judge...")
        print("-" * 60)

        for idx, sample in enumerate(self.samples, 1):
            print(f"\n[{idx}/{len(self.samples)}] Evaluating {sample.id}...")

            judgment, raw_response = self.evaluate(sample)

            if judgment:
                # Wrap sample with failure judgment
                labeled_sample = DIYRepairWithFailureLabels(
                    **sample.model_dump(), failure_judgment=judgment
                )
                labeled_samples.append(labeled_sample)

                # Update stats
                failure_stats["total_evaluated"] += 1

                # Failure mode stats
                if not judgment.overall_failure:
                    failure_stats["total_passed_failure"] += 1
                    print(f"  ✅ FAILURE CHECK: PASS")
                else:
                    failure_stats["total_failed_failure"] += 1
                    flagged = [
                        mode
                        for mode in self.FAILURE_MODES
                        if getattr(judgment, mode) == 1
                    ]
                    print(f"  ❌ FAILURE CHECK: FAIL - {', '.join(flagged)}")
                    for mode in flagged:
                        failure_stats["failure_mode_counts"][mode] += 1
            else:
                print(f"  ⚠️  Evaluation error - keeping sample without judgment")
                # Create error judgment
                error_judgment = FailureJudgment(
                    trace_id=sample.id, reasoning="Evaluation error"
                )
                labeled_sample = DIYRepairWithFailureLabels(
                    **sample.model_dump(), failure_judgment=error_judgment
                )
                labeled_samples.append(labeled_sample)
                failure_stats["evaluation_errors"] += 1

        return labeled_samples, failure_stats

    def print_summary(self, failure_stats: Dict[str, Any]):
        """Print failure labeling summary statistics"""
        print("\n" + "=" * 60)
        print("📊 FAILURE LABELING SUMMARY")
        print("=" * 60)
        print(f"Total Evaluated: {failure_stats['total_evaluated']}")

        print("\n🚨 FAILURE MODE RESULTS:")
        print(
            f"  Passed: {failure_stats['total_passed_failure']} "
            f"({failure_stats['total_passed_failure']/max(1, failure_stats['total_evaluated'])*100:.1f}%)"
        )
        print(
            f"  Failed: {failure_stats['total_failed_failure']} "
            f"({failure_stats['total_failed_failure']/max(1, failure_stats['total_evaluated'])*100:.1f}%)"
        )

        print("\n  Failure Mode Breakdown:")
        for mode, count in failure_stats["failure_mode_counts"].items():
            if count > 0:
                pct = count / max(1, failure_stats["total_evaluated"]) * 100
                print(f"    • {mode}: {count} ({pct:.1f}%)")
