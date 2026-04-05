"""
LLM-as-Judge for Quality Dimension Scoring
Evaluates each sample across 8 quality dimensions
"""

import json
from typing import Dict, Any, Optional, Tuple, List
from ..Phase_03_failure_labeling.pydantic_classes import DIYRepairWithFailureLabels
from .pydantic_classes import (
    QualityJudgment,
    QualityScores,
    DIYRepairWithQualityScores,
)
from ..pipeline_core.llm import chat as llm_chat
from ..pipeline_core.utils import try_parse_json
from braintrust import current_span, traced
from ._quality_prompt import quality_prompt


class QualityJudge:
    """LLM-as-Judge for quality dimension scoring"""

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

    def __init__(self, input_data: List[Dict[str, Any]]):
        self.phase_name = "04_quality_evaluation"
        self.samples = [DIYRepairWithFailureLabels(**item) for item in input_data]
        print(f"📊 Loaded {len(self.samples)} samples from Phase 03")

    @traced(type="llm", name="quality_judge_evaluation", notrace_io=True)
    def evaluate(
        self,
        sample: DIYRepairWithFailureLabels,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[QualityJudgment], str]:
        """
        Evaluate one sample for quality dimensions
        Returns: (QualityJudgment, raw_response)
        """
        if params is None:
            params = {
                "max_tokens": 500,
                "temperature": 0.2,
            }

        sample_json = sample.model_dump_json(indent=2)
        prompt = quality_prompt.replace("{sample_json}", sample_json)

        try:
            raw_response = llm_chat(
                prompt=prompt,
                params=params,
                metadata={
                    "phase": self.phase_name,
                    "sample_id": sample.id,
                },
            )

            ok, data = try_parse_json(raw_response)
            if not ok:
                return None, raw_response

            quality_scores = QualityScores(**data["quality_scores"])
            judgment = QualityJudgment(
                quality_scores=quality_scores,
                quality_pass=data["quality_pass"],
                overall_score=data["overall_score"],
                reasoning=data["reasoning"],
            )

            self._log_braintrust(sample, judgment, raw_response)
            return judgment, raw_response

        except Exception as e:
            print(f"ERROR - [{self.phase_name}] Error evaluating {sample.id}: {e}")
            return None, ""

    def _log_braintrust(self, sample, judgment, raw_response):
        """Log evaluation to Braintrust"""
        current_span().log(
            input={"sample_id": sample.id, "question": sample.question},
            output=judgment.model_dump() if judgment else None,
            metadata={
                "phase": self.phase_name,
                "sample_id": sample.id,
                "quality_pass": judgment.quality_pass if judgment else False,
                "overall_score": judgment.overall_score if judgment else 0.0,
            },
        )

    def evaluate_dataset(
        self,
    ) -> Tuple[List[DIYRepairWithQualityScores], Dict[str, Any]]:
        """Evaluate all samples and return labeled dataset + statistics"""
        labeled_samples = []
        quality_stats = {
            "total_samples": len(self.samples),
            "evaluated": 0,
            "quality_pass_count": 0,
            "dimension_pass_counts": {dim: 0 for dim in self.QUALITY_DIMENSIONS},
            "average_score": 0.0,
        }

        print(f"\n🔍 Evaluating {len(self.samples)} samples for quality...")

        for i, sample in enumerate(self.samples, 1):
            print(f"  [{i}/{len(self.samples)}] Evaluating {sample.id}...", end=" ")

            judgment, _ = self.evaluate(sample)

            if judgment:
                labeled_sample = DIYRepairWithQualityScores(
                    **sample.model_dump(), quality_judgment=judgment
                )
                labeled_samples.append(labeled_sample)

                quality_stats["evaluated"] += 1
                if judgment.quality_pass:
                    quality_stats["quality_pass_count"] += 1

                for dim in self.QUALITY_DIMENSIONS:
                    if getattr(judgment.quality_scores, dim) == 1:
                        quality_stats["dimension_pass_counts"][dim] += 1

                print("✓")
            else:
                print("✗ (evaluation failed)")

        if quality_stats["evaluated"] > 0:
            quality_stats["average_score"] = (
                sum(s.quality_judgment.overall_score for s in labeled_samples)
                / quality_stats["evaluated"]
            )

        return labeled_samples, quality_stats

    def print_summary(self, stats: Dict[str, Any]):
        """Print quality evaluation summary"""
        print("\n" + "=" * 60)
        print("📊 QUALITY EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Successfully Evaluated: {stats['evaluated']}")
        print(
            f"Quality Pass Rate: {stats['quality_pass_count']}/{stats['evaluated']} "
            f"({stats['quality_pass_count']/stats['evaluated']*100:.1f}%)"
        )
        print(f"Average Quality Score: {stats['average_score']:.3f}")
        print("\nDimension Pass Rates:")
        for dim, count in stats["dimension_pass_counts"].items():
            rate = count / stats["evaluated"] * 100 if stats["evaluated"] > 0 else 0
            print(f"  {dim}: {count}/{stats['evaluated']} ({rate:.1f}%)")
        print("=" * 60)
