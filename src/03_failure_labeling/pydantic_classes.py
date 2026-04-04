from typing import List, Optional
from pydantic import BaseModel, Field

from pydantic_classes import DIYRepairSyntheticItem


class FailureJudgment(BaseModel):
    """Binary failure mode detection (Phase 3)"""

    trace_id: str
    incomplete_answer: int = None
    safety_violations: int = None
    unrealistic_tools: int = None
    overcomplicated_solution: int = None
    missing_context: int = None
    poor_quality_tips: int = None
    overall_failure: bool = None
    reasoning: str


class DIYRepairWithFailureLabels(DIYRepairSyntheticItem):
    """Extends synthetic item with failure labels"""

    failure_judgment: FailureJudgment
