import importlib
from pydantic import BaseModel
from Phase_03_failure_labeling.pydantic_classes import DIYRepairWithFailureLabels


class QualityScores(BaseModel):
    """8 quality dimensions (binary: 1 = pass, 0 = fail)"""

    answer_coherence: int
    step_actionability: int
    tool_realism: int
    safety_specificity: int
    tip_usefulness: int
    problem_answer_alignment: int
    appropriate_scope: int
    category_accuracy: int


class QualityJudgment(BaseModel):
    """Quality evaluation result"""

    quality_scores: QualityScores
    quality_pass: bool
    overall_score: float  # 0.0 to 1.0
    reasoning: str


class DIYRepairWithQualityScores(DIYRepairWithFailureLabels):
    """Extends failure-labeled item with quality scores"""

    quality_judgment: QualityJudgment
