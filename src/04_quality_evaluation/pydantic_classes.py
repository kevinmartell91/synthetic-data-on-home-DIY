import importlib
from pydantic import BaseModel

# Phase 3 package is `03_failure_labeling` (digit prefix → not a valid `from x import` path).
# Module name also depends on cwd / -m style:
#   repo root: `python -m src.04_quality_evaluation.run` → `src.03_failure_labeling...`
#   under src/: `python -m 04_quality_evaluation.run` → `03_failure_labeling...`
try:
    _phase3_mod = importlib.import_module("src.03_failure_labeling.pydantic_classes")
except ModuleNotFoundError:
    _phase3_mod = importlib.import_module("03_failure_labeling.pydantic_classes")

DIYRepairWithFailureLabels = _phase3_mod.DIYRepairWithFailureLabels


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
