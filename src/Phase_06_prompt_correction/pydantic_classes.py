"""
Phase 06: Data Models

Extends the quality-scored item with correction metadata produced
by the prompt corrector analysis run.
"""

from typing import List, Optional
from pydantic import BaseModel

from ..Phase_04_quality_evaluation.pydantic_classes import DIYRepairWithQualityScores


class CorrectionMetadata(BaseModel):
    """Traceable record of corrections applied during this analysis run"""

    iteration_num: int
    corrections_applied: List[dict]   # [{change, reason, template_affected}]
    baseline_failure_rate: float
    categories_affected: List[str]
    corrected_prompts_path: str


class DIYRepairWithCorrectionMetadata(DIYRepairWithQualityScores):
    """Extends quality-scored item with Phase 06 correction metadata.

    correction_metadata is None for items whose category had no failures
    and therefore required no prompt corrections.
    """

    correction_metadata: Optional[CorrectionMetadata] = None
