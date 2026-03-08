"""
Defines the OutputStructure pydantic class
"""

from typing import Literal, List, Optional, Dict
from pydantic import BaseModel, Field


class OutputStructureBase(BaseModel):
    question: str = Field(
        ..., description="A realistic DIY repair question from a homeowner"
    )
    answer: str = Field(
        ...,
        description="Synthetic answer generated: A clear, actionable answer with step-by-step guidance",
    )
    equipment_problem: str = Field(
        description="The specific problem being addressed (e.g. 'dripping faucet')"
    )
    tools_required: List[str] = Field(
        ...,
        min_items=1,
        description="Tools a typical homeowner would realistically own",
    )
    steps: List[str] = Field(
        ..., min_items=3, description="Ordered, numbered repair steps"
    )
    safety_info: str = Field(
        ..., description="Relevant safety warnings and precautions"
    )
    tips: str = Field(
        description="Practical tips to make the repair easier or more reliable"
    )


class DIYRepairSyntheticItem(OutputStructureBase):
    id: str = Field(
        ..., description="Unique ID for the item", examples=["qa_001", "qa_002"]
    )


class ItemProcessingMetadata(BaseModel):
    prompt_template: str = Field(..., description="Which prompt template was used")
    is_valid_schema: bool = Field(
        ..., descripti="Whether the item passed structural validation"
    )
    failure_mode: str = Field(
        decription="Which failure modes were flagged by the judge"
    )
    timestamp: str = Field(description="Timestamp")
    model: str = Field(description="Model used")


class Metadata(BaseModel):
    issue_type: Literal[
        "appliance_repair",
        "plumbing_repair",
        "electrical_repair",
        "hvac_repair",
        "general_home_repair",
    ] = Field(description="The issue type")
    response_type: Literal[
        "complete_solution",
        "incomplete_answer",
        "safety_violations",
        "unrealistic_tools",
        "overcomplicated_solution",
        "missing_context",
        "poor_quality_tips",
    ] = Field(description="The response type")
    # NEW FIELDS for better evaluation
    difficulty_level: Literal["beginner", "intermediate", "advanced"] = Field(
        default="beginner", description="Expected difficulty level"
    )
    requires_professional: bool = Field(
        default=False, description="Whether professional help is required"
    )
    estimated_time_minutes: int = Field(
        default=30, description="Estimated time in minutes"
    )
    estimated_cost_usd: int = Field(default=50, description="Estimated cost in USD")


class CriteriaScores(BaseModel):
    """Structured scores for evaluation criteria (0-10 scale)"""

    safety: int = Field(..., ge=0, le=10, description="Safety score")
    practicality: int = Field(..., ge=0, le=10, description="Practicality score")
    completeness: int = Field(..., ge=0, le=10, description="Completeness score")
    clarity: int = Field(..., ge=0, le=10, description="Clarity score")
    contextual_fit: int = Field(..., ge=0, le=10, description="Contextual fit score")


class JudgeEvaluation(BaseModel):
    """Judge's evaluation of a synthetic sample"""

    judgment: Literal[
        "EXCELLENT", "ACCEPTABLE", "NEEDS_IMPROVEMENT", "UNSAFE", "IMPRACTICAL"
    ] = Field(description="Overall judgment category")
    reasoning: str = Field(description="Detailed explanation of the judgment")

    criteria_scores: CriteriaScores = Field(
        description="Scores 0-10 for each evaluation criterion"
    )

    failure_modes_detected: List[str] = Field(
        default_factory=list,
        description="List of detected failure modes from the 7 categories",
    )

    specific_issues: List[str] = Field(
        default_factory=list, description="Specific problems identified"
    )

    strengths: List[str] = Field(
        default_factory=list, description="Positive aspects of the response"
    )


class OutputStructure(BaseModel):
    id: str
    # NEW: Add evaluation field (populated by judge)
    evaluation: Optional[JudgeEvaluation] = Field(
        default=None, description="Judge's evaluation of this sample"
    )


class OverallAssessment(BaseModel):
    realism_score: int = Field(..., ge=1, le=10)
    diversity_score: int = Field(..., ge=1, le=10)
    authenticity_score: int = Field(..., ge=1, le=10)
    response_quality_score: int = Field(..., ge=1, le=10)
    language_naturalness_score: int = Field(..., ge=1, le=10)


class DetailedAnalysis(BaseModel):
    strengths: List[str]
    weaknesses: List[str]
    synthetic_patterns: List[str]
    improvement_suggestions: List[str]


class SampleEvaluation(BaseModel):
    sample_id: int
    realism_rating: Literal["high", "medium", "low"]
    notes: str


class EvaluationReport(BaseModel):
    overall_assessment: OverallAssessment
    detailed_analysis: DetailedAnalysis
    sample_evaluations: List[SampleEvaluation]


# Missing fields counter for Data Quality
# ===================================================
# Missing fields metadata details counter
class MissingFieldsMetadataCounter(BaseModel):
    # add a default value of 0 for all fields
    issue_type: int = 0
    response_type: int = 0


class MissingFieldsCounter(BaseModel):
    question: int = 0
    answer: int = 0
    equipment_problem: int = 0
    tools_required: int = 0
    step: int = 0
    safety_info: int = 0
    tips: int = 0
    metadata: MissingFieldsMetadataCounter
