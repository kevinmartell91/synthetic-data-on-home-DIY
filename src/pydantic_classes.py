"""
Defines the OutputStructure pydantic class
"""

from typing import Literal, List
from pydantic import BaseModel, Field


class Metadata(BaseModel):
    issue_type: Literal[
        "appliance_repair",
        "plumbing_repair",
        "electrical_repair",
        "hvac_repair",
        "general_home_repair",
    ] = Field("This is the ussue type")
    response_type: Literal[
        "appliance_repair",
        "plumbing_repair",
        "electrical_repair",
        "hvac_repair",
        "general_home_repair",
    ] = Field("tThis is the response type")


class OutputStructure(BaseModel):

    id: int
    question: str = Field(description="The user's query")
    answer: str = Field(description="Sythetic answer generated")
    equipment_problem: str = Field(
        description="Equipment required to solve the problem"
    )
    tools_required: str = Field(description="Tools required to solve the problem")
    step: str = Field(description="Step by step instructions")
    safety_info: str = Field(description=" Safety information is available")
    tips: str = Field(description="Tips to solve the problem")
    metadata: Metadata = Field(description="Metadata: issue type and response types")


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
