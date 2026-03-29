from email.policy import default
from typing import List, Any, Dict
from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    total_samples: int = Field(
        description="The total number of valid and invalid samples", default=0
    )
    valid_samples: int = Field(
        description="The total number of valid samples", default=0
    )
    invalid_samples: int = Field(
        description="The total number of invalid samples", default=0
    )
    validation_errors: List = Field(
        description="A list of errors from invalid samples", default=[]
    )


# Missing fields counter for Data Quality
# ===================================================
# Missing fields metadata details counter
class MissingFieldsMetadataCounter(BaseModel):
    # add a default value of 0 for all fields
    issue_type: int = 0
    user_query: int = 0


class MissingFieldsCounter(BaseModel):
    question: int = 0
    answer: int = 0
    equipment_problem: int = 0
    tools_required: int = 0
    step: int = 0
    safety_info: int = 0
    tips: int = 0
    metadata: MissingFieldsMetadataCounter


class DataQualityResults(BaseModel):
    total_records: int = Field(..., description="Number of total records")
    duplicate_rate: float = Field(..., description="duplicated rate")
    completeness_rate: float = Field(..., description="completeness rate")
    missing_fields: int = Field(..., description="Number of missing_fields")
    missing_fields_details: Any = Field(..., description="missing fields details")
    quality_score: float = Field(..., description="quality score")


class DiversityResult(BaseModel):
    vocabulary_richness: float = Field(..., description="vocabulary richness")
    unique_words: int = Field(..., description="unique words")
    total_words: int = Field(..., description="total words")
    label_balance_ratio: float = Field(..., description="label balance ratio")
    issue_type_entropy: float = Field(..., description="issue type entropy")
    # response_type_entropy: float = Field(..., description="response type entropy")
    issue_distribution: Any = Field(..., description="issue distribution")
    # response_distribution: Any = Field(..., description="response distribution")
    diversity_score: float = Field(..., description="diversity score")


class LinguisticResults(BaseModel):
    average_coherence: Any = Field(..., description="average coherence")
    repetition_rate: Any = Field(..., description="repetition rate")
    exact_text_repetitions: Any = Field(..., description="exact text repetitions")
    informal_words_per_text: Any = Field(..., description="informal word pertext")
    contractions_per_text: Any = Field(..., description="contractions per text")
    linguistic_quality_score: Any = Field(..., description="linguistic quality score")


class CombinationAnalysis(BaseModel):
    total_combinations: int = Field(
        ..., description="Number of unique category+pattern combinations observed"
    )
    avg_samples_per_combination: float = Field(
        ..., description="Mean number of samples per combination"
    )
    underrepresented_combinations: Dict[str, int] = Field(
        ..., description="Combinations with fewer samples than the average"
    )
    well_represented_combinations: Dict[str, int] = Field(
        ..., description="Combinations with sample counts >= average"
    )


class QueryPatternAnalysis(BaseModel):
    pattern_distribution: Dict[str, int] = Field(
        ..., description="Frequency count of each query pattern"
    )
    pattern_diversity: int = Field(..., description="Number of unique query patterns")


class CoverageAnalysisResult(BaseModel):
    combination_analysis: CombinationAnalysis
    query_pattern_analysis: QueryPatternAnalysis
    recommendations: List[str] = Field(
        ..., description="List of recommended actions to improve dataset coverage"
    )


class DataQuality(BaseModel):
    quality_score: float = Field(..., description="Overall data quality score")
    # Add any additional fields your `data_quality` dict contains
    # e.g., missing_rate: float, consistency_score: float, etc.


class DiversityAnalysis(BaseModel):
    diversity_score: float = Field(..., description="Overall diversity score")
    # Add additional fields from your `diversity_analysis` dict
    # e.g., category_distribution: Dict[str, int]


class OverallScores(BaseModel):
    data_quality_score: float = Field(
        ..., description="Score derived from data quality metrics"
    )
    diversity_score: float = Field(
        ..., description="Score derived from diversity analysis"
    )
    overall_score: float = Field(..., description="Weighted or combined overall score")


class CoverageSummary(BaseModel):
    data_quality: DataQualityResults
    diversity_analysis: DiversityResult
    overall_scores: OverallScores
