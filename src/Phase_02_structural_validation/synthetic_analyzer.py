"""
Syntheic data Analyzer class
"""

from collections import defaultdict
import json, os, re
import statistics
from typing import Counter, Dict, Any, List, Union
from failure_labeler import main as start_failure_labeler
from .pydantic_classes import (
    CombinationAnalysis,
    DataQualityResults,
    MissingFieldsCounter,
    MissingFieldsMetadataCounter,
    DiversityResult,
    LinguisticResults,
    CoverageAnalysisResult,
    OverallScores,
    QueryPatternAnalysis,
    CoverageSummary,
)
from pydantic_classes import (
    DIYRepairSyntheticItem,
    DIYRepairSyntheticMalformedItem,
    Metadata,
)


class SyntheticAnalyzer:

    def __init__(self, input_data: List[Dict[str, Any]]):
        self.phase_name = "02_structural_validation"
        self.input_data: List[
            Union[DIYRepairSyntheticItem, DIYRepairSyntheticMalformedItem]
        ] = list(
            map(
                lambda i: (
                    DIYRepairSyntheticItem(**i)
                    if i.get("error") is None
                    else DIYRepairSyntheticMalformedItem(**i)
                ),
                input_data,
            )
        )

    def prepare_failure_labeling_dataset(self):
        """Prepare dataset for failure labeling"""
        start_failure_labeler()

    def evaluate_data_quality(self) -> DataQualityResults:
        """1️ Data Quality Evaluation"""
        print(" Evaluating Data Quality...", type(self.input_data[0]))

        # Check for duplicates
        questions = [sample.question for sample in self.input_data if sample.question]

        duplicate_question = len(questions) - len(set(questions))
        duplicate_rate = (duplicate_question) / (1 * len(self.input_data)) * 100

        # track missing fields
        details = MissingFieldsCounter(metadata=MissingFieldsMetadataCounter())

        # Check for missing/empty fields
        missing_fields = 0
        metadata = None
        for sample in self.input_data:
            if not sample.question:
                missing_fields += 1
                details.question += 1
            if not sample.answer:
                details.answer += 1
                missing_fields += 1
            if not sample.equipment_problem:
                missing_fields += 1
                details.equipment_problem += 1
            if not sample.tools_required:
                missing_fields += 1
                details.tools_required += 1
            if not sample.steps:
                missing_fields += 1
                details.step += 1
            if not sample.safety_info:
                missing_fields += 1
                details.safety_info += 1
            if not sample.tips:
                missing_fields += 1
                details.tips += 1
            if not sample.metadata:
                missing_fields += 1
                details.metadata.issue_type += 1
                details.metadata.user_query += 1

        # Data completeness
        completeness_rate = (1 - missing_fields / (len(self.input_data) * 9)) * 100

        # Quality score calculation
        quality_score = max(0, 100 - duplicate_rate - (100 - completeness_rate))

        return DataQualityResults(
            total_records=len(self.input_data),
            duplicate_rate=duplicate_rate,
            completeness_rate=completeness_rate,
            missing_fields=missing_fields,
            missing_fields_details=details.model_dump(),
            quality_score=quality_score,
        )

    def evaluate_diversity(self) -> DiversityResult:
        """2️ Diversity Analysis"""
        print(" Evaluating Data Diversity...")

        # Vocabulary analysis
        all_text = " ".join(
            [
                (item.question if item.question else "")
                + " "
                + (item.answer if item.answer else "")
                + " "
                + (item.equipment_problem if item.equipment_problem else "")
                + " "
                + ((" ").join(item.tools_required) if item.tools_required else "")
                + " "
                + ((" ").join(item.steps) if item.steps else "")
                + " "
                + (item.safety_info if item.safety_info else "")
                + " "
                + (item.tips if item.tips else "")
                for item in self.input_data
            ]
        )

        # Regex pattern
        """
        So this pattern matches:
            Whole words like "hello", "world", "42", "data_science"
            But not punctuation or whitespace
        """
        words = re.findall(r"\b\w+\b", all_text.lower())
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / len(words) if words else 0

        # Category distribution
        issue_counts = Counter(
            [
                sample.metadata.issue_type
                for sample in self.input_data
                if sample.metadata
            ]
        )
        # response_counts = Counter(
        #     [sample.metadata.response_type for sample in self.input_data]
        # )

        # Label balance (for issue types)
        max_count = max(issue_counts.values()) if issue_counts else 1
        min_count = min(issue_counts.values()) if issue_counts else 1
        label_balance_ratio = min_count / max_count if max_count > 0 else 0

        # Diversity score
        # the higer the entropy, the higer the diversity and viceversa
        issue_entropy = self._calculate_entropy(issue_counts)
        # response_entropy = self._calculate_entropy(response_counts)

        diversity_score = (
            vocabulary_richness * 100
            + issue_entropy * 10
            # + response_entropy * 10
            + label_balance_ratio * 30
        )

        return DiversityResult(
            vocabulary_richness=vocabulary_richness,
            unique_words=len(unique_words),
            total_words=len(words),
            label_balance_ratio=label_balance_ratio,
            issue_type_entropy=issue_entropy,
            # response_type_entropy=response_entropy,
            issue_distribution=dict(issue_counts),
            # response_distribution=dict(response_counts),
            diversity_score=min(100, diversity_score),
        )

    def evaluate_linguistics_quality(self) -> LinguisticResults:
        print(" Evaluating liguistic Quality")

        all_text = []
        for sample in self.input_data:
            all_text.append(sample.question if sample.question else "")
            all_text.append(sample.answer if sample.question else "")
            all_text.append(sample.equipment_problem if sample.question else "")
            all_text.append(" ".join(sample.tools_required) if sample.question else "")
            all_text.append(" ".join(sample.steps) if sample.question else "")
            all_text.append(sample.safety_info if sample.question else "")
            all_text.append(sample.tips if sample.question else "")

        # Text repetition analysis
        text_counts = Counter(all_text)
        exact_repetitions = [count - 1 for count in text_counts.values() if count > 1]
        repetition_rate = (
            (exact_repetitions / len(text_counts)) * 100 if all_text else 0
        )

        # Coherence analysis with simple heuristic based on sentence structure
        coherence_scores = []
        for text in all_text:
            sentence = re.split(
                r"[.!?]+"  # match one or more sentence that ends in '.', '!', '?',
            )
            sentence = [s.strip() for s in sentence if s.strip()]

            if len(sentence) > 0:
                # Simple coherence: ratio of connencting words to total words
                connector_words = [
                    "and",
                    "but",
                    "however",
                    "therefore",
                    "because",
                    "since",
                    "although",
                ]
                words = text.lower().split()
                counting_words = sum([1 for word in words if word in connector_words])

                coherence = counting_words / len(words) if words else 0
                coherence_scores.append(coherence)

        avg_coherence = statistics.mean(coherence_scores) if coherence_scores else 0

        # Informal language detection
        informal_words = [
            "gonna",
            "wanna",
            "yeah",
            "ok",
            "okay",
            "hey",
            "hi",
            "thanks",
            "thx",
        ]
        contractions = ["'re", "'ve", "'ll", "'d", "n't", "'s", "'m"]

        informal_counts = []
        contractions_counts = []

        for text in all_text:
            words = text.lower().split()
            informal_count = sum(
                [1 for word in words if any(inf in word for inf in informal_words)]
            )
            contraction_count = sum(
                [1 for word in words if any(cont in word for cont in contractions)]
            )
            informal_counts.append(informal_count)
            contractions_counts.append(contraction_count)

        avg_informal = statistics.mean(informal_counts) if informal_counts else 0
        avg_contraction = (
            statistics.mean(contractions_counts) if contractions_counts else 0
        )

        # Liguistic score
        repetition_penalty = min(50, repetition_rate * 2)
        coherence_bonus = avg_coherence * 100
        naturalness_bonus = (avg_informal + avg_contraction) * 10

        linguistic_score = max(
            0, 100 - repetition_penalty + coherence_bonus + naturalness_bonus
        )
        linguistic_score = min(100, linguistic_score)

        return LinguisticResults(
            average_coherence=avg_coherence,
            repetition_rate=repetition_rate,
            exact_text_repetitions=exact_repetitions,
            informal_words_per_text=avg_informal,
            contractions_per_text=avg_contraction,
            linguistic_quality_score=linguistic_score,
        )

    def _calculate_entropy(self, counter: Counter) -> float:
        """Calculates Shannin entropy for diversity measurement"""
        import math

        total = sum(counter.values())
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def analyse_coverage_gap(self) -> CoverageAnalysisResult:
        """Identify potential gaps in scenario cover"""

        # Analyze issue type coverage
        issue_types = [sample.metadata.issue_type for sample in self.input_data]
        # response_type = [sample.metadata.user_query for sample in self.input_data]

        # check for combinations
        # `defaultdict(list)`
        # this creates dictionary where each key is unique.
        # It does not require to check whether the key exits
        # when apending new elements
        issue_response_combination = defaultdict(list)

        for sample in self.input_data:
            # key = f"{sample.metadata.issue_type}_{sample.metadata.user_query}"
            key = f"{sample.metadata.issue_type}"
            issue_response_combination[key].append(sample.id)

        # Check for unrepresented combinations
        underrepresented_threshold = 0.5
        combination_counts = {k: len(v) for k, v in issue_response_combination}
        avg_combination_count = statistics.mean(combination_counts.values())

        underrepresented = {
            k: v
            for k, v in combination_counts.items()
            if v < avg_combination_count * underrepresented_threshold
        }

        # Analyse patterns
        query_patterns = []
        for sample in self.input_data:
            query = sample.question.lower() if sample.question else ""
            # Identify common patterns
            if "cant't" in query or "cannot" in query:
                query_patterns.append("inability_statement")
            elif "error" in query:
                query_patterns.append("error_sample")
            elif "?" in query:
                query_patterns.append("question_format")
            elif "how" in query:
                query_patterns.append("how_to_question")
            elif "where" in query:
                query_patterns.append("where_question")
            else:
                query_patterns.append("question_format")

            return CoverageAnalysisResult(
                combination_analysis=CombinationAnalysis(
                    total_combinations=len(combination_counts),
                    avg_samples_per_combination=avg_combination_count,
                    underrepresented_combinations=underrepresented,
                    well_represented_combinations={
                        k: v
                        for k, v in combination_counts.items()
                        if v >= avg_combination_count
                    },
                ),
                query_pattern_analysis=QueryPatternAnalysis(
                    pattern_distribution=dict(Counter(query_patterns)),
                    pattern_diversity=len(set(query_patterns)),
                ),
                recommendations=self._generate_coverage_recommendations(
                    underrepresented, query_patterns
                ),
            )

    def _generate_coverage_recommendations():
        pass

    def run_comprehensive_evaluation(self) -> CoverageSummary:
        """Run comprehensive synthetic data evaluation with detailed metrics"""
        print(" Starting Comprehensive Synthetic Data Evaluation...")
        print("=" * 60)

        # Run all evaluation components
        data_quality = self.evaluate_data_quality()
        diversity_analysis = self.evaluate_diversity()

        # Calculate overall scores
        overall_score = (
            data_quality.quality_score * 0.5 + diversity_analysis.diversity_score * 0.5
        )

        return CoverageSummary(
            data_quality=data_quality,
            diversity_analysis=diversity_analysis,
            overall_scores=OverallScores(
                data_quality_score=data_quality.quality_score,
                diversity_score=diversity_analysis.diversity_score,
                overall_score=overall_score,
            ),
        )

    def print_summary(self, results: CoverageSummary):
        # Final assessment
        overall_score = results.overall_scores.overall_score
        print(f"\n Evaluation Complete!")

        if overall_score >= 80:
            print(" Excellent data quality! Ready for production use.")
        elif overall_score >= 60:
            print("  Good data quality with room for improvement.")
        else:
            print(" Data quality needs significant improvement before use.")
