"""
Syntheic data Analyzer class
"""

from collections import defaultdict
import json, os, re
import statistics
from typing import Counter, Dict, Any
from failure_labeler import main as start_failure_labeler
from pydantic_classes import OutputStructure, MissingFieldsCounter, MissingFieldsMetadataCounter
from data_validator import DataValidator


class SyntheticDataAnalyzer:

    def __init__(
        self,
        pass_number="01",
        dataset_filename: str = "diy_synthetic_dataset.json",
    ) -> None:
        # Get the directory where this script is located
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.pass_number = f"eval_pass_{pass_number}"
        
        dataset_path = os.path.join(self.script_dir, self.pass_number, dataset_filename)
        dataset = self.load_dataset(dataset_path)
        self.dataset, _ = DataValidator().validate_dataset(dataset) if dataset else (None, None)

    def load_dataset(self, filename: str):
        """Load synthetic dataset"""
        try:
            with open(filename, mode="r", encoding="utf-8") as f:
                dataset = json.load(f)
                return dataset

        except FileNotFoundError:
            print(f"❌ Loading dataset {filename} - ERROR: File not found")
            return None
        except json.JSONDecodeError as e:
            print(f"❌ Loading dataset {filename} - ERROR: Invalid JSON format - {e}")
            return None
        except Exception as e:
            print(f"❌ Loading dataset {filename} - ERROR: {type(e).__name__}: {e}")
            return None

    def prepare_failure_labeling_dataset(self):
        """Prepare dataset for failure labeling"""
        start_failure_labeler()


    def evaluate_data_quality(self) -> Dict[str, Any]:
        """1️⃣ Data Quality Evaluation"""
        print("🔍 Evaluating Data Quality...")

        # Check for duplicates
        question = [sample.question for sample in self.dataset]

        duplicate_question = len(question) - len(set(question))
        duplicate_rate = (duplicate_question) / (1 * len(self.dataset)) * 100

        # track missing fields
        details = MissingFieldsCounter(metadata=MissingFieldsMetadataCounter())

        # Check for missing/empty fields
        missing_fields = 0
        for sample in self.dataset:
            if not sample.question.strip():
                missing_fields += 1
                details.question += 1
            if not sample.answer.strip():
                details.answer += 1
                missing_fields += 1
            if not sample.equipment_problem.strip():
                missing_fields += 1
                details.equipment_problem += 1
            if not sample.tools_required.strip():
                missing_fields += 1
                details.tools_required += 1
            if not sample.step.strip():
                missing_fields += 1
                details.step += 1
            if not sample.safety_info.strip():
                missing_fields += 1
                details.safety_info += 1
            if not sample.tips.strip():
                missing_fields += 1
                details.tips += 1
            if not sample.metadata.issue_type.strip():
                missing_fields += 1
                details.metadata.issue_type += 1
            if not sample.metadata.response_type.strip():
                missing_fields += 1
                details.metadata.response_type += 1

        # Data completeness
        completeness_rate = (1 - missing_fields / (len(self.dataset) * 9)) * 100

        # Quality score calculation
        quality_score = max(0, 100 - duplicate_rate - (100 - completeness_rate))

        return {
            "total_records": len(self.dataset),
            "duplicate_rate": duplicate_rate,
            "completeness_rate": completeness_rate,
            "missing_fields": missing_fields,
            "missing_fields_details": details.model_dump(),
            "quality_score": quality_score,
        }

    def evaluate_diversity(self) -> Dict[str, Any]:
        """2️⃣ Diversity Analysis"""
        print("🌈 Evaluating Data Diversity...")

        # Vocabulary analysis
        all_text = " ".join(
            [
                sample.question
                + " "
                + sample.answer
                + " "
                + sample.equipment_problem
                + " "
                + sample.tools_required
                + " "
                + sample.step
                + " "
                + sample.safety_info
                + " "
                + sample.tips
                for sample in self.dataset
            ]
        )

        # Regex pattern
        """
        So this pattern matches:
            Whole words like "hello", "world", "42", "data_science"
            But not punctuation or whitespace
        """
        words = re.findall(r'\b\w+\b', all_text.lower())
        unique_words = set(words)
        vocabulary_richness = len(unique_words) / len(words) if words else 0

        # Category distribution
        issue_counts = Counter([sample.metadata.issue_type for sample in self.dataset])
        response_counts = Counter([sample.metadata.response_type for sample in self.dataset])

        # Label balance (for issue types)
        max_count = max(issue_counts.values()) if issue_counts else 1
        min_count = min(issue_counts.values()) if issue_counts else 1
        label_balance_ratio = min_count / max_count if max_count > 0 else 0

        # Diversity score
        # the higer the entropy, the higer the diversity and viceversa
        issue_entropy = self._calculate_entropy(issue_counts)
        response_entropy = self._calculate_entropy(response_counts)

        diversity_score = (
            vocabulary_richness * 100
            + issue_entropy * 10
            + response_entropy * 10
            + label_balance_ratio * 30
        )

        return {
            "vocabulary_richness": vocabulary_richness,
            "unique_words": len(unique_words),
            "total_words": len(words),
            "label_balance_ratio": label_balance_ratio,
            "issue_type_entropy": issue_entropy,
            "response_type_entropy": response_entropy,
            "issue_distribution": dict(issue_counts),
            "response_distribution": dict(response_counts),
            "diversity_score": min(100, diversity_score),
        }

    def evaluate_linguistics_quality(self) -> Dict[str, Any]:
        print(" Evaluating liguistic Quality")

        all_text = []
        for sample in self.dataset:
            all_text.append(sample.question)
            all_text.append(sample.answer)
            all_text.append(sample.equipment_problem)
            all_text.append(sample.tools_required)
            all_text.append(sample.step)
            all_text.append(sample.safety_info)
            all_text.append(sample.tips)

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

        return {
            "average_coherence": avg_coherence,
            "repetition_rate": repetition_rate,
            "exact_text_repetitions": exact_repetitions,
            "informal_words_per_text": avg_informal,
            "contractions_per_text": avg_contraction,
            "linguistic_quality_score": linguistic_score,
        }

    def _calculate_entropy(self, counter: Counter):
        """Calculates Shannin entropy for diversity measurement"""
        import math

        total = sum(counter.values())
        entropy = 0
        for count in counter.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy

    def analyse_coverage_gap(self) -> Dict[str, Any]:
        """Identify potential gaps in scenario cover"""

        # Analyze issue type coverage
        issue_types = [sample.issue_type for sample in self.dataset]
        response_type = [sample.response_type for sample in self.dataset]

        # check for combinations
        # `defaultdict(list)`
        # this creates dictionary where each key is unique.
        # It does not require to check whether the key exits
        # when apending new elements
        issue_response_combination = defaultdict(list)

        for sample in self.dataset:
            key = f"{sample.issue_type}_{sample.response_type}"
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
        for sample in self.dataset:
            query = sample.question.lower()
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

            return {
                "combination_analysis": {
                    "total_combinations": len(combination_counts),
                    "avg_samples_per_combination": avg_combination_count,
                    "underrepresented_combinations": underrepresented,
                    "well_represented_combinations": {
                        k: v
                        for k, v in combination_counts.items()
                        if v >= avg_combination_count
                    },
                },
                "query_pattern_analysis": {
                    "pattern_distribution": dict(Counter(query_patterns)),
                    "pattern_diversity": len(set(query_patterns)),
                },
                "recommendations": self._generate_coverage_recommendations(
                    underrepresented, query_patterns
                ),
            }

    def _generate_coverage_recommendations():
        pass

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """🚀 Run comprehensive synthetic data evaluation with detailed metrics"""
        print("🚀 Starting Comprehensive Synthetic Data Evaluation...")
        print("=" * 60)

        # Run all evaluation components
        data_quality = self.evaluate_data_quality()
        diversity_analysis = self.evaluate_diversity()

        # Calculate overall scores
        overall_score = (
            data_quality["quality_score"] * 0.5
            + diversity_analysis["diversity_score"] * 0.5
        )

        return {
            "data_quality": data_quality,
            "diversity_analysis": diversity_analysis,
            "overall_scores": {
                "data_quality_score": data_quality["quality_score"],
                "diversity_score": diversity_analysis["diversity_score"],
                "overall_score": overall_score,
            },
        }


def main():
    analizer = SyntheticDataAnalyzer(pass_number="03")
        
    # Auto label failure modes as .csv
    analizer.prepare_failure_labeling_dataset()

    # Run evaluation
    results = analizer.run_comprehensive_evaluation()
    print("results", json.dumps(results, indent=2))

    # Save detailed results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(script_dir, analizer.pass_number, "synthetic_data_evaluation_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Results saved to: {path}")

    # Final assessment
    overall_score = results["overall_scores"]["overall_score"]
    print(f"\n✅ Evaluation Complete!")

    if overall_score >= 80:
        print("🎉 Excellent data quality! Ready for production use.")
    elif overall_score >= 60:
        print("⚠️  Good data quality with room for improvement.")
    else:
        print("❌ Data quality needs significant improvement before use.")


if __name__ == "__main__":
    main()
