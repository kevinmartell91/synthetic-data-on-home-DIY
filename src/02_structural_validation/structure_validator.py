from typing import Any, TypeVar, Type, Optional, Tuple, List
from .pydantic_classes import ValidationResult
from pipeline_core.io import save_report
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructureValidator:

    def __init__(self, input_data: any):
        self.input_data = input_data
        self.results = None

    def validate(self, schema: Optional[Type[T]]) -> Tuple[List[T], ValidationResult]:
        print("Validating dataset...")

        valid_samples = []
        if self.results is None:
            self.results = ValidationResult()

        for sample in self.input_data:
            try:
                ok = schema.model_validate(sample)
                valid_samples.append(ok)
                self.results.valid_samples += 1

            except ValidationError as e:
                self.results.invalid_samples += 1
                # append the malformed sample (computed in stage 01 - generation)
                self.results.validation_errors.append({"sample": sample})
        self.results.total_samples = (
            self.results.valid_samples + self.results.invalid_samples
        )
        print(f"Validated {self.results.total_samples} samples")

        return valid_samples, self.results

    def print_summary(self):
        """Print structure validation summary"""
        print("\n" + "=" * 60)
        print("📊 VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total samples: {self.results.total_samples}")
        print(f"✅ Valid samples: {self.results.valid_samples}")
        print(f"❌ Invalid samples: {self.results.invalid_samples}")

        if self.results.invalid_samples > 0:
            print(f"\n⚠️  {self.results.invalid_samples} samples failed validation")
            print("See validation_report.json for details")
        else:
            print("\n🎉 All samples passed validation!")
