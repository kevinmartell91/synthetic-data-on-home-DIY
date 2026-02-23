"""
Data Validation Module for Synthetic DIY Repair QA
Validates structural correctness and filters invalid entries
"""

import json
from typing import List, Dict, Any, Tuple
from pydantic import ValidationError
from pydantic_classes import OutputStructure


class DataValidator:
    """Validates synthetic dataset using Pydantic schema"""
    
    def __init__(self):
        self.validation_report = {
            "total_samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "validation_errors": []
        }
    
    def validate_dataset(self, dataset: List[Dict[str, Any]]) -> Tuple[List[OutputStructure], Dict[str, Any]]:
        """
        Validates entire dataset and returns valid samples + report
        
        Returns:
            (valid_samples, validation_report)
        """
        valid_samples = []
        self.validation_report["total_samples"] = len(dataset)
        
        for idx, sample in enumerate(dataset):
            try:
                # Attempt Pydantic validation
                validated_sample = OutputStructure(**sample)
                valid_samples.append(validated_sample)
                self.validation_report["valid_samples"] += 1
                
            except ValidationError as e:
                self.validation_report["invalid_samples"] += 1
                self.validation_report["validation_errors"].append({
                    "sample_id": sample.get("id", idx),
                    "error_type": "ValidationError",
                    "errors": e.errors(),
                    "sample_preview": str(sample)[:200]
                })
                print(f"❌ Validation failed for sample {sample.get('id', idx)}: {e}")
                
            except Exception as e:
                self.validation_report["invalid_samples"] += 1
                self.validation_report["validation_errors"].append({
                    "sample_id": sample.get("id", idx),
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "sample_preview": str(sample)[:200]
                })
                print(f"❌ Unexpected error for sample {sample.get('id', idx)}: {e}")
        
        return valid_samples, self.validation_report
    
    def save_validation_report(self, filename: str = "validation_report.json"):
        """Save validation report to file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.validation_report, f, indent=2, ensure_ascii=False)
        print(f"📋 Validation report saved to: {filename}")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*60)
        print("📊 VALIDATION SUMMARY")
        print("="*60)
        print(f"Total samples: {self.validation_report['total_samples']}")
        print(f"✅ Valid samples: {self.validation_report['valid_samples']}")
        print(f"❌ Invalid samples: {self.validation_report['invalid_samples']}")
        
        if self.validation_report['invalid_samples'] > 0:
            print(f"\n⚠️  {self.validation_report['invalid_samples']} samples failed validation")
            print("See validation_report.json for details")
        else:
            print("\n🎉 All samples passed validation!")


def main():
    """Standalone validation script"""
    import os
    import sys
    
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pass_number = "03"
    folder_name = f"eval_pass_{pass_number}"
    dataset_file = os.path.join(script_dir, folder_name, "diy_synthetic_dataset.json")
    report_file = os.path.join(script_dir, folder_name, "validation_report.json")
    filtered_file = os.path.join(script_dir, folder_name, "validated_dataset.json")
    
    # Load dataset
    print(f"📂 Loading dataset from: {dataset_file}")
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_file}")
        sys.exit(1)
    
    # Validate
    validator = DataValidator()
    valid_samples, report = validator.validate_dataset(dataset)
    
    # Print summary
    validator.print_summary()
    
    # Save report
    validator.save_validation_report(report_file)
    
    # Save filtered dataset (valid samples only)
    if valid_samples:
        filtered_data = [s.model_dump() for s in valid_samples]
        with open(filtered_file, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        print(f"💾 Validated dataset saved to: {filtered_file}")
    
    print("\n✅ Validation complete!")


if __name__ == "__main__":
    main()