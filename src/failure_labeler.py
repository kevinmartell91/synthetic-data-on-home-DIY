"""
Failure Labeling Module for Synthetic DIY Repair QA
Creates Pandas DataFrame with binary failure mode columns
"""

import json
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from pydantic_classes import OutputStructure


class FailureLabeler:
    """Creates and manages failure mode labels for synthetic dataset"""
    
    FAILURE_MODES = [
        "incomplete_answer",
        "safety_violations",
        "unrealistic_tools",
        "overcomplicated_solution",
        "missing_context",
        "poor_quality_tips"
    ]
    
    def __init__(self, dataset: List[OutputStructure]):
        self.dataset = dataset
        self.df = None
    
    def create_dataframe(self) -> pd.DataFrame:
        """
        Creates Pandas DataFrame with:
        - Trace ID (sample.id)
        - All structured fields
        - Binary columns for each failure mode (0 = success, 1 = failure)
        """
        rows = []
        
        for sample in self.dataset:
            row = {
                "trace_id": sample.id,
                "question": sample.question,
                "answer": sample.answer,
                "equipment_problem": sample.equipment_problem,
                "tools_required": sample.tools_required,
                "step": sample.step,
                "safety_info": sample.safety_info,
                "tips": sample.tips,
                "issue_type": sample.metadata.issue_type,
                "response_type": sample.metadata.response_type,
            }
            
            # Initialize all failure modes to 0 (success)
            for mode in self.FAILURE_MODES:
                row[f"failure_{mode}"] = 0
            
            # If judge evaluation exists, populate from there
            if sample.evaluation:
                for mode in sample.evaluation.failure_modes_detected:
                    if mode in self.FAILURE_MODES:
                        row[f"failure_{mode}"] = 1
                
                # Add judge metadata
                row["judge_judgment"] = sample.evaluation.judgment
                row["judge_reasoning"] = sample.evaluation.reasoning
                row["safety_score"] = sample.evaluation.criteria_scores.safety
                row["practicality_score"] = sample.evaluation.criteria_scores.practicality
                row["completeness_score"] = sample.evaluation.criteria_scores.completeness
                row["clarity_score"] = sample.evaluation.criteria_scores.clarity
                row["contextual_fit_score"] = sample.evaluation.criteria_scores.contextual_fit
            
            rows.append(row)
        
        self.df = pd.DataFrame(rows)
        return self.df
    
    def export_for_manual_labeling(self, filename: str = "manual_labeling_template.csv"):
        """
        Exports CSV template for manual labeling
        Includes only essential columns for human review
        """
        if self.df is None:
            self.create_dataframe()
        
        # Select columns for manual review
        manual_columns = ["trace_id", "question", "answer", "issue_type", "response_type"]
        manual_columns.extend([f"failure_{mode}" for mode in self.FAILURE_MODES])
        
        manual_df = self.df[manual_columns].copy()
        manual_df.to_csv(filename, index=False)
        print(f"📝 Manual labeling template exported to: {filename}")
        print(f"   Instructions: Edit the 'failure_*' columns (0 or 1) and save")
        return manual_df
    
    def import_manual_labels(self, filename: str = "manual_labeling_template.csv"):
        """
        Imports manually labeled CSV and updates DataFrame
        """
        try:
            manual_df = pd.read_csv(filename)
            
            # Validate columns
            required_cols = ["trace_id"] + [f"failure_{mode}" for mode in self.FAILURE_MODES]
            missing_cols = [col for col in required_cols if col not in manual_df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing columns in manual labels: {missing_cols}")
            
            # Update failure mode columns
            for mode in self.FAILURE_MODES:
                col = f"failure_{mode}"
                self.df[col] = self.df["trace_id"].map(
                    manual_df.set_index("trace_id")[col]
                ).fillna(0).astype(int)
            
            print(f"✅ Manual labels imported from: {filename}")
            return self.df
            
        except FileNotFoundError:
            print(f"❌ Manual labels file not found: {filename}")
            return None
        except Exception as e:
            print(f"❌ Error importing manual labels: {e}")
            return None
    
    def auto_label_from_judge(self):
        """
        Automatically labels failure modes using judge evaluations
        Already done in create_dataframe(), but this makes it explicit
        """
        if self.df is None:
            self.create_dataframe()
        
        labeled_count = self.df["judge_judgment"].notna().sum()
        print(f"✅ Auto-labeled {labeled_count}/{len(self.df)} samples from judge evaluations")
        return self.df
    
    def get_failure_summary(self) -> Dict[str, Any]:
        """
        Returns summary statistics of failure modes
        """
        if self.df is None:
            self.create_dataframe()
        
        summary = {
            "total_samples": len(self.df),
            "failure_mode_counts": {},
            "failure_mode_percentages": {},
            "samples_with_failures": 0,
            "samples_without_failures": 0,
        }
        
        for mode in self.FAILURE_MODES:
            col = f"failure_{mode}"
            count = self.df[col].sum()
            summary["failure_mode_counts"][mode] = int(count)
            summary["failure_mode_percentages"][mode] = round(count / len(self.df) * 100, 2)
        
        # Count samples with at least one failure
        failure_cols = [f"failure_{mode}" for mode in self.FAILURE_MODES]
        summary["samples_with_failures"] = int((self.df[failure_cols].sum(axis=1) > 0).sum())
        summary["samples_without_failures"] = len(self.df) - summary["samples_with_failures"]
        
        return summary
    
    def save_labeled_dataset(self, filename: str = "labeled_dataset.csv"):
        """Save labeled dataset to CSV"""
        if self.df is None:
            self.create_dataframe()
        
        self.df.to_csv(filename, index=False)
        print(f"💾 Labeled dataset saved to: {filename}")
    
    def save_labeled_json(self, filename: str = "labeled_dataset.json"):
        """Save labeled dataset back to JSON format with failure labels"""
        if self.df is None:
            self.create_dataframe()
        
        labeled_samples = []
        for sample in self.dataset:
            sample_dict = sample.model_dump()
            
            # Add failure mode labels
            row = self.df[self.df["trace_id"] == sample.id].iloc[0]
            sample_dict["failure_labels"] = {
                mode: int(row[f"failure_{mode}"]) for mode in self.FAILURE_MODES
            }
            
            labeled_samples.append(sample_dict)
        
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(labeled_samples, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Labeled dataset (JSON) saved to: {filename}")


def main():
    """Standalone failure labeling script"""
    import sys
    
    # Configuration
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pass_number = "03"
    folder_name = f"eval_pass_{pass_number}"
    
    # Option 1: Load evaluated dataset (with judge evaluations)
    evaluated_file = os.path.join(script_dir, folder_name, "evaluated_dataset.json")
    
    # Option 2: Load original dataset (for manual labeling)
    original_file = os.path.join(script_dir, folder_name, "diy_synthetic_dataset.json")
    
    # Try evaluated first, fall back to original
    dataset_file = evaluated_file if os.path.exists(evaluated_file) else original_file
    
    print(f"📂 Loading dataset from: {dataset_file}")
    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            dataset_raw = json.load(f)
        dataset = [OutputStructure(**s) for s in dataset_raw]
        print(f"✅ Loaded {len(dataset)} samples")
    except FileNotFoundError:
        print(f"❌ Dataset file not found: {dataset_file}")
        sys.exit(1)
    
    # Initialize labeler
    labeler = FailureLabeler(dataset)
    
    # Create DataFrame
    print("\n📊 Creating failure mode DataFrame...")
    df = labeler.create_dataframe()
    print(f"✅ DataFrame created with {len(df)} rows and {len(df.columns)} columns")
    
    # Auto-label from judge (if evaluations exist)
    if "judge_judgment" in df.columns and df["judge_judgment"].notna().any():
        print("\n🤖 Auto-labeling from judge evaluations...")
        labeler.auto_label_from_judge()
    
    # Get summary
    print("\n📈 Failure Mode Summary:")
    summary = labeler.get_failure_summary()
    print(f"Total samples: {summary['total_samples']}")
    print(f"Samples with failures: {summary['samples_with_failures']}")
    print(f"Samples without failures: {summary['samples_without_failures']}")
    print("\nFailure Mode Counts:")
    for mode, count in summary['failure_mode_counts'].items():
        pct = summary['failure_mode_percentages'][mode]
        print(f"  {mode}: {count} ({pct}%)")
    
    # Save outputs
    output_csv = os.path.join(script_dir, folder_name, "labeled_dataset.csv")
    output_json = os.path.join(script_dir, folder_name, "labeled_dataset.json")
    manual_template = os.path.join(script_dir, folder_name, "manual_labeling_template.csv")
    
    labeler.save_labeled_dataset(output_csv)
    labeler.save_labeled_json(output_json)
    
    # Export manual labeling template
    print("\n📝 Exporting manual labeling template...")
    labeler.export_for_manual_labeling(manual_template)
    
    print("\n✅ Failure labeling complete!")
    print("\n💡 Next steps:")
    print(f"   1. Review and edit: {manual_template}")
    print(f"   2. Re-import with: labeler.import_manual_labels('{manual_template}')")
    print(f"   3. Proceed to Phase 3 (Analysis)")


if __name__ == "__main__":
    main()