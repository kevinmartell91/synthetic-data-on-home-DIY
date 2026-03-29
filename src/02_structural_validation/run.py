from typing import Any
from pipeline_core.runner import PhaseRunner
from pipeline_core.io import save_report
from pydantic_classes import DIYRepairSyntheticItem
from .structure_validator import StructureValidator
from .synthetic_analyzer import SyntheticAnalyzer


def run_phase(input_data: Any, output_path: str):
    print("Running phase 02: Structural Validation")

    # validate structure
    validator = StructureValidator(input_data)
    dataset, report = validator.validate(schema=DIYRepairSyntheticItem)
    validator.print_summary()
    save_report(report, output_path, "validator_results.json")

    # syntethic analyser
    analyzer = SyntheticAnalyzer(input_data)
    results = analyzer.run_comprehensive_evaluation()
    analyzer.print_summary(results)
    save_report(results, output_path, "analyzer_results.json")

    return dataset


if __name__ == "__main__":

    PhaseRunner(
        phase_name="02_structural_validation",
        input_phase="phase_01",
        output_phase="phase_02",
        run_fn=run_phase,
        run_fn_kwargs={"output_path": "data/phase_02"},
    ).run()
