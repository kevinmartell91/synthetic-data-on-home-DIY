"""
Phase 07: Benchmark Comparison & Judge Calibration Runner
Validates the LLM judge against benchmark data and measures quality gap
"""

from typing import Any
from ..pipeline_core.runner import PhaseRunner
from ..pipeline_core.io import save_report
from .benchmark_loader import BenchmarkLoader
from .benchmark_analyzer import BenchmarkAnalyzer


def run_phase(input_data: Any, output_path: str):
    """
    Phase 07: Benchmark Comparison & Judge Calibration
    - Load benchmark dataset and sample >=50 items
    - Run quality judge on benchmark
    - Validate judge is calibrated (>=95% pass rate)
    - Compare benchmark vs generated dataset quality
    - Generate gap analysis report and visualization
    """
    benchmark_items = BenchmarkLoader().load_benchmark_dataset(num_samples=50)

    analyzer = BenchmarkAnalyzer()
    evaluated_items, benchmark_stats = analyzer.run_judge_on_benchmark(benchmark_items)
    calibrated, message = analyzer.check_calibration(benchmark_stats)
    print(message)

    generated_stats = analyzer.compute_generated_stats(input_data)
    gap_analysis = analyzer.compare_quality_gap(benchmark_stats, generated_stats)
    analyzer.create_gap_analysis_chart(
        gap_analysis, f"{output_path}/analysis/benchmark_vs_generated_gap.png"
    )

    report = analyzer.generate_final_report(
        benchmark_stats,
        generated_stats,
        gap_analysis,
        calibrated,
        f"{output_path}/analysis/benchmark_comparison_report.json",
    )
    analyzer.print_final_summary(calibrated, gap_analysis, report)
    save_report(benchmark_stats, f"{output_path}/analysis", "benchmark_statistics.json")

    return evaluated_items


if __name__ == "__main__":
    PhaseRunner(
        phase_name="07_benchmark_comparison",
        input_phase="phase_04",
        output_phase="phase_07",
        run_fn=run_phase,
        run_fn_kwargs={"output_path": "src/data/phase_07"},
    ).run()
