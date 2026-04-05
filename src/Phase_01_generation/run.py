import argparse
from ..pipeline_core.runner import PhaseRunner
from .synthetic_generator import SyntheticGenerator


def run_phase(input_data: any, num_samples: int):
    generator = SyntheticGenerator()
    dataset = generator.generate_synthetic_dataset(num_samples)
    return dataset


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Phase 01: Synthetic Data Generation")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1,
        help="Number of samples to generate",
    )
    args = parser.parse_args()

    PhaseRunner(
        phase_name="01_generation",
        input_phase=None,  # no previous phase
        output_phase="phase_01",
        run_fn=run_phase,
        run_fn_kwargs={"num_samples": args.num_samples},
    ).run()
