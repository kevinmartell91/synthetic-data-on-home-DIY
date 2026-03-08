"""
Generic runner for a pipeline phase.

Responsibilities:
- Load input from previous phase
- Execute the phase's run function
- Save output to this phase's folder
"""

from pathlib import Path
from typing import Callable, Any, Optional, Dict
from .io import load_dataset, save_dataset
from .utils import timestamp
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

if not os.path.exists(DATA_DIR):
    print(f"Creating data dir: {DATA_DIR}")
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


class PhaseRunner:
    """
    Generic runner for a pipeline phase.
    """

    def __init__(
        self,
        phase_name: str,
        input_phase: Optional[str],
        output_phase: str,
        run_fn: Callable[[Any], Any],
        run_fn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the runner.
        Args:
            phase_name: The name of the phase.
            input_phase: The name of the previous phase (optional).
            output_phase: The name of the current phase.
            run_fn: The function to run the phase.
        Returns:
            None
        """
        self.phase_name = phase_name
        self.input_path = (
            Path(f"{DATA_DIR}/{input_phase}/output.json") if input_phase else None
        )
        self.output_path = Path(f"{DATA_DIR}/{output_phase}/output_{timestamp()}.json")
        self.run_fn = run_fn
        self.run_fn_kwargs = run_fn_kwargs or {}

    def load_input(self):
        """
        Load the input dataset from the previous phase.
        """
        if self.input_path and self.input_path.exists():
            return load_dataset(self.input_path)
        return None

    def run(self):
        """
        Run the phase.
        """
        print(f"\n=== Running Phase: {self.phase_name} ===")

        # 1. Load input
        input_data = self.load_input()

        # 2. Execute phase logic
        output_data = self.run_fn(input_data, **self.run_fn_kwargs)

        # 3. Save output
        save_dataset(output_data, self.output_path)

        print(f"✓ Output saved to: {self.output_path}\n")
