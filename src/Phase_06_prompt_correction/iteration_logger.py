"""
Iteration Logging
Tracks prompt corrections and improvements over iterations
"""

from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path


class IterationLogger:
    """
    Logs iteration details to architectural documentation
    """

    @staticmethod
    def create_iteration_log_entry(
        iteration_num: int,
        analysis_summary: str,
        corrections_applied: List[Dict[str, Any]],
        improvement_ratio: float,
        success: bool,
    ) -> str:
        """
        Create a markdown log entry for this iteration

        Returns:
            Formatted markdown string
        """
        timestamp = datetime.now().isoformat()
        status = "✅ SUCCESS" if success else "⚠️  IN PROGRESS"

        entry = f"""
## Iteration {iteration_num}

**Date:** {timestamp}
**Status:** {status}

### Analysis
{analysis_summary}

### Corrections Applied
"""

        for correction in corrections_applied:
            entry += (
                f"- **{correction.get('template_affected', 'Unknown')}**: "
                f"{correction.get('change', 'Updated')}\n"
                f"  - Reason: {correction.get('reason', 'Improvement')}\n"
            )

        entry += f"""
### Improvement
- **Improvement Ratio:** {improvement_ratio:+.1%}
- **Target:** >=80%
- **Result:** {'✅ Exceeded' if improvement_ratio >= 0.80 else '⚠️  Did not meet'} target

---

"""
        return entry

    @staticmethod
    def append_to_iteration_log(
        iteration_num: int,
        analysis_summary: str,
        corrections: List[Dict[str, Any]],
        improvement_ratio: float,
        success: bool,
        log_file_path: str = "src/data/phase_06/analysis/iteration_logs.md",
    ):
        """Append iteration entry to the iteration logs file"""
        entry = IterationLogger.create_iteration_log_entry(
            iteration_num, analysis_summary, corrections, improvement_ratio, success
        )

        try:
            log_path = Path(log_file_path)

            # Create file with header if it doesn't exist
            if not log_path.exists():
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_file_path, "w", encoding="utf-8") as f:
                    f.write(
                        "# Iteration Logs - Phase 06 Prompt Correction\n\nThis file logs iterations of prompt corrections and refinements for the Phase 06 prompt correction pipeline.\n\n## Iterations\n\n"
                    )

            with open(log_file_path, "a", encoding="utf-8") as f:
                f.write(entry)
            print(f"📝 Iteration {iteration_num} logged to {log_file_path}")
        except Exception as e:
            print(f"⚠️  Could not write iteration log: {e}")
