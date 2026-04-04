import importlib
import json
import regex as re

from typing import Tuple, Any
from datetime import datetime


def try_parse_json(raw: str) -> Tuple[bool, Any]:
    """
    Extracts the first valid JSON object from a string.
    Handles:
    - code fences
    - markdown
    - extra text
    - multiple JSON blocks
    """
    JSON_OBJECT_REGEX = re.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)

    # 1. Remove code fences like ```json ... ```
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "")

    # 2. Find the first {...} block using recursive regex
    match = JSON_OBJECT_REGEX.search(cleaned)
    if not match:
        return False, None

    candidate = match.group(0)

    # 3. Try to parse it
    try:
        return True, json.loads(candidate)
    except json.JSONDecodeError:
        return False, None


def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if __name__ == "__main__":

    raw = """```json
{
  "id": "fridge_humming_noise_001",
  "question": "My refrigerator is making a loud humming noise and not cooling properly.",
  "answer": "Check the compressor and evaporator fan for any obstructions or damage. If the hum is loud and continuous, the compressor might be failing and may need replacement. It's also worth checking the condenser coils for dust buildup, as this can affect cooling efficiency.",
  "equipment_problem": "loud humming noise and inadequate cooling",
  "tools_required": ["screwdriver", "multimeter", "soft brush", "vacuum cleaner"],
  "steps": [
    "Unplug the refrigerator for safety.",
    "Remove the back access panel to examine the compressor and fan.",
    "Check for obstructions around the fan and clear any dust from the condenser coils.",
    "Use a multimeter to test the compressor for functionality.",
    "If the compressor is faulty, schedule a replacement with a professional."
  ],
  "safety_info": "Always unplug the refrigerator before performing any repairs to avoid electric shock. Wear gloves to protect your hands.",
  "tips": "Regularly clean the coils to maintain efficiency. If you're not comfortable with electrical components, consider calling a professional for compressor issues."
}
```"""

    ok, data = try_parse_json(raw)
    print(ok, data)
