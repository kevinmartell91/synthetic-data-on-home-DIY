# Phase 1 = “LLM JSON sanity check”

```json
{
  "phase_01_generation_message": {
    "summary": "Phase 1 performs the initial structural quality check.",
    "analogy": "Like a factory verifying that each part has the correct shape before deeper inspection.",
    "purpose": "Ensure the LLM output is valid JSON and matches the expected schema.",
    "why_it_matters": "Only structurally correct items move forward to semantic validation in Phase 2."
  }
}
```

## In Phase 1 we’re doing things like:

- Did the LLM return valid JSON?
- Does it match the Pydantic schema?
- Are required fields present?
- Are lists the right type?
- Is the structure parseable?
- This is basic structural correctness.

## It ensures the LLM didn’t hallucinate garbage like:

- missing fields
- wrong types
- broken JSON
- random text around the JSON
- empty lists
- invalid enum values

## Run this module

This module is responsible for preparing the output directory used by Phase 01.
It ensures that the folder structure exists before saving any generated synthetic data,
so the phase can run reliably without manual setup.

```bash
python -m 01_generation.run --num-samples 20
```

