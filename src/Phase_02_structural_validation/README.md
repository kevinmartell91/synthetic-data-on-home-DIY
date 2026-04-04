# Phase 2 = “Domain-level structural validation”

```json
{
  "phase_02_structural_validation_message": {
    "summary": "Phase 2 validates domain meaning, consistency across fields, and batch-level quality—semantic, cross-item, and pipeline-level.",
    "analogy": "Like a building inspector who checks not only that blueprints parse (Phase 1) but that steps match tools, safety fits the repair, and the whole run is not full of duplicates or templates.",
    "purpose": "Catch issues Pydantic cannot: realistic steps and tools, logical alignment between question, category, and answer, cross-item patterns, and quality gating before downstream use.",
    "why_it_matters": "Schema-valid JSON can still be nonsensical, unsafe, or low-quality; Phase 2 rejects those so only trustworthy items advance."
  }
}
```

## Phase 2 checks things that Pydantic cannot check, such as:

### Domain constraints

- Does steps contain actual steps, not one long paragraph?
- Does tools_required contain realistic homeowner tools?
- Is equipment_problem consistent with the question?
- Is the answer actually step‑by‑step?

### Logical consistency

- Do the steps reference tools that appear in tools_required?
- Does the safety info match the type of repair?
- Does the question match the category?

### Cross‑item quality checks

- Are there duplicates?
- Are there repeated patterns?
- Are there template artifacts?
- Are there category imbalances?

### Quality gating

- Reject items that are structurally correct but low‑quality
- Reject items that are structurally correct but nonsensical
- Reject items that are structurally correct but unsafe
