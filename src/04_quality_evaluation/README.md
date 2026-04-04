# Phase 4 = "LLM-as-Judge Quality Dimension Scoring"

```json
{
  "phase_04_quality_evaluation_message": {
    "summary": "Phase 4 uses an independent LLM as a judge to score each sample across 8 quality dimensions, measuring how close generated data is to benchmark standards.",
    "analogy": "Like a quality inspector who grades each product on multiple criteria (durability, finish, accuracy, etc.) to ensure it meets professional standards.",
    "purpose": "Quantify quality across 8 dimensions to identify which aspects of generated data need improvement and how far they are from benchmark quality.",
    "why_it_matters": "Passing failure detection (Phase 3) doesn't guarantee high quality. Phase 4 measures whether your data is actually good enough to use."
  }
}
```

## In Phase 4 we're evaluating:

### Quality Dimensions (Binary: 1 = pass, 0 = fail)

1. **answer_coherence**: Does the answer flow logically and integrate all fields?
2. **step_actionability**: Are steps clear, specific, and actionable?
3. **tool_realism**: Are tools realistic for a typical homeowner?
4. **safety_specificity**: Is safety info specific to this repair?
5. **tip_usefulness**: Do tips add genuine value beyond the obvious?
6. **problem_answer_alignment**: Does the answer directly address the question?
7. **appropriate_scope**: Is the repair appropriate for DIY?
8. **category_accuracy**: Does the category match the repair type?

## Run this module

```bash
python -m src.04_quality_evaluation.run
```

## Output Structure

```
data/phase_04/
├── output.jsonl                          # Samples with quality scores
└── analysis/
    ├── quality_statistics.json           # Overall quality metrics
    ├── quality_analysis_report.json      # Detailed dimension analysis
    ├── quality_heatmap.png               # Samples × dimensions heatmap
    └── quality_dimension_distribution.png # Pass rates per dimension
```

## Success Criteria

- ✅ All samples evaluated with quality scores
- ✅ Overall quality pass rate ≥ 80% (all 8 dimensions pass)
- ✅ Average quality score ≥ 0.80
- ✅ Visualizations show dimension-level performance
- ✅ Ready for Phase 5 (combined failure + quality analysis)