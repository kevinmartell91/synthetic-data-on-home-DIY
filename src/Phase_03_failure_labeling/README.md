# Phase 3 = "LLM-as-Judge Failure Detection & Pattern Analysis"

```json
{
  "phase_03_failure_labeling_message": {
    "summary": "Phase 3 uses an independent LLM as a judge to evaluate each sample across 6 failure modes and 8 quality dimensions, then analyzes patterns to identify root causes.",
    "analogy": "Like a quality inspector who not only checks each product for defects but also tracks which defects appear together, revealing systemic manufacturing issues.",
    "purpose": "Detect semantic failures that schema validation cannot catch, identify correlated failure patterns, and pinpoint which prompt templates need correction.",
    "why_it_matters": "Schema-valid JSON can still be unsafe, incomplete, or low-quality. Phase 3 reveals what's actually wrong and where to fix it."
  }
}
```

## In Phase 3 we're doing things like:

### LLM-as-Judge Evaluation (Binary Failure Modes)

- Is the answer incomplete or vague?
- Does it violate safety guidelines?
- Does it recommend unrealistic tools?
- Is the solution overcomplicated for a homeowner?
- Is critical context missing?
- Are the tips low-quality or generic?

### LLM-as-Judge Evaluation (Quality Dimensions)

- **Answer Coherence**: Does the answer read as a unified narrative?
- **Step Actionability**: Are steps specific enough to follow?
- **Tool Realism**: Are tools realistic for homeowners?
- **Safety Specificity**: Are safety warnings task-specific?
- **Tip Usefulness**: Do tips provide non-obvious value?
- **Problem-Answer Alignment**: Does the answer match the question?
- **Appropriate Scope**: Is the repair safe for homeowners?
- **Category Accuracy**: Does the category match the repair type?

### Failure Pattern Analysis

- Which failure modes are most frequent?
- Are certain failures correlated? (e.g., `missing_context` ↔ `incomplete_answer`)
- Which repair category produces the most failures?
- Which samples have 3+ simultaneous failures?

## Visualizations Created

All visualizations follow the requirements from `11_visualization_requirements.md`:

### 1. Failure Mode Co-occurrence Heatmap (`01_failure_cooccurrence_heatmap.png`)

- **Purpose**: Shows which failure modes tend to appear together
- **Insight**: Does `missing_context` correlate with `incomplete_answer`?
- **Color Scale**: Diverging (RdBu_r) - red for positive correlation, blue for negative
- **Axes**: Labeled with failure mode names
- **Title**: Clear and descriptive

### 2. Failure Rates by Repair Category (`02_failure_rates_by_category.png`)

- **Purpose**: Compares failure rates across all repair categories
- **Insight**: Which prompt template produces the most failures?
- **Color Scale**: Sequential (YlOrRd) - darker = higher failure rate
- **Axes**: Category names (x), Failure Rate % (y)
- **Labels**: Shows percentage and count on each bar

### 3. Per-Mode Failure Trend (`03_per_mode_failure_trend.png`)

- **Purpose**: Shows each failure mode individually (baseline vs. post-correction)
- **Insight**: Not just overall failure rate, but per-mode improvements
- **Color Scale**: Red for baseline, green for post-correction
- **Axes**: Failure modes (x), Failure Rate % (y)
- **Labels**: Percentage on each bar

### 4. Most Problematic Items (`04_most_problematic_items.png`)

- **Purpose**: Highlights items with 3+ simultaneous failure flags
- **Insight**: These are the worst cases and best diagnostic targets
- **Visualization**: Stacked horizontal bar chart
- **Axes**: Sample ID (y), Number of failures (x)
- **Legend**: Shows which failure modes each item has

### 5. Quality Dimension Scores (`05_quality_dimension_scores.png`)

- **Purpose**: Shows pass rates across all 8 quality dimensions
- **Insight**: Baseline vs. post-correction scores
- **Color Scale**: RdYlGn (red = low pass rate, green = high pass rate)
- **Axes**: Quality dimensions (x), Pass Rate % (y)
- **Threshold Line**: 80% pass rate threshold marked

## It ensures the LLM didn't produce:

- Dangerous advice (e.g., "just touch the live wire")
- Vague steps (e.g., "tighten until secure")
- Unrealistic tools (e.g., "use a thermal imaging camera")
- Overcomplicated solutions (e.g., "rebuild the entire appliance")
- Missing context (e.g., no mention of turning off power)
- Generic tips (e.g., "be careful" or "stay safe")

## Phase 3 checks things that Pydantic cannot check:

### Semantic Quality

- Does the answer actually help a homeowner?
- Are the steps actionable or just descriptions?
- Are safety warnings specific to the hazard?
- Do tips provide real value or just filler?

### Domain Expertise

- Would a professional recommend this approach?
- Are the tools realistic for a typical homeowner?
- Is the scope appropriate for DIY vs. professional work?
- Does the category match the repair type?

### Cross-Field Consistency

- Do the steps reference tools from `tools_required`?
- Does the safety info match the repair type?
- Does the answer integrate all fields coherently?

### Pattern Detection

- Are certain failure modes always appearing together?
- Is one repair category failing more than others?
- Are there systemic issues in the generation prompts?

## Run this module

This module evaluates all samples from Phase 02 using an LLM-as-Judge, then analyzes failure patterns and creates visualizations.

```bash
python -m src.03_failure_labeling.run
```

## Output Structure

```
data/phase_03/
├── output.jsonl                          # Labeled samples with failure judgments
└── analysis/
    ├── failure_statistics.json           # Overall failure rates and counts
    ├── failure_analysis_report.json      # Detailed analysis with correlations
    ├── failure_heatmap.png               # Samples × failure modes heatmap
    ├── failure_correlation.png           # Failure mode correlation matrix
    ├── failure_by_issue_type.png         # Failures grouped by repair category
    └── quality_heatmap.png               # Quality dimension scores heatmap
```

## Key Outputs

### 1. Failure Statistics (`failure_statistics.json`)

```json
{
  "total_samples": 20,
  "overall_failure_rate": 0.35,
  "failure_mode_counts": {
    "incomplete_answer": 5,
    "safety_violations": 3,
    "unrealistic_tools": 2,
    "overcomplicated_solution": 4,
    "missing_context": 6,
    "poor_quality_tips": 7
  },
  "quality_dimension_pass_rates": {
    "answer_coherence": 0.85,
    "step_actionability": 0.75,
    ...
  }
}
```

### 2. Failure Analysis Report (`failure_analysis_report.json`)

```json
{
  "common_failures": {
    "most_common_failure": ["poor_quality_tips", 7],
    "samples_with_multiple_failures": 8,
    "average_failures_per_sample": 1.35
  },
  "correlations": [
    {
      "mode1": "missing_context",
      "mode2": "incomplete_answer",
      "correlation": 0.72
    }
  ],
  "issue_type_analysis": {
    "electrical_repair": {
      "safety_violations": 5,
      "missing_context": 3
    }
  }
}
```

### 3. Labeled Samples (`output.jsonl`)

Each sample now includes:

```json
{
  "question": "...",
  "answer": "...",
  "failure_judgment": {
    "incomplete_answer": 0,
    "safety_violations": 1,
    "unrealistic_tools": 0,
    "overcomplicated_solution": 0,
    "missing_context": 1,
    "poor_quality_tips": 0,
    "overall_failure": true,
    "quality_scores": {
      "answer_coherence": 1,
      "step_actionability": 1,
      "tool_realism": 1,
      "safety_specificity": 0,
      ...
    },
    "quality_pass": false,
    "reasoning": "Safety warning is generic..."
  }
}
```

## What Phase 3 Reveals

### Diagnostic Insights

- **Most problematic failure mode**: Which quality issue is most common?
- **Correlated failures**: Which failures share a root cause?
- **Worst-performing category**: Which prompt template needs the most work?
- **Quality gaps**: Which dimensions are furthest from benchmark standards?

### Actionable Next Steps

- If `safety_violations` and `missing_context` correlate → Add explicit safety checklist to prompts
- If `electrical_repair` has 3× more failures → Focus prompt corrections on that template
- If `poor_quality_tips` is most common → Add examples of good vs. bad tips to prompts
- If `overcomplicated_solution` appears often → Add "homeowner-appropriate" constraint

## Success Criteria

- ✅ All samples evaluated with structured judgments
- ✅ Failure statistics computed and saved
- ✅ Correlation matrix identifies co-occurring failures
- ✅ Visualizations reveal actionable patterns
- ✅ Analysis report documents root causes
- ✅ Ready for Phase 6 prompt correction

## Why This Phase Matters

Phase 3 is your **diagnostic engine**. It doesn't just tell you "this is bad"—it tells you:

- **What** is bad (which failure modes)
- **How bad** (failure rates and correlations)
- **Where** it's bad (which repair categories)
- **Why** it's bad (correlated patterns suggest root causes)

Without Phase 3, you're guessing. With Phase 3, you're making **data-driven decisions** about how to fix your prompts.
