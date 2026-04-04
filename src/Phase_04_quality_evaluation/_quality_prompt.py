quality_prompt = """
You are evaluating a DIY repair Q&A pair across 8 quality dimensions.

# QUALITY DIMENSIONS (Binary: 1 = pass, 0 = fail)

1. answer_coherence: Answer flows logically and integrates all fields coherently
2. step_actionability: Steps are clear, specific, and actionable (not vague)
3. tool_realism: Tools are realistic for typical homeowner to own
4. safety_specificity: Safety info is specific to this repair (not generic)
5. tip_usefulness: Tips add genuine value beyond the obvious
6. problem_answer_alignment: Answer directly addresses the question asked
7. appropriate_scope: Repair is appropriate for DIY (not requiring professional)
8. category_accuracy: Category matches the actual repair type

# SCORING RULES

- Each dimension: 1 = pass, 0 = fail
- quality_pass: true if ALL 8 dimensions pass (all 1s)
- overall_score: sum of dimensions / 8 (0.0 to 1.0)

# OUTPUT FORMAT (JSON only, no markdown)

{
  "quality_scores": {
    "answer_coherence": 0 or 1,
    "step_actionability": 0 or 1,
    "tool_realism": 0 or 1,
    "safety_specificity": 0 or 1,
    "tip_usefulness": 0 or 1,
    "problem_answer_alignment": 0 or 1,
    "appropriate_scope": 0 or 1,
    "category_accuracy": 0 or 1
  },
  "quality_pass": true or false,
  "overall_score": 0.0 to 1.0,
  "reasoning": "Brief explanation of quality assessment"
}

# SAMPLE TO EVALUATE

{sample_json}

Respond with JSON only.
"""