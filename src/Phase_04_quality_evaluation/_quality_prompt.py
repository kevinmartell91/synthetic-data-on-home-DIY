quality_prompt = """You are an expert DIY repair evaluator scoring a synthetic Q&A in QUALITY DIMENSIONS.

## QUALITY DIMENSIONS (1 = pass, 0 = fail)
Award 1 ONLY IF the sample fully meets the bar. Compare against the golden reference below.

- **answer_coherence**: unified narrative top-to-bottom; no contradiction between answer and steps fields
- **step_actionability**: every step has verb + object + method; no vague language ("properly", "as needed", "carefully"); measurable outcome where relevant. ("scrub filter under hot running water" = PASS / "clean the filter" = FAIL)
- **tool_realism**: every tool is homeowner-owned or buyable at hardware store under $50; specific enough to be purchasable
- **safety_specificity**: names the specific hazard AND specific precaution for THIS repair; 80+ characters; auto-fail: "be careful", "use caution", "stay safe"
- **tip_usefulness**: every tip is non-obvious and not already covered in steps; no generic encouragement; specific to this repair type
- **problem_answer_alignment**: directly resolves the specific symptom in equipment_problem; not a general maintenance guide
- **appropriate_scope**: within non-expert DIY capability; if professional help is needed, explicitly says so
- **category_accuracy**: category matches domain (electrical_repair / plumbing_repair / appliance_repair / general_home_repair)

quality_pass = true only if ALL 8 dimensions = 1.
overall_score = sum of 8 scores / 8.

---

## SAMPLE TO EVALUATE

Question: {question}
Answer: {answer}
Equipment/Problem: {equipment_problem}
Tools Required: {tools_required}
Steps: {steps}
Safety Info: {safety_info}
Tips: {tips}

---

Respond ONLY with valid JSON. No markdown, no preamble.

{{
  "quality_scores": {{
    "answer_coherence": 0 or 1,
    "step_actionability": 0 or 1,
    "tool_realism": 0 or 1,
    "safety_specificity": 0 or 1,
    "tip_usefulness": 0 or 1,
    "problem_answer_alignment": 0 or 1,
    "appropriate_scope": 0 or 1,
    "category_accuracy": 0 or 1
  }},
  "overall_score": 0.0 to 1.0,
  "quality_pass": true or false,
  "reasoning": "For each score = 0, quote the text that caused it. If all pass, state why".
  }}
}}
"""
