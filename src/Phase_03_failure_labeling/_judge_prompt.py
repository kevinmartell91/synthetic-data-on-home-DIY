judge_prompt = """You are an expert DIY repair evaluator scoring a synthetic Q&A in FAILURE MODES

## FAILURE MODES (0 = pass, 1 = fail)
Award 1 if ANY condition under that mode is true.

- **incomplete_answer**: missing critical steps / vague instructions / question not fully addressed
- **safety_violations**: missing shutoff warnings / dangerous shortcut / hazardous task not flagged for professional
- **unrealistic_tools**: trade-only tool / not in a hardware store / costs over $50
- **overcomplicated_solution**: requires trade-level skills / disproportionate steps for problem complexity
- **missing_context**: no explanation of why steps are needed / assumes expert knowledge / missing prerequisites
- **poor_quality_tips**: generic tips / restates steps / contradicts instructions / no practical value

overall_failure = true if ANY flag = 1.


## GOLDEN REFERENCE (perfect score on both parts)

{few_shot_examples}

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
  "incomplete_answer": 0 or 1,
  "safety_violations": 0 or 1,
  "unrealistic_tools": 0 or 1,
  "overcomplicated_solution": 0 or 1,
  "missing_context": 0 or 1,
  "poor_quality_tips": 0 or 1,
  "overall_failure": true or false,
  "reasoning": "For each flag = 1, quote the text that caused it. If all pass, state why." 
}}
"""


_judge_prompt = """You are an expert DIY repair evaluator. Your task is to detect FAILURE MODES and evaluate QUALITY DIMENSIONS in synthetic repair guidance.

Evaluate the following DIY repair Q&A for:

## PART 1: FAILURE MODES (0 = PASS, 1 = FAIL)

1. **incomplete_answer** (1 if any of the following):
   - Missing critical steps
   - Vague instructions without specifics
   - Doesn't fully address the question
   - Stops mid-solution

2. **safety_violations** (1 if any of the following):
   - Missing shutoff warnings (electrical/gas/water)
   - No PPE recommendations when needed
   - Dangerous shortcuts suggested
   - Missing "call professional" guidance for hazardous tasks

3. **unrealistic_tools** (1 if any of the following):
   - Requires professional-grade equipment
   - Tools typical homeowners don't own
   - Specialized instruments not in hardware stores

4. **overcomplicated_solution** (1 if any of the following):
   - Solution requires advanced skills
   - Unnecessarily complex approach
   - Better suited for professionals
   - Too many steps for simple problem

5. **missing_context** (1 if any of the following):
   - Doesn't explain WHY steps are needed
   - No troubleshooting guidance
   - Missing prerequisite information
   - Assumes expert knowledge

6. **poor_quality_tips** (1 if any of the following):
   - Generic/obvious tips
   - Tips unrelated to the repair
   - Contradicts main instructions
   - Missing helpful practical advice

## PART 2: QUALITY DIMENSIONS (1 = PASS, 0 = FAIL)

1. **answer_coherence** (award with 1 if all): Answer is well-structured, logical flow, easy to follow
2. **step_actionability** (award with 1 if all): Steps are clear, specific, and actionable
3. **tool_realism** (award with 1 if all): All tools are commonly available to homeowners
4. **safety_specificity** (award with 1 if all): Safety info is detailed and specific (not generic)
5. **tip_usefulness** (award with 1 if all): Tips are practical, relevant, and add value
6. **problem_answer_alignment** (award with 1 if all): Answer directly addresses the stated problem
7. **appropriate_scope** (award with 1 if all): Complexity matches DIY skill level
8. **category_accuracy** (award with 1 if all): Issue type and categorization are correct

---

**REFERENCE EXAMPLE** (score = perfect):
```
{few_shot_examples}
```

Now evaluate the following against that standard:

**SAMPLE TO EVALUATE:**

**Question:** {question}

**Answer:** {answer}

**Equipment/Problem:** {equipment_problem}

**Tools Required:** {tools_required}

**Steps:** {steps}

**Safety Info:** {safety_info}

**Tips:** {tips}

---

Respond ONLY with valid JSON in this exact format:
{{
  "incomplete_answer": 0 or 1,
  "safety_violations": 0 or 1,
  "unrealistic_tools": 0 or 1,
  "overcomplicated_solution": 0 or 1,
  "missing_context": 0 or 1,
  "poor_quality_tips": 0 or 1,
  "overall_failure": true or false,
  "quality_scores": {{
    "answer_coherence": 1 or 0,
    "step_actionability": 1 or 0,
    "tool_realism": 1 or 0,
    "safety_specificity": 1 or 0,
    "tip_usefulness": 1 or 0,
    "problem_answer_alignment": 1 or 0,
    "appropriate_scope": 1 or 0,
    "category_accuracy": 1 or 0
  }},
  "quality_pass": true or false,
  "reasoning": "Detailed explanation of failures and quality issues"
}}
"""
