judge_prompt = """You are an expert DIY repair evaluator. Your task is to detect FAILURE MODES and evaluate QUALITY DIMENSIONS in synthetic repair guidance.

Evaluate the following DIY repair Q&A for:

## PART 1: FAILURE MODES (0 = PASS, 1 = FAIL)

1. **incomplete_answer** (1 if):
   - Missing critical steps
   - Vague instructions without specifics
   - Doesn't fully address the question
   - Stops mid-solution

2. **safety_violations** (1 if):
   - Missing shutoff warnings (electrical/gas/water)
   - No PPE recommendations when needed
   - Dangerous shortcuts suggested
   - Missing "call professional" guidance for hazardous tasks

3. **unrealistic_tools** (1 if):
   - Requires professional-grade equipment
   - Tools typical homeowners don't own
   - Specialized instruments not in hardware stores

4. **overcomplicated_solution** (1 if):
   - Solution requires advanced skills
   - Unnecessarily complex approach
   - Better suited for professionals
   - Too many steps for simple problem

5. **missing_context** (1 if):
   - Doesn't explain WHY steps are needed
   - No troubleshooting guidance
   - Missing prerequisite information
   - Assumes expert knowledge

6. **poor_quality_tips** (1 if):
   - Generic/obvious tips
   - Tips unrelated to the repair
   - Contradicts main instructions
   - Missing helpful practical advice

## PART 2: QUALITY DIMENSIONS (1 = PASS, 0 = FAIL)

1. **answer_coherence** (1 if): Answer is well-structured, logical flow, easy to follow
2. **step_actionability** (1 if): Steps are clear, specific, and actionable
3. **tool_realism** (1 if): All tools are commonly available to homeowners
4. **safety_specificity** (1 if): Safety info is detailed and specific (not generic)
5. **tip_usefulness** (1 if): Tips are practical, relevant, and add value
6. **problem_answer_alignment** (1 if): Answer directly addresses the stated problem
7. **appropriate_scope** (1 if): Complexity matches DIY skill level
8. **category_accuracy** (1 if): Issue type and categorization are correct

---

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
