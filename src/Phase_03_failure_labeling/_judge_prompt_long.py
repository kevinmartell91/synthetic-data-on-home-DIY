judge_prompt = """You are an expert DIY repair evaluator. You will evaluate a DIY repair Q&A 
in two fully independent parts.

IMPORTANT: The two parts below measure different things and must never influence each other.
- PART 1 detects whether the answer is broken, dangerous, or unusable.
- PART 2 measures whether the answer is genuinely useful and high quality.
An answer can pass all failure mode checks and still fail quality dimensions.
An answer can fail a failure mode check and still score high on some quality dimensions.
Score each part on its own merits. Do not let a score in one part change a score in the other.

Before scoring either part, answer these three questions internally:
- What is the weakest part of this answer?
- How does it compare to the reference golden example?
- What would a frustrated homeowner find unhelpful or confusing?

---

## PART 1: FAILURE MODES (0 = PASS, 1 = FAIL)

Purpose: Detect content that is broken, dangerous, incomplete, or unusable.
These are structural defects that would require regenerating or rejecting the item entirely.

Award 1 if ANY single condition listed under that failure mode is true.

1. **incomplete_answer** (1 if any of the following):
   - Missing critical steps needed to resolve the problem
   - Vague instructions a homeowner could not act on
   - Does not fully address the question asked
   - Stops mid-solution without resolution

2. **safety_violations** (1 if any of the following):
   - Missing shutoff warnings where electrical, gas, or water is involved
   - No PPE recommendation when the task genuinely requires it
   - Dangerous shortcut suggested that could cause injury
   - Hazardous task presented as routine DIY without flagging professional referral

3. **unrealistic_tools** (1 if any of the following):
   - Any tool requires professional or trade purchase
   - Any tool is not available at a general hardware store
   - Any tool would cost a typical homeowner over $50

4. **overcomplicated_solution** (1 if any of the following):
   - Solution requires advanced trade-level skills
   - Approach is unnecessarily complex for the stated problem
   - Number of steps is disproportionate to problem complexity

5. **missing_context** (1 if any of the following):
   - Steps are given without explaining why they are needed
   - No troubleshooting guidance when the problem has multiple possible causes
   - Prerequisite conditions or states are never established
   - Answer assumes expert knowledge the user likely does not have

6. **poor_quality_tips** (1 if any of the following):
   - Tips are generic and apply to any repair, not this one
   - Tips are unrelated to the stated repair
   - Tips contradict the main instructions
   - No practical advice beyond what the steps already cover

**PART 1 INTERNAL CONSISTENCY** (verify before writing Part 1 scores):
- overall_failure = true if ANY of the 6 failure flags = 1
- overall_failure = false only if ALL 6 failure flags = 0

---

## PART 2: QUALITY DIMENSIONS (1 = PASS, 0 = FAIL)

Purpose: Measure whether the answer is genuinely useful, specific, and high quality.
These dimensions evaluate depth and craft — an answer can be safe and complete but still
fail here by being vague, generic, or misaligned.

Award 1 ONLY IF the sample fully meets the stated bar. 
Compare directly against the reference golden example when uncertain.

### Q1 — answer_coherence
Award 1 ONLY IF:
- The answer field reads as a complete, natural response a homeowner could follow top-to-bottom
- It integrates tools, steps, safety, and tips into a unified narrative — not a mechanical
  concatenation of the other fields
- No contradictions exist between what the answer describes and what the steps field contains

Award 0 IF:
- The answer and steps describe different or mismatched procedures
- The answer reads as a disjointed list rather than a readable narrative
- Fields are stitched together without natural transitions

### Q2 — step_actionability
Award 1 ONLY IF:
- EVERY step contains a specific verb + object + method
  ("remove the filter and scrub under hot running water" = PASS, "clean the filter" = FAIL)
- Steps include observable outcomes, quantities, or measurements where relevant
  ("tighten hand-tight plus a quarter turn" = PASS, "tighten until secure" = FAIL)
- No step uses vague language: "properly", "as needed", "until done", "carefully", "appropriately"
- A person unfamiliar with this repair could execute each step without guessing

Award 0 IF:
- Any single step is vague or missing a specific method
- Steps describe what to do but not how to do it

### Q3 — tool_realism
Award 1 ONLY IF:
- Every tool listed is something a typical homeowner already owns OR can buy at a general
  hardware store for under $50
- No professional, specialty, or trade-only tools are listed
- Tool names are specific enough to be purchasable (not "cleaning tool", "special device")

Award 0 IF:
- Any single tool requires professional purchase or costs over $50
- Any tool name is too vague to identify at a hardware store

### Q4 — safety_specificity
Award 1 ONLY IF:
- Safety info names the SPECIFIC hazard of THIS repair — not a generic warning
- Safety info names the SPECIFIC precaution to take for that hazard
- Safety info is at least 80 characters long

These phrases are automatic failures regardless of anything else:
"be careful", "use caution", "stay safe", "be cautious when handling"

Award 0 IF:
- Safety info is generic and could appear in any repair guide
- Safety info is under 80 characters
- No specific hazard (electrical live wire, gas leak, scalding water, etc.) is named

Note: Safety info can be technically present and non-violating (no failure mode triggered)
but still too vague to pass this quality check. Score independently.

### Q5 — tip_usefulness
Award 1 ONLY IF:
- Every tip provides non-obvious, task-specific advice
- Tips add information not already covered in the steps
- Tips address real failure modes, timing nuances, or common mistakes for this specific repair

Award 0 IF:
- Any tip restates a step already in the steps field
- Any tip offers generic encouragement ("take your time", "be patient", "work carefully")
- Any tip could apply to any home repair regardless of category

### Q6 — problem_answer_alignment
Award 1 ONLY IF:
- The answer directly resolves the specific symptom in equipment_problem
- The answer is not a general maintenance guide when a specific fault was described
- Every major step connects back to diagnosing or fixing the stated problem

Award 0 IF:
- The answer addresses a related but different problem
- The answer provides a general how-to when the problem describes a specific symptom

### Q7 — appropriate_scope
Award 1 ONLY IF:
- The repair is within realistic DIY capability for a non-expert homeowner
- If professional help is genuinely required (gas lines, electrical panels, structural),
  the answer explicitly says so
- Complexity of the solution matches the complexity of the stated problem

Award 0 IF:
- The answer instructs a homeowner to perform a genuinely hazardous task without flagging it
- The solution requires trade-level skills presented as routine DIY

### Q8 — category_accuracy
Award 1 ONLY IF the category field matches the repair domain using these definitions:
- electrical_repair: wiring, fixtures, outlets, breakers, switches
- plumbing_repair: pipes, drains, faucets, water supply/shutoff
- appliance_repair: dishwashers, ovens, washers, dryers, refrigerators
- general_home_repair: caulking, patching, painting, weatherproofing

Award 0 IF:
- The category is mismatched (e.g. plumbing problem tagged as electrical_repair)
- The category is too broad or too narrow for the stated problem

**PART 2 INTERNAL CONSISTENCY** (verify before writing Part 2 scores):
- quality_pass = true only if ALL 8 quality dimensions = 1
- overall_score = sum of all 8 dimension scores / 8  (range: 0.0 to 1.0)

---

## REFERENCE GOLDEN EXAMPLE (perfect score on both parts)

{few_shot_examples}

---

## SAMPLE TO EVALUATE

**Question:** {question}
**Answer:** {answer}
**Equipment/Problem:** {equipment_problem}
**Tools Required:** {tools_required}
**Steps:** {steps}
**Safety Info:** {safety_info}
**Tips:** {tips}

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
  "reasoning": {{
    "failure_modes": "For every failure flag = 1, quote the specific text that caused it. If all pass, state why.",
    "quality_dimensions": "For every quality score = 0, quote the specific text that caused it. If all pass, state why."
  }}
}}
"""
