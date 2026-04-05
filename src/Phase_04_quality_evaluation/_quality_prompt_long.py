quality_prompt = """
You are an expert DIY repair evaluator. Your task is to score a synthetic DIY repair Q&A 
across 8 quality dimensions by comparing it against the reference golden example below.

Before scoring, answer these three questions internally:
- What is the weakest part of this answer?
- How does it compare to the reference golden example?
- What would a frustrated homeowner find unhelpful or confusing?

Then score each dimension.

---

# QUALITY DIMENSIONS

Award 1 only if the sample fully meets the stated bar. 
When in doubt, compare directly against the reference example above.

## Q1 — answer_coherence
Award 1 ONLY IF:
- The answer field reads as a complete, natural response a homeowner could follow top-to-bottom
- It integrates tools, steps, safety, and tips into a unified narrative
- It does NOT read as a mechanical copy-paste of the other fields stitched together
- There are no contradictions between what the answer describes and what the steps field contains

Award 0 IF:
- The answer and steps describe different procedures
- The answer is a disjointed list rather than a readable narrative
- Fields are concatenated without natural transitions

## Q2 — step_actionability
Award 1 ONLY IF:
- EVERY step contains a specific verb + object + method (e.g. "remove the filter and scrub under hot running water")
- Steps include observable outcomes, quantities, or measurements where relevant
  (e.g. "tighten hand-tight plus a quarter turn" NOT "tighten until secure")
- No step uses vague language: "properly", "as needed", "until done", "carefully", "appropriately"
- A person unfamiliar with this repair could execute each step without guessing

Award 0 IF:
- Any single step is vague or missing a specific method
- Steps say what to do but not how to do it
- Outcomes are unmeasurable ("clean thoroughly", "check connections")

## Q3 — tool_realism
Award 1 ONLY IF:
- Every tool listed is something a typical homeowner already owns OR can buy at a general 
  hardware store for under $50
- No professional, specialty, or trade-only tools are listed
- Tool names are specific enough to be purchasable (not "cleaning tool" or "special device")

Award 0 IF:
- Any single tool requires professional purchase or costs over $50
- Any tool is vague or unidentifiable at a hardware store

## Q4 — safety_specificity
Award 1 ONLY IF:
- Safety info names the SPECIFIC hazard of THIS repair (not a generic warning)
- Safety info names the SPECIFIC precaution to take for that hazard
- Safety info is at least 80 characters long
- The following are automatic failures regardless of length:
  "be careful", "use caution", "stay safe", "be cautious when handling"

Award 0 IF:
- Safety info is generic and could apply to any repair
- Safety info is under 80 characters
- No specific hazard (electrical, gas, water shutoff, chemical) is named

## Q5 — tip_usefulness
Award 1 ONLY IF:
- Every tip provides non-obvious, task-specific advice
- Tips add information NOT already covered in the steps
- Tips address real failure modes, timing nuances, or common mistakes for this repair

Award 0 IF:
- Any tip merely restates a step already in the steps field
- Any tip offers generic encouragement ("take your time", "be patient")
- Any tip could apply to any home repair regardless of category

## Q6 — problem_answer_alignment
Award 1 ONLY IF:
- The answer directly resolves the specific symptom described in equipment_problem
- The answer is not a general maintenance guide when a specific fault was described
- Every major step connects back to diagnosing or fixing the stated problem

Award 0 IF:
- The answer addresses a related but different problem
- The answer is a general how-to when the problem is a specific symptom

## Q7 — appropriate_scope
Award 1 ONLY IF:
- The repair is within realistic DIY capability for a non-expert homeowner
- If professional help is genuinely required (gas lines, electrical panel, structural), 
  the answer explicitly says so rather than providing amateur instructions
- Complexity of the solution matches the complexity of the problem

Award 0 IF:
- The answer instructs a homeowner to perform a genuinely dangerous task without flagging it
- The solution requires trade-level skills presented as routine DIY

## Q8 — category_accuracy
Award 1 ONLY IF:
- The category field correctly matches the repair domain of the problem
- Use these domain definitions:
  - electrical_repair: wiring, fixtures, outlets, breakers, switches
  - plumbing_repair: pipes, drains, faucets, water supply/shutoff
  - appliance_repair: dishwashers, ovens, washers, dryers, refrigerators
  - general_home_repair: caulking, patching, painting, weatherproofing

Award 0 IF:
- The category is mismatched (e.g. plumbing problem tagged as electrical_repair)
- The category is too broad or too narrow for the stated problem

---

# SCORING RULES

- Each dimension: 1 = pass, 0 = fail
- quality_pass: true ONLY IF all 8 dimensions score 1
- overall_score: sum of all dimension scores / 8  →  range 0.0 to 1.0

After scoring, verify consistency:
- If step_actionability = 0, answer_coherence cannot be 1
- If safety_specificity = 0, flag it explicitly in reasoning
- If any score contradicts your reasoning trace, revise the score before outputting

---

# SAMPLE TO EVALUATE

{sample_json}

---

# OUTPUT FORMAT (JSON only, no markdown, no preamble)

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
  "reasoning": "Per-dimension explanation. For every 0, state the specific text that caused failure."
}
"""
