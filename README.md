# Synthetic Data Pipeline: DIY Repair Q&A
**An AI systems project that generates, evaluates, debugs, and improves synthetic training data in production.**

> **Generate diverse synthetic data → validate structure → detect quality failures with LLM judges → analyze failure patterns → correct generation prompts → measure improvement. A real-world MLOps workflow applied to data generation.**

---

## What This Is

A **production-grade data generation pipeline** that demonstrates **debugging-first and evaluation-centric principles**. Rather than hoping generated data is good, every sample is scored, every failure is analyzed, and every prompt is iteratively improved based on quantified evidence.

This mirrors real MLOps teams who generate datasets, run evals, diagnose failures, and ship improvements—but applied to the **data generation pipeline itself**.

### Core Capability
Generate 1000s of high-quality DIY home repair Q&A pairs by:
- Building structured, domain-expert prompts
- Running rigorous quality judgment across 8 dimensions
- Detecting semantic failures a schema validator can't catch
- Identifying which failure modes correlate and why
- Iteratively correcting prompts based on failure patterns
- Validating judge calibration against benchmark datasets
- Comparing your output against professional standards

---

## How It Works

```
Phase 1: Generate          → LLM produces structured JSON samples
         ↓
Phase 2: Validate          → Pydantic schema checks, domain constraint checks
         ↓
Phase 3: Judge & Analyze   → LLM-as-Judge scores 6 failure modes, 8 quality dimensions
         ↓
Phase 4: Quality Eval      → Structured quality dimension scoring
         ↓
Phase 5: Failure Analysis  → Identify correlated failures and root causes
         ↓
Phase 6: Correct Prompts   → Re-engineer generation prompts based on findings
         ↓
Phase 7: Benchmark Test    → Validate judge is calibrated, compare against baselines
         ↓
        LOOP: Iterate until quality targets met
```

---

## The 7 Phases

### **Phase 1: Generation**
Creates structurally valid JSON samples from 5 domain-specific prompt templates:
- Appliance Repair, Plumbing Repair, Electrical Repair, HVAC Maintenance, General Home Repair
- Each template uses expert personas, safety constraints, and structured output formats
- Output: 100s of samples matching Pydantic schema

### **Phase 2: Structural Validation**
Filters generation by schema correctness AND domain logic:
- Validates JSON structure, required fields, type correctness
- Checks domain constraints: realistic tools, logical step flow, category alignment
- Identifies template artifacts and duplicates
- Output: Structurally sound samples ready for quality judgment

### **Phase 3: Failure Labeling & Pattern Analysis**
Deploys an independent LLM judge to detect semantic failures:
- **6 Failure Modes**: Incomplete answers, safety violations, unrealistic tools, overcomplicated solutions, missing context, poor tips
- **8 Quality Dimensions**: Answer coherence, step actionability, tool realism, safety specificity, tip usefulness, problem alignment, appropriate scope, category accuracy
- **Correlation Analysis**: Which failures appear together? Which repair categories fail most?
- Output: Structured judgments + heatmaps revealing systemic issues

### **Phase 4: Quality Dimension Scoring**
Quantifies how far generated data is from professional standards:
- Scores each sample on 8 independent quality dimensions (binary pass/fail)
- Identifies weakest dimensions (targets for prompt refinement)
- Baseline metrics for comparing against benchmarks
- Output: Quality statistics and per-dimension dashboards

### **Phase 5: Failure Analysis & Root Cause**
Synthesizes failure patterns to find actionable fixes:
- Correlations between failure modes (e.g., "missing_context ↔ incomplete_answer")
- Category-level breakdown (which repair types fail most?)
- Identifies whether issues are prompt-level or systematic
- Output: Diagnostic report pinpointing exactly what to fix

### **Phase 6: Prompt Correction**
Iteratively re-engineers generation prompts:
- Add few-shot examples of good outputs
- Tighten safety constraints and specificity requirements
- Inject failure patterns discovered in Phase 3/5
- Re-generate failed samples with corrected prompts
- Output: Improved samples ready for re-evaluation

### **Phase 7: Benchmark Comparison & Judge Calibration**
Validates the entire system and measures production readiness:
- Runs judge against 50+ benchmark samples from professional dataset
- Verifies judge calibration (≥95% pass rate = trustworthy)
- Compares your generated quality vs. benchmark standards
- Identifies quality gaps and production readiness
- Output: Calibration report + benchmark comparison charts

---

## Why This Matters

**Debugging-First Mindset**: Rather than blindly re-generating, you instrument failures, measure them, and fix the root cause.

**Evaluation-Centric**: Every claim is backed by structured metrics. Quality isn't subjective—it's scored on 8 dimensions against benchmarks.

**Production-Ready**: By Phase 7, you know:
- ✅ Your judge is trustworthy (calibrated against baselines)
- ✅ Your data quality gap vs. professionals (measured quantitatively)
- ✅ Exactly which prompts need work (pinpointed by failure analysis)
- ✅ Your iteration strategy (data-driven, not guesswork)

This is **how real MLOps teams operate**. Generate → Evaluate → Diagnose → Iterate.

---

## Key Results

**Sample Benchmark Run (50 items)**:
```json
{
  "total_samples": 50,
  "quality_pass_rate": 86%,
  "dimension_scores": {
    "answer_coherence": 100%,
    "step_actionability": 98%,
    "safety_specificity": 94%,
    "tip_usefulness": 92%
  },
  "judge_calibration": "well_calibrated"
}
```

**Pattern Example**:
```
Failure Analysis reveals:
  - Missing context ↔ Incomplete answer (0.72 correlation)
  → Solution: Add explicit safety checklists to prompts
  
  - Electrical repair has 3× more failures
  → Solution: Tighten electrical safety template
```

---

## Tech Stack

- **LLM Inference**: OpenRouter (unified API for Claude, GPT-4, etc.)
- **Observability**: Braintrust (tracing, cost tracking, quality metrics)
- **Validation**: Pydantic (schema enforcement), custom domain validators
- **Analysis**: Pandas, NumPy, Matplotlib (failure heatmaps, correlations, dashboards)
- **Orchestration**: Custom Python pipeline runner with phase tracking

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
export OPENROUTER_API_KEY=your_key
export BRAINTRUST_API_KEY=your_key

# Run the full pipeline
python -m src.pipeline_core.orchestrator

# Or run individual phases
python -m src.Phase_01_generation.run
python -m src.Phase_02_structural_validation.run
python -m src.Phase_03_failure_labeling.run
# ... and so on
```

---

## Project Structure

```
src/
├── Phase_01_generation/          # LLM sample generation
├── Phase_02_structural_validation/  # Schema + domain validation
├── Phase_03_failure_labeling/    # LLM judge, failure detection, pattern analysis
├── Phase_04_quality_evaluation/  # Quality dimension scoring
├── Phase_05_failure_analysis/    # Root cause analysis
├── Phase_06_prompt_correction/   # Iterative prompt refinement
├── Phase_07_benchmark_comparison/  # Judge calibration & production readiness
└── pipeline_core/                # Shared orchestration, I/O, utilities

data/
├── phase_01/                     # Generated samples
├── phase_02/                     # Validated samples
├── phase_03/                     # Failure judgments + analysis
├── phase_04/                     # Quality scores
├── phase_05/                     # Root cause reports
├── phase_06/                     # Corrected prompts + re-generated
└── phase_07/                     # Benchmark comparison results
```

---

## What You Learn From This

1. **Prompt Engineering at Scale**: How to design templates that generate diverse, domain-specific, safe outputs
2. **LLM Evaluation**: How to build judges, calibrate them, and measure reliability
3. **Failure Analysis**: How to detect patterns in generated data and pinpoint root causes
4. **Iterative Improvement**: How to close feedback loops: generate → evaluate → diagnose → fix
5. **MLOps Thinking**: How data teams actually operate in production (not just training)

---

## Next Steps

- Run Phase 1-7 on 100+ samples
- Integrate with downstream training pipeline (LLM fine-tuning validation)
- Set up continuous evaluation (re-score on new LLM checkpoints)
- Extend to other domains (software debugging, medical Q&A, etc.)

---

**Built with a real-world MLOps mindset: measure everything, debug systematically, improve iteratively.**
