#!/bin/bash

# Complete 7-Phase Synthetic Data Pipeline Runner
# Usage: bash RUN_PIPELINE.sh [num_samples] [run_phases]
#   num_samples: Number of samples to generate (default: 50)
#   run_phases: Which phases to run (default: "all")
#              Options: "baseline" (1-4), "analysis" (5,7), "improve" (6), "all" (1-7)

set -e

NUM_SAMPLES="${1:-50}"
RUN_PHASES="${2:-all}"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Synthetic Data Pipeline Runner${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Samples: $NUM_SAMPLES"
echo "Phases: $RUN_PHASES"
echo ""

# Activate conda environment
echo -e "${YELLOW}Activating conda environment...${NC}"
source /home/kevin/anaconda3/etc/profile.d/conda.sh
conda activate ai_bootcamp_mini_project_01_py312

# Phase 01: Generation
if [[ "$RUN_PHASES" == "all" || "$RUN_PHASES" == "baseline" ]]; then
    echo -e "${GREEN}[Phase 01] Generating synthetic Q&A...${NC}"
    python -m src.Phase_01_generation.run --num-samples "$NUM_SAMPLES"
    echo -e "${GREEN}✅ Phase 01 Complete${NC}\n"
fi

# Phase 02: Structural Validation
if [[ "$RUN_PHASES" == "all" || "$RUN_PHASES" == "validation" ]]; then
    echo -e "${GREEN}[Phase 02] Validating structure...${NC}"
    python -m src.Phase_02_structural_validation.run
    echo -e "${GREEN}✅ Phase 02 Complete${NC}\n"
fi

# Phase 03: Failure Labeling
if [[ "$RUN_PHASES" == "all" || "$RUN_PHASES" == "failure" ]]; then
    echo -e "${GREEN}[Phase 03] Detecting failure modes...${NC}"
    python -m src.Phase_03_failure_labeling.run
    echo -e "${GREEN}✅ Phase 03 Complete${NC}\n"
fi

# Phase 04: Quality Evaluation
if [[ "$RUN_PHASES" == "all" || "$RUN_PHASES" == "quality" ]]; then
    echo -e "${GREEN}[Phase 04] Evaluating quality dimensions...${NC}"
    python -m src.Phase_04_quality_evaluation.run
    echo -e "${GREEN}✅ Phase 04 Complete${NC}\n"
fi

# Phase 05: Failure & Quality Analysis
if [[ "$RUN_PHASES" == "all" || "$RUN_PHASES" == "analysis" ]]; then
    echo -e "${GREEN}[Phase 05] Analyzing failure-quality correlations...${NC}"
    python -m src.Phase_05_failure_quality_analysis.run
    echo -e "${GREEN}✅ Phase 05 Complete${NC}\n"
fi


