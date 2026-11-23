#!/bin/bash
# Runs one step inside a SLURM allocation. Usage: run_experiment.sh <step> <cpu|cuda>
#   smoke | preflight          code checks (save no models)
#   <variant>                  train the variant and generate its forecasts
#   train:<v> | predict:<v> | evaluate:<v>   one stage of a variant (or of several, comma-separated)
#   evaluate                   evaluate every available variant
#   compare                    ablation tests and comparison with Version 1
set -euo pipefail
step="$1"
export CMP_DEVICE="$2"

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project folder}"
if [ ! -f Cross_Market_Project.ipynb ]; then
    echo "ERROR: submit from the project folder that contains Cross_Market_Project.ipynb." >&2
    exit 1
fi
source amarel/env.sh
source amarel/variants.sh

# Everything off; Version 1 always stays off in this branch
export CMP_PROJECT_ROOT="$PWD"
export CMP_RUN_MODE=off CMP_PREDICTIONS=0 CMP_EVALUATION=0 CMP_SMOKE_TESTS=0 CMP_PREFLIGHT=0 CMP_V2=0
unset CMP_EXPERIMENTS CMP_V2_STAGES CMP_V2_VARIANTS
commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)
[ -z "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] || commit="$commit-modified"
export CMP_GIT_COMMIT="$commit"

all=$(IFS=,; echo "${ALL_VARIANTS[*]}")
case "$step" in
    smoke)     export CMP_SMOKE_TESTS=1 ;;
    preflight) export CMP_PREFLIGHT=1 ;;
    evaluate)  export CMP_V2=1 CMP_V2_STAGES=evaluate CMP_V2_VARIANTS="$all" ;;
    compare)   export CMP_V2=1 CMP_V2_STAGES=compare CMP_V2_VARIANTS="$all" ;;
    train:*|predict:*|evaluate:*)
               export CMP_V2=1 CMP_V2_STAGES="${step%%:*}" CMP_V2_VARIANTS="${step#*:}" ;;
    *)         is_variant "$step" || { echo "ERROR: unknown step '$step'." >&2; exit 1; }
               export CMP_V2=1 CMP_V2_STAGES=train,predict CMP_V2_VARIANTS="$step" ;;
esac

echo "Step: $step | device: $CMP_DEVICE | epochs: ${CMP_EPOCHS:-100} | commit: $commit | job: ${SLURM_JOB_ID:-none} | host: $(hostname)"
echo "Data: $CMP_DATA_DIR | Version 1 results: $CMP_V1_RESULTS"
srun python -u amarel/run_notebook.py "${step//[:,]/-}"
