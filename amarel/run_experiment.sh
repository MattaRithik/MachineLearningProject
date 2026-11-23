#!/bin/bash
# Runs one experiment inside a SLURM allocation. Usage: run_experiment.sh <experiment> <cpu|cuda>
set -euo pipefail
experiment="$1"
export CMP_DEVICE="$2"

cd "${SLURM_SUBMIT_DIR:?Submit this job with sbatch from the project folder}"
if [ ! -f Cross_Market_Project.ipynb ]; then
    echo "ERROR: submit from the project folder that contains Cross_Market_Project.ipynb." >&2
    exit 1
fi
source amarel/env.sh

# Start with everything off, then enable only the requested experiment
export CMP_PROJECT_ROOT="$PWD"
export CMP_RUN_MODE=off CMP_PREDICTIONS=0 CMP_EVALUATION=0 CMP_SMOKE_TESTS=0 CMP_PREFLIGHT=0
unset CMP_EXPERIMENTS
case "$experiment" in
    baselines)   export CMP_RUN_MODE=full_research CMP_EXPERIMENTS=elastic_net,xgboost ;;
    elastic_net|xgboost|local|pretrained|direct|finetuned|joint|localmatched)
                 export CMP_RUN_MODE=full_research CMP_EXPERIMENTS="$experiment" ;;
    predictions) export CMP_PREDICTIONS=1 ;;
    evaluation)  export CMP_EVALUATION=1 ;;
    smoke)       export CMP_SMOKE_TESTS=1 ;;
    preflight)   export CMP_PREFLIGHT=1 ;;
    *) echo "ERROR: unknown experiment '$experiment'." >&2; exit 1 ;;
esac

echo "Experiment: $experiment | device: $CMP_DEVICE | epochs: ${CMP_EPOCHS:-100} | job: ${SLURM_JOB_ID:-none} | host: $(hostname)"
srun python -u amarel/run_notebook.py "$experiment"
