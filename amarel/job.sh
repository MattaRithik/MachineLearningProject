#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
# Runs one stage of the notebook. Submitted by submit_v3.sh: job.sh <stage> <cpu|cuda>
set -euo pipefail
stage="$1"
export CMP_DEVICE="$2"
cd "${SLURM_SUBMIT_DIR:?Use amarel/submit_v3.sh}"
export CMP_PROJECT_ROOT="$PWD" PATH="$CMP_ENV/bin:$PATH" PYTHONNOUSERSITE=1
: "${CMP_GIT_COMMIT:?Submit with amarel/submit_v3.sh, which records the Git commit}"
current=$(sha256sum Cross_Market_Project.ipynb | cut -d' ' -f1)
if [ "$current" != "$CMP_NOTEBOOK_SHA256" ]; then
    echo "ERROR: the notebook changed after submission. Submit again." >&2
    exit 1
fi

# Every switch off, then only the requested stage on
export CMP_FULL_PIPELINE=0 CMP_DATA_CHECKS=0 CMP_BASELINES=0 CMP_PRETRAINING=0 CMP_LOCAL_TRAINING=0 CMP_TRANSFER_TRAINING=0
export CMP_SELECTION=0 CMP_PREDICTIONS=0 CMP_FINAL_EVALUATION=0 CMP_CONFIG=""
CONFIGS=(control relative)
case "$stage" in
    preflight)   export CMP_DATA_CHECKS=1 ;;
    baselines)   export CMP_BASELINES=1 ;;
    pretraining) export CMP_PRETRAINING=1 CMP_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}" ;;
    local)       export CMP_LOCAL_TRAINING=1 CMP_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}" ;;
    transfer)    export CMP_TRANSFER_TRAINING=1 CMP_CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}" ;;
    selection)   export CMP_SELECTION=1 ;;
    predictions) export CMP_PREDICTIONS=1 ;;
    evaluation)  export CMP_FINAL_EVALUATION=1 ;;
    *)           echo "ERROR: unknown stage '$stage'." >&2; exit 1 ;;
esac
echo "Stage $stage $CMP_CONFIG | commit $CMP_GIT_COMMIT (dirty $CMP_GIT_DIRTY) | job ${SLURM_JOB_ID:-none} | $(hostname) | $(date)"
python -c "import platform, torch; print('Python', platform.python_version(), '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
srun python -u amarel/run_notebook.py "$stage${CMP_CONFIG:+-$CMP_CONFIG}"
echo "Finished after $SECONDS seconds"
