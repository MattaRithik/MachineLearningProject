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
export CMP_DATA_CHECKS=0 CMP_BASELINES=0 CMP_SCREENING=0 CMP_FINALIST_SELECTION=0 CMP_FINAL_TRAINING=0 CMP_PREDICTIONS=0 CMP_FINAL_EVALUATION=0
unset CMP_CONFIGS CMP_FINALIST
CONFIGS=(control relative hybrid stable small gated filtered combined)
case "$stage" in
    preflight)   export CMP_DATA_CHECKS=1 ;;
    baselines)   export CMP_BASELINES=1 ;;
    screening)   export CMP_SCREENING=1 CMP_CONFIGS="${CONFIGS[$SLURM_ARRAY_TASK_ID]}" ;;
    selection)   export CMP_FINALIST_SELECTION=1 ;;
    training)    export CMP_FINAL_TRAINING=1 CMP_FINALIST="$SLURM_ARRAY_TASK_ID" ;;
    predictions) export CMP_PREDICTIONS=1 ;;
    evaluation)  export CMP_FINAL_EVALUATION=1 ;;
    *)           echo "ERROR: unknown stage '$stage'." >&2; exit 1 ;;
esac
echo "Stage $stage ${CMP_CONFIGS:-}${CMP_FINALIST:-} | commit $CMP_GIT_COMMIT (dirty $CMP_GIT_DIRTY) | job ${SLURM_JOB_ID:-none} | $(hostname)"
python -c "import platform, torch; print('Python', platform.python_version(), '| torch', torch.__version__, '| CUDA', torch.cuda.is_available())"
srun python -u amarel/run_notebook.py "$stage${CMP_CONFIGS:+-$CMP_CONFIGS}${CMP_FINALIST:+-$CMP_FINALIST}"
