#!/bin/bash
# Complete research run on Amarel (run from the project folder on a login node).
#   bash amarel/submit.sh                     checks, jobs and resources; submits nothing
#   bash amarel/submit.sh --submit            start a fresh run; results/ must not exist or be empty
#   bash amarel/submit.sh --resume            checks and plan for continuing the run in results/
#   bash amarel/submit.sh --resume --submit   continue that run; its complete parts are verified and skipped
# Optional: CMP_ENV (Python environment), CMP_DATA_DIR (Parquet files), CMP_EPOCHS (default 100), CMP_REFERENCE_DATA=0 (accept other
# JKP extracts), CMP_CPU_PARTITION (default main), CMP_GPU_PARTITION (default gpu).
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
submit_jobs=0 resume=0
for argument in "$@"; do
    case "$argument" in
        --submit) submit_jobs=1 ;;
        --resume) resume=1 ;;
        *) echo "Usage: bash amarel/submit.sh [--resume] [--submit]" >&2; exit 1 ;;
    esac
done

export CMP_ENV="${CMP_ENV:-${CONDA_PREFIX:-$HOME/envs/cross-market}}" CMP_EPOCHS="${CMP_EPOCHS:-100}"
export CMP_DATA_DIR="${CMP_DATA_DIR:-$PWD/data}"
CPU_PARTITION="${CMP_CPU_PARTITION:-main}" GPU_PARTITION="${CMP_GPU_PARTITION:-gpu}"
problems=()
[[ "$CMP_EPOCHS" =~ ^([1-9][0-9]?|100)$ ]] || problems+=("CMP_EPOCHS must be between 1 and 100")
for country in USA GBR JPN; do
    [ -f "$CMP_DATA_DIR/$country.parquet" ] || problems+=("$CMP_DATA_DIR/$country.parquet is missing (copy it to data/ or set CMP_DATA_DIR)")
done

# Installed packages must match requirements.txt (read without importing them)
python="$CMP_ENV/bin/python"
if [ ! -x "$python" ]; then
    problems+=("no Python environment at $CMP_ENV (see README.md, or set CMP_ENV)")
elif ! packages=$("$python" - 2>&1 <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version
if sys.version_info[:2] != (3, 12):
    sys.exit(f"Python 3.12 is required, found {sys.version.split()[0]}")
for line in open("requirements.txt"):
    if "==" in line and not line.startswith("#"):
        name, wanted = line.strip().split("==")
        try:
            found = version(name)
        except PackageNotFoundError:
            sys.exit(f"{name} is not installed")
        if found.split("+")[0] != wanted:
            sys.exit(f"{name} {wanted} is required, found {found}")
        if name == "torch" and "+cpu" in found:
            sys.exit("PyTorch was installed without CUDA")
PY
); then
    problems+=("environment $CMP_ENV: ${packages##*$'\n'}")
fi

# SLURM partitions; every GPU job also stops if CUDA is not available
for partition in "$CPU_PARTITION" "$GPU_PARTITION"; do
    if ! nodes=$(sinfo -h -p "$partition" -o "%G" 2>/dev/null); then
        problems+=("sinfo is not available; run this on an Amarel login node")
        break
    fi
    [ -n "$nodes" ] || problems+=("partition $partition was not found")
    if [ "$partition" = "$GPU_PARTITION" ] && [[ "$nodes" != *gpu* ]]; then problems+=("partition $partition has no GPUs"); fi
done

# The identity of a run: a fresh run never reads files of an earlier one
CMP_NOTEBOOK_SHA256=$( (sha256sum Cross_Market_Project.ipynb || shasum -a 256 Cross_Market_Project.ipynb) 2>/dev/null | cut -d' ' -f1)
CMP_GIT_COMMIT=$(git rev-parse --verify HEAD 2>/dev/null || echo "not recorded")  # Git is optional and not used by the jobs
CMP_GIT_DIRTY=$([ -z "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && echo 0 || echo 1)
field() { sed -n "s/.*\"$1\": \"\{0,1\}\([^\",}]*\).*/\1/p" results/run.json 2>/dev/null || true; }
if [ "$resume" = 1 ]; then
    CMP_RUN_ID=$(field run_id)
    [ -n "$CMP_RUN_ID" ] || problems+=("results/run.json was not found; there is nothing to resume")
    [ "$(field notebook_sha256)" = "$CMP_NOTEBOOK_SHA256" ] || problems+=("results/ was produced by another version of the notebook; it cannot be resumed")
    [ "$(field max_epochs)" = "$CMP_EPOCHS" ] || problems+=("results/ was produced with another CMP_EPOCHS")
    [ ! -f results/run.json ] || export CMP_REFERENCE_DATA="$(field reference_data)"
else
    CMP_RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
    [ -z "$(ls -A results 2>/dev/null)" ] || problems+=("results/ is not empty; move it away for a fresh run, or use --resume")
fi
export CMP_RUN_ID CMP_NOTEBOOK_SHA256 CMP_GIT_COMMIT CMP_GIT_DIRTY CMP_LOG_DIR="logs/$CMP_RUN_ID" CMP_REFERENCE_DATA="${CMP_REFERENCE_DATA:-1}"

echo "Run $CMP_RUN_ID ($([ "$resume" = 1 ] && echo resume || echo fresh)) | epochs $CMP_EPOCHS | data $CMP_DATA_DIR | environment $CMP_ENV"
for problem in ${problems[@]+"${problems[@]}"}; do echo "PROBLEM: $problem" >&2; done
[ ${#problems[@]} != 0 ] || echo "All checks passed."

if [ "$submit_jobs" = 1 ]; then
    [ ${#problems[@]} = 0 ] || { echo "ERROR: nothing was submitted." >&2; exit 1; }
    queued=$(squeue -h -u "$USER" -o %j) || { echo "ERROR: squeue failed; nothing was submitted." >&2; exit 1; }
    ! grep -q '^cmp-' <<< "$queued" || { echo "ERROR: jobs of this project are already queued or running." >&2; exit 1; }
    mkdir -p results "$CMP_LOG_DIR"
    [ "$resume" = 1 ] || printf '{\n "run_id": "%s",\n "max_epochs": %s,\n "reference_data": %s,\n "notebook_sha256": "%s",\n "git_commit": "%s"\n}\n' \
        "$CMP_RUN_ID" "$CMP_EPOCHS" "$CMP_REFERENCE_DATA" "$CMP_NOTEBOOK_SHA256" "$CMP_GIT_COMMIT" > results/run.json
fi

CPU="--partition=$CPU_PARTITION --cpus-per-task=8 --mem=48G --time=12:00:00"
GPU="--partition=$GPU_PARTITION --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=1-00:00:00"
submitted=()
cancel_submitted() {  # a failed submission must not leave half a workflow in the queue
    [ ${#submitted[@]} = 0 ] || { echo "ERROR: submission failed; cancelling ${submitted[*]}. Retry with --resume --submit." >&2; scancel "${submitted[@]}"; }
}
trap cancel_submitted ERR

add_job() {  # add_job <stage> <cpu|cuda> <array or -> <dependency or ->; sets `job`
    local options=(--parsable --export=ALL --job-name="cmp-$1")
    local resources="$CPU" log="$CMP_LOG_DIR/%x_%j.out"
    [ "$2" = cpu ] || resources="$GPU"
    [ "$3" = - ] || { options+=(--array="$3"); log="$CMP_LOG_DIR/%x_%A_%a.out"; }
    [ "$4" = - ] || options+=(--dependency="afterok:$4" --kill-on-invalid-dep=yes)
    options+=($resources --output="$log")
    if [ "$submit_jobs" = 0 ]; then
        echo "  sbatch ${options[*]} amarel/job.sh $1 $2"
        job="<$1>"
        return
    fi
    job=$(sbatch "${options[@]}" amarel/job.sh "$1" "$2")
    job="${job%%;*}"
    [[ "$job" =~ ^[0-9]+$ ]]
    submitted+=("$job")
    echo "$1 $job" | tee -a "$CMP_LOG_DIR/jobs.txt"
}

# Array tasks 0 and 1 are the configurations control and relative
add_job preflight cpu - -;                                        preflight=$job
add_job baselines cpu - "$preflight";                             baselines=$job
add_job pretraining cuda 0-1 "$preflight";                        pretraining=$job
add_job local cuda 0-1 "$preflight";                              local_training=$job
add_job transfer cuda 0-1 "$pretraining";                         transfer=$job
add_job selection cpu - "$baselines:$local_training:$transfer";   selection=$job
add_job predictions cuda - "$selection";                          predictions=$job
add_job evaluation cpu - "$predictions"
trap - ERR
if [ "$submit_jobs" = 1 ]; then
    echo "Submitted. Job numbers are in $CMP_LOG_DIR/jobs.txt."
else
    echo "Nothing was submitted. Add --submit to start."
fi
