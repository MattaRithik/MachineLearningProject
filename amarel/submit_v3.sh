#!/bin/bash
# Version 3 full run on Amarel (run from the project folder on a login node).
#   bash amarel/submit_v3.sh full             show the complete plan; launches nothing
#   bash amarel/submit_v3.sh full --submit    submit the complete workflow
# Submitting again after a failure resumes: completed runs are verified and skipped.
# Optional: CMP_ENV (Python environment), CMP_DATA_DIR (Parquet files), CMP_V2_DIR (Version 2 results, read only), CMP_EPOCHS (default 100).
set -Eeuo pipefail
[ -f Cross_Market_Project.ipynb ] || { echo "ERROR: run this from the project folder." >&2; exit 1; }
usage="Usage: bash amarel/submit_v3.sh full [--submit]"
[ $# -le 2 ] && [ "${1:-}" = full ] && [[ "${2:-}" =~ ^(--submit)?$ ]] || { echo "$usage" >&2; exit 1; }
mode="${2:-}"

export CMP_ENV="${CMP_ENV:-$HOME/envs/cross-market}"
export CMP_DATA_DIR="${CMP_DATA_DIR:-$PWD/data}"
[ -f "$CMP_DATA_DIR/USA.parquet" ] || export CMP_DATA_DIR="$PWD/../MachineLearningProject/data"
export CMP_V2_DIR="${CMP_V2_DIR:-$PWD/../MachineLearningProject/results_v2}"
export CMP_EPOCHS="${CMP_EPOCHS:-100}"
[[ "$CMP_EPOCHS" =~ ^([1-9][0-9]?|100)$ ]] || { echo "ERROR: CMP_EPOCHS must be between 1 and 100." >&2; exit 1; }
results=results_v3_full
[ "$CMP_EPOCHS" = 100 ] || results="results_v3_full_${CMP_EPOCHS}epochs"
export CMP_LOG_DIR="$results/logs"

# Git state is resolved here, because compute nodes may have no git
CMP_GIT_COMMIT=$(git rev-parse --verify HEAD)
changes=$(git status --porcelain --untracked-files=no)
if [ -z "$changes" ]; then CMP_GIT_DIRTY=0; else CMP_GIT_DIRTY=1; fi
CMP_NOTEBOOK_SHA256=$(sha256sum Cross_Market_Project.ipynb | cut -d' ' -f1)
export CMP_GIT_COMMIT CMP_GIT_DIRTY CMP_NOTEBOOK_SHA256
echo "Commit $CMP_GIT_COMMIT (uncommitted changes: $CMP_GIT_DIRTY) | data $CMP_DATA_DIR | Version 2 $CMP_V2_DIR | results $results | epochs $CMP_EPOCHS"

problems=()
[ -x "$CMP_ENV/bin/python" ] || problems+=("no Python environment at $CMP_ENV (set CMP_ENV)")
for country in USA GBR JPN; do
    [ -f "$CMP_DATA_DIR/$country.parquet" ] || problems+=("$CMP_DATA_DIR/$country.parquet is missing (set CMP_DATA_DIR)")
done
[ -f "$CMP_V2_DIR/rank/predictions/GBR_train_1y_finetuned.parquet" ] || problems+=("no Version 2 forecasts in $CMP_V2_DIR (set CMP_V2_DIR)")
[ "$CMP_GIT_DIRTY" = 0 ] || problems+=("uncommitted changes")
stamp="notebook $CMP_NOTEBOOK_SHA256 epochs $CMP_EPOCHS"
if [ -e "$results" ] && [ "$(head -n 1 "$results/run.txt" 2>/dev/null || true)" != "$stamp" ]; then
    problems+=("$results holds another run; move it away first")
fi
for problem in ${problems[@]+"${problems[@]}"}; do echo "PROBLEM: $problem" >&2; done

if [ "$mode" = "--submit" ]; then
    [ ${#problems[@]} = 0 ] || { echo "ERROR: nothing was submitted." >&2; exit 1; }
    queued=$(squeue -h -u "$USER" -o %j) || { echo "ERROR: squeue failed; nothing was submitted." >&2; exit 1; }
    ! grep -q '^cmp3-' <<< "$queued" || { echo "ERROR: Version 3 jobs are already queued or running." >&2; exit 1; }
    mkdir -p "$results/logs"
    [ -f "$results/run.txt" ] || printf '%s\ncommit %s\n' "$stamp" "$CMP_GIT_COMMIT" > "$results/run.txt"
fi

CPU="--partition=main --cpus-per-task=8 --mem=48G --time=12:00:00"
GPU="--partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=1-00:00:00"
submitted=()
cancel_submitted() {  # a failed submission must not leave half a workflow in the queue
    [ ${#submitted[@]} = 0 ] || { echo "ERROR: submission failed; cancelling ${submitted[*]}." >&2; scancel "${submitted[@]}"; }
}
trap cancel_submitted ERR

submit() {  # submit <stage> <cpu|cuda> <array or -> <dependency or ->; sets `job`
    local options=(--parsable --export=ALL --job-name="cmp3-$1")
    local resources="$CPU" log="$CMP_LOG_DIR/%x_%j.out"
    [ "$2" = cpu ] || resources="$GPU"
    [ "$3" = - ] || { options+=(--array="$3"); log="$CMP_LOG_DIR/%x_%A_%a.out"; }
    [ "$4" = - ] || options+=(--dependency="afterok:$4" --kill-on-invalid-dep=yes)
    options+=($resources --output="$log")
    if [ "$mode" != "--submit" ]; then
        echo "  sbatch ${options[*]} amarel/job.sh $1 $2"
        job="<$1>"
        return
    fi
    job=$(sbatch "${options[@]}" amarel/job.sh "$1" "$2")
    job="${job%%;*}"
    [[ "$job" =~ ^[0-9]+$ ]]
    submitted+=("$job")
    echo "$1 $job" | tee -a "$results/run.txt"
}

# Array tasks 0 and 1 are the configurations control and relative
submit preflight cpu - -;                                        preflight=$job
submit baselines cpu - "$preflight";                             baselines=$job
submit pretraining cuda 0-1 "$preflight";                        pretraining=$job
submit local cuda 0-1 "$preflight";                              local_training=$job
submit transfer cuda 0-1 "$pretraining";                         transfer=$job
submit selection cpu - "$baselines:$local_training:$transfer";   selection=$job
submit predictions cuda - "$selection";                          predictions=$job
submit evaluation cpu - "$predictions"
trap - ERR
if [ "$mode" = "--submit" ]; then
    echo "Submitted. Job numbers are in $results/run.txt."
else
    echo "Nothing was submitted. Use: bash amarel/submit_v3.sh full --submit"
fi
