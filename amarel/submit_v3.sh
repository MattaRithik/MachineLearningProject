#!/bin/bash
# Version 3 on Amarel (run from the project folder on a login node).
#   bash amarel/submit_v3.sh screening             Stage 1 plan: data checks, screening on the folds, finalist choice
#   bash amarel/submit_v3.sh screening --submit    submit Stage 1 only
#   bash amarel/submit_v3.sh final                 Stage 2 plan: finalist training, baselines, predictions, evaluation
#   bash amarel/submit_v3.sh final --submit        submit Stage 2, after Stage 1 has finished and been reviewed
# Without --submit nothing is launched. Completed runs are verified and skipped.
# Optional: CMP_ENV (Python environment), CMP_DATA_DIR (Parquet files), CMP_EPOCHS (default 100).
set -euo pipefail
[ -f Cross_Market_Project.ipynb ] || { echo "ERROR: run this from the project folder." >&2; exit 1; }
stage="${1:-}" mode="${2:-}"
usage="Usage: bash amarel/submit_v3.sh <screening|final> [--submit]"
[ $# -le 2 ] && [[ "$stage" =~ ^(screening|final)$ ]] && [[ "$mode" =~ ^(--submit)?$ ]] || { echo "$usage" >&2; exit 1; }

export CMP_ENV="${CMP_ENV:-$HOME/envs/cross-market}"
export CMP_DATA_DIR="${CMP_DATA_DIR:-$PWD/data}"
[ -f "$CMP_DATA_DIR/USA.parquet" ] || export CMP_DATA_DIR="$PWD/../MachineLearningProject/data"
export CMP_EPOCHS="${CMP_EPOCHS:-100}"
[ -x "$CMP_ENV/bin/python" ] || { echo "ERROR: no Python environment at $CMP_ENV." >&2; exit 1; }
for country in USA GBR JPN; do
    [ -f "$CMP_DATA_DIR/$country.parquet" ] || { echo "ERROR: $CMP_DATA_DIR/$country.parquet is missing." >&2; exit 1; }
done

# Git state is resolved here, because compute nodes may have no git
CMP_GIT_COMMIT=$(git rev-parse --verify HEAD)
changes=$(git status --porcelain --untracked-files=no)
if [ -z "$changes" ]; then CMP_GIT_DIRTY=0; else CMP_GIT_DIRTY=1; fi
CMP_NOTEBOOK_SHA256=$(sha256sum Cross_Market_Project.ipynb | cut -d' ' -f1)
export CMP_GIT_COMMIT CMP_GIT_DIRTY CMP_NOTEBOOK_SHA256
echo "Commit $CMP_GIT_COMMIT (uncommitted changes: $CMP_GIT_DIRTY) | data $CMP_DATA_DIR | epochs $CMP_EPOCHS"
if [ "$stage" = final ]; then
    results=results_v3
    [ "$CMP_EPOCHS" = 100 ] || results="results_v3_${CMP_EPOCHS}epochs"
    if [ -f "$results/finalists.json" ]; then
        count=$("$CMP_ENV/bin/python" -c 'import json, sys; print(len(json.load(open(sys.argv[1]))["finalists"]))' "$results/finalists.json")
        echo "Finalists from Stage 1: $(tr -d '\n' < "$results/finalists.json")"
    elif [ "$mode" = "--submit" ]; then
        echo "ERROR: $results/finalists.json is missing. Run and review Stage 1 first." >&2
        exit 1
    else
        count=3
        echo "NOTE: Stage 1 has not finished; the plan assumes 3 finalists and cannot be submitted yet."
    fi
fi
if [ "$mode" = "--submit" ]; then
    [ "$CMP_GIT_DIRTY" = 0 ] || { echo "ERROR: commit or undo the changes before submitting." >&2; exit 1; }
    queued=$(squeue -h -u "$USER" -o %j | grep '^cmp3-' || true)
    [ -z "$queued" ] || { echo "ERROR: Version 3 jobs are already queued or running." >&2; exit 1; }
    mkdir -p logs_v3
fi

CPU="--partition=main --cpus-per-task=8 --mem=48G --time=12:00:00"
GPU="--partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=48G --time=1-00:00:00"
submit() {  # submit <stage> <cpu|cuda> <array or -> <dependency or ->
    local options=(--parsable --export=ALL --job-name="cmp3-$1")
    local resources="$CPU" log="logs_v3/%x_%j.out"
    [ "$2" = cpu ] || resources="$GPU"
    [ "$3" = - ] || { options+=(--array="$3"); log="logs_v3/%x_%A_%a.out"; }
    [ "$4" = - ] || options+=(--dependency="afterok:$4" --kill-on-invalid-dep=yes)
    options+=($resources --output="$log")
    if [ "$mode" != "--submit" ]; then
        echo "  sbatch ${options[*]} amarel/job.sh $1 $2" >&2
        echo "<$1>"
        return
    fi
    local id
    id=$(sbatch "${options[@]}" amarel/job.sh "$1" "$2")
    echo "${id%%;*}"
}

if [ "$stage" = screening ]; then
    preflight=$(submit preflight cpu - -)
    configurations=$(submit screening cuda 0-7 "$preflight")
    selection=$(submit selection cpu - "$configurations")
    echo "preflight $preflight | screening $configurations | selection $selection"
else
    # The array has one task per finalist (2 or 3)
    training=$(submit training cuda "0-$((count - 1))" -)
    baselines=$(submit baselines cpu - -)
    predictions=$(submit predictions cuda - "$training:$baselines")
    evaluation=$(submit evaluation cpu - "$predictions")
    echo "training $training | baselines $baselines | predictions $predictions | evaluation $evaluation"
fi
[ "$mode" = "--submit" ] || echo "Nothing was submitted. Add --submit to submit this plan."
