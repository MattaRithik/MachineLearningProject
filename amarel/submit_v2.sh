#!/bin/bash
# Submits Version 2 (run from the project folder). Nothing is submitted without --submit.
#   bash amarel/submit_v2.sh all                  show the full plan
#   bash amarel/submit_v2.sh all --submit         submit every variant, then evaluation and comparison
#   bash amarel/submit_v2.sh robust --submit      submit one variant (training and forecasts)
#   bash amarel/submit_v2.sh evaluate --submit    evaluate available variants, then compare
set -euo pipefail
[ -f Cross_Market_Project.ipynb ] || { echo "ERROR: run this from the project folder." >&2; exit 1; }
source amarel/env.sh
source amarel/variants.sh
target="${1:?Usage: submit_v2.sh <all|evaluate|variant> [--submit]}"
mode="${2:-}"
for country in USA GBR JPN; do
    [ -f "$CMP_DATA_DIR/$country.parquet" ] || { echo "ERROR: $CMP_DATA_DIR/$country.parquet is missing." >&2; exit 1; }
done
mkdir -p logs_v2

submit() {  # submit <script> <step> [dependency job IDs joined by ':']
    local name="cmp2-$2"
    if [ "$mode" != "--submit" ]; then
        echo "PLAN" >&2
        echo "  sbatch --job-name=$name ${3:+--dependency=afterok:$3 }amarel/$1 $2" >&2
        echo "<$2>"
        return
    fi
    if [ -n "$(squeue -h -u "$USER" -n "$name" -o %i)" ]; then
        echo "ERROR: a job named $name is already queued or running; two jobs must not share one variant." >&2
        exit 1
    fi
    local options=(--parsable --job-name="$name")
    [ -n "${3:-}" ] && options+=(--dependency="afterok:$3" --kill-on-invalid-dep=yes)
    local id
    id=$(sbatch "${options[@]}" "amarel/$1" "$2")
    echo "${id%%;*}"
}
submit_variant() {
    if is_baseline "$1"; then submit cpu_job.slurm "$1"; else submit gpu_job.slurm "$1"; fi
}

case "$target" in
    all)
        jobs=()
        for variant in "${BASELINE_VARIANTS[@]}" "${GPU_VARIANTS[@]}"; do
            id=$(submit_variant "$variant")
            jobs+=("$id")
            echo "$variant $id"
        done
        dependency=$(IFS=:; echo "${jobs[*]}")
        evaluation=$(submit cpu_job.slurm evaluate "$dependency")
        comparison=$(submit cpu_job.slurm compare "$evaluation")
        echo "evaluate $evaluation"
        echo "compare $comparison"
        ;;
    evaluate)
        evaluation=$(submit cpu_job.slurm evaluate)
        comparison=$(submit cpu_job.slurm compare "$evaluation")
        echo "evaluate $evaluation"
        echo "compare $comparison"
        ;;
    *)
        is_variant "$target" || { echo "ERROR: unknown variant '$target'. Choose: ${ALL_VARIANTS[*]}" >&2; exit 1; }
        id=$(submit_variant "$target")
        echo "$target $id"
        ;;
esac
[ "$mode" = "--submit" ] || echo "Nothing was submitted. Add --submit to submit this plan."
