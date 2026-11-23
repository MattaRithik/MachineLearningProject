#!/bin/bash
# Submits the full research sequence with dependencies. Usage (from the project folder): bash amarel/submit_all.sh
set -euo pipefail
[ -f Cross_Market_Project.ipynb ] || { echo "ERROR: run this from the project folder." >&2; exit 1; }
for country in USA GBR JPN; do
    [ -f "data/$country.parquet" ] || { echo "ERROR: data/$country.parquet is missing." >&2; exit 1; }
done

submit() {  # submit <script> <experiment> [dependency job IDs joined by ':']
    local options=(--parsable --job-name="cmp-$2")
    [ -n "${3:-}" ] && options+=(--dependency="afterok:$3" --kill-on-invalid-dep=yes)
    local id
    id=$(sbatch "${options[@]}" "amarel/$1" "$2")
    echo "${id%%;*}"
}

baselines=$(submit cpu_job.slurm baselines)
local_only=$(submit gpu_job.slurm local)
pretrained=$(submit gpu_job.slurm pretrained)
# These share one results file, so they run one after another
direct=$(submit gpu_job.slurm direct "$pretrained")
finetuned=$(submit gpu_job.slurm finetuned "$direct")
joint=$(submit gpu_job.slurm joint "$finetuned")
localmatched=$(submit gpu_job.slurm localmatched "$joint")
predictions=$(submit gpu_job.slurm predictions "$baselines:$local_only:$localmatched")
evaluation=$(submit cpu_job.slurm evaluation "$predictions")

echo "baselines $baselines | local $local_only | pretrained $pretrained | direct $direct | finetuned $finetuned"
echo "joint $joint | localmatched $localmatched | predictions $predictions | evaluation $evaluation"
