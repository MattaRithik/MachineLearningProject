#!/bin/bash
# Submits the read-only audit of completed Version 2 results (run from the project folder, in a clone of v2-finalization).
#   bash amarel/submit_audit.sh              show the plan
#   bash amarel/submit_audit.sh --submit     submit one CPU job; it trains nothing
# CMP_V2_RESULTS: completed results, read only (default: ../MachineLearningProject_v2/results_v2)
set -euo pipefail
[ -f Cross_Market_Project.ipynb ] || { echo "ERROR: run this from the project folder." >&2; exit 1; }
source amarel/env.sh
source amarel/git_state.sh
resolve_git_state
export CMP_V2_RESULTS="${CMP_V2_RESULTS:-$PWD/../MachineLearningProject_v2/results_v2}"
[ -f "$CMP_V2_RESULTS/comparison/tests.csv" ] || { echo "ERROR: no completed results in $CMP_V2_RESULTS" >&2; exit 1; }
export CMP_AUDIT_DIR="${CMP_AUDIT_DIR:-$PWD/audit_v2}"
[ "$CMP_GIT_DIRTY" = 0 ] || { echo "ERROR: commit or discard local changes first." >&2; exit 1; }
[ ! -e "$CMP_AUDIT_DIR/provenance.json" ] || { echo "ERROR: $CMP_AUDIT_DIR holds a finished audit; move it first." >&2; exit 1; }
echo "Commit: $CMP_GIT_COMMIT | results (read only): $CMP_V2_RESULTS | output: $CMP_AUDIT_DIR"
if [ "${1:-}" != "--submit" ]; then
    echo "PLAN: sbatch --export=ALL --job-name=cmp2-audit --time=08:00:00 amarel/cpu_job.slurm audit"
    echo "Nothing was submitted. Add --submit to submit this plan."
    exit 0
fi
mkdir -p logs_v2
sbatch --export=ALL --job-name=cmp2-audit --time=08:00:00 amarel/cpu_job.slurm audit
