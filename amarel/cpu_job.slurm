#!/bin/bash
#SBATCH --job-name=cmp2-cpu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs_v2/%x_%j.out
# Usage (from the project folder): sbatch amarel/cpu_job.slurm <baseline variant|evaluate|compare|smoke>
bash "$SLURM_SUBMIT_DIR/amarel/run_experiment.sh" "${1:?Give a step}" cpu
