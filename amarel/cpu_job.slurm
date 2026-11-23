#!/bin/bash
#SBATCH --job-name=cmp-cpu
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
# Usage (from the project folder): sbatch amarel/cpu_job.slurm <baselines|elastic_net|xgboost|evaluation|smoke|preflight>
bash "$SLURM_SUBMIT_DIR/amarel/run_experiment.sh" "${1:?Give an experiment name}" cpu
