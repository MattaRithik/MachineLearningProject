#!/bin/bash
#SBATCH --job-name=cmp-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x_%j.out
# Usage (from the project folder):
#   sbatch amarel/gpu_job.slurm <local|pretrained|direct|finetuned|joint|localmatched|predictions|smoke|preflight>
bash "$SLURM_SUBMIT_DIR/amarel/run_experiment.sh" "${1:?Give an experiment name}" cuda
