# Cross-Market Transfer Learning for Stock Return Prediction

Can a model trained on U.S. stocks help predict next-month stock returns in the UK and Japan, especially with little
local data? The project compares Elastic Net, XGBoost and a cross-sectional Transformer (local-only, direct transfer,
fine-tuned, joint and ensemble) on the Jensen–Kelly–Pedersen (JKP) data.

## Structure

| Path | Purpose |
|---|---|
| `Cross_Market_Project.ipynb` | All research code; every training switch is off by default |
| `amarel/` | SLURM jobs that run the notebook on Rutgers Amarel |
| `data/` | Licensed input data, **not included** (see `data/README.md`) |
| `results/`, `logs/` | Models, predictions, evaluation and job logs, not included |
| `requirements.txt` | Pinned Python packages |

## 1. Clone on Amarel

Off campus, connect to the Rutgers VPN first. Replace `NETID` with your NetID.

```bash
ssh NETID@amarel.rutgers.edu
cd ~
git clone https://github.com/MattaRithik/MachineLearningProject.git
cd MachineLearningProject
```

## 2. Transfer the data privately

Run this **on your own computer**, from the folder that holds the three files. `-L` follows symbolic links.

```bash
rsync -avL --progress USA.parquet GBR.parquet JPN.parquet NETID@amarel.rutgers.edu:MachineLearningProject/data/
```

On Amarel, check them with `ls -lh ~/MachineLearningProject/data`. The notebook reads `data/` inside the project folder;
set `CMP_DATA_DIR` to use another folder. Git ignores the data, so `git pull` never changes or deletes it.

## 3. Create the Python environment

Amarel does not allow programs on login nodes, so work in an interactive job:

```bash
srun --partition=main --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
```

Python 3.12 is required. Use any conda available to you. The user guide lists a community Miniconda; confirm that it
still exists after the move to RHEL 9:

```bash
ls /projects/community/miniconda3/bin/conda
/projects/community/miniconda3/bin/conda create -y -p ~/envs/cross-market python=3.12
```

If it is missing, install [Miniforge](https://github.com/conda-forge/miniforge) in your home directory and use its
`conda`. Then install PyTorch with CUDA 12.6, followed by the other packages:

```bash
cd ~/MachineLearningProject
~/envs/cross-market/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
~/envs/cross-market/bin/pip install -r requirements.txt
exit
```

No CUDA module is needed, because the PyTorch package includes its CUDA libraries. If your environment is elsewhere, run
`export CMP_ENV=/path/to/env` before submitting jobs.

## 4. Verify the environment

```bash
srun --partition=gpu --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
    ~/envs/cross-market/bin/python amarel/verify_environment.py
```

## 5. Small verification jobs

Submit every job from the project folder.

```bash
sbatch --time=01:00:00 amarel/gpu_job.slurm preflight   # does the largest cross-section fit in GPU memory?
sbatch --time=01:00:00 amarel/gpu_job.slurm smoke       # short code checks; saves no models
```

A short rehearsal with 5 epochs is kept apart in `results_5epochs/`:

```bash
CMP_EPOCHS=5 sbatch amarel/gpu_job.slurm local
```

## 6. Individual experiments

| Order | Experiment | Command | Requires |
|---|---|---|---|
| 1 | Elastic Net and XGBoost | `sbatch amarel/cpu_job.slurm baselines` | – |
| 2 | Local Transformers | `sbatch amarel/gpu_job.slurm local` | – |
| 3 | U.S. pretraining | `sbatch amarel/gpu_job.slurm pretrained` | – |
| 4 | Direct transfer | `sbatch amarel/gpu_job.slurm direct` | 3 |
| 5 | Fine-tuning | `sbatch amarel/gpu_job.slurm finetuned` | 3 |
| 6 | Joint training | `sbatch amarel/gpu_job.slurm joint` | 3 |
| 7 | Matched local Transformers | `sbatch amarel/gpu_job.slurm localmatched` | 3 |
| 8 | Test predictions | `sbatch amarel/gpu_job.slurm predictions` | 1–7 |
| 9 | Final evaluation | `sbatch amarel/cpu_job.slurm evaluation` | 8 |

Run experiments 3–7 one at a time, because they share one results file. Never run the same experiment twice at once.
Resources can be changed at submission, for example `sbatch --mem=64G --time=2-00:00:00 amarel/gpu_job.slurm joint`.
Add `--constraint=...` or `--account=...` only if your allocation needs it.

## 7. Full sequence

```bash
bash amarel/submit_all.sh
```

Each step starts only if its prerequisites succeeded. For a 5-epoch rehearsal, use `CMP_EPOCHS=5 bash amarel/submit_all.sh`.

**Restarting.** Saved experiments are verified and skipped, and unfinished ones start again, so a failed or preempted job
can simply be resubmitted. Predictions are never overwritten automatically: to replace them, move
`results/test_predictions.parquet` first.

## 8. Monitor and retrieve

```bash
squeue -u $USER                                          # job status
tail -f logs/cmp-pretrained_JOBID.out                    # live progress
sacct -j JOBID --format=JobID,State,Elapsed,MaxRSS       # finished jobs
scancel JOBID                                            # cancel
```

On your own computer:

```bash
rsync -av NETID@amarel.rutgers.edu:MachineLearningProject/results/ results/
```

## 9. Update the code

```bash
cd ~/MachineLearningProject
git pull
```

Data, results and logs are ignored by Git and stay in place. Do not pull while jobs are waiting to start, because they
read the notebook when they begin. Clear notebook outputs that show stock-level data before committing.

## Notes

- GPU jobs are limited to 72 hours and one GPU per user at a time, and jobs may be preempted. See the
  [Amarel user guide](https://go.rutgers.edu/hpc-user-guide).
- The returns contain extreme values that are not clipped or removed; see the notebook.
- Nothing in this repository has been run on Amarel yet.
