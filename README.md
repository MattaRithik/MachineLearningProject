# Cross-Market Transfer Learning for Stock Return Prediction

Can a model pretrained on U.S. stocks help predict next-month stock returns in the UK and Japan, especially with little local data? The
project uses Jensen–Kelly–Pedersen (JKP) data and cross-sectional Transformers, with Elastic Net and XGBoost as baselines.

All research code is in [Cross_Market_Project.ipynb](Cross_Market_Project.ipynb), which explains the design. The three data files are its only
inputs. Training ends in December 2017; 2018–2019 chooses checkpoints and the final model. The 2020–2025 test period was examined in earlier
experiments, so its results are exploratory. The full run has not been executed yet; no result is claimed.

| Path | Content |
|---|---|
| `Cross_Market_Project.ipynb` | data checks, features, models, training, selection, forecasts, evaluation, figures |
| `amarel/` | `submit.sh` (plan and submit), `job.sh` (one SLURM job), `run_notebook.py` (runs the notebook without a browser) |
| `data/` | the Parquet files (not included) |
| `results/`, `logs/` | created by a run |

## Data (not included)

Place `USA.parquet` (1,604,094 rows), `GBR.parquet` (534,295) and `JPN.parquet` (1,175,157) in `data/`. Source: WRDS `contrib.global_factor`,
January 2000 – December 2025, with `common = exch_main = primary_sec = obs_main = 1`; 165 columns. The data are licensed and must not
be redistributed. Results contain stock-level forecasts and must stay private. Git ignores `data/`, `results/` and `logs/`.

The notebook verifies the checksums of the reference files. For another JKP extract with the same columns, set `REQUIRE_REFERENCE_DATA = False` in
the notebook or `CMP_REFERENCE_DATA=0` on Amarel; results will then differ.

## Environment

Python 3.12. On a Mac: `pip install -r requirements.txt`. On Amarel, create it once in an interactive job:

```bash
srun --partition=main --cpus-per-task=4 --mem=16G --time=02:00:00 --pty bash
conda create -y -p ~/envs/cross-market python=3.12      # for example /projects/community/miniconda3/bin/conda, or Miniforge
~/envs/cross-market/bin/pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
~/envs/cross-market/bin/pip install -r requirements.txt
exit
```

## Run in Jupyter

Run All with the defaults only defines everything: it reads no data and trains nothing. To run a stage, set its switch in Section 1 to `True`, restart the
kernel and Run All, in the order of the table at the top of the notebook (data checks first). `RUN_FULL_PIPELINE = True` runs everything. A fresh run needs an
empty `results/`; `RESUME = True` continues an interrupted one. Training is meant for a GPU.

## Run on Amarel

```bash
rsync -av --exclude .git --exclude results --exclude logs Cross_Market_Project NETID@amarel.rutgers.edu:    # from the Mac, with data/
cd ~/Cross_Market_Project                  # on an Amarel login node
bash amarel/submit.sh                      # checks, jobs and resources; submits nothing
bash amarel/submit.sh --submit             # fresh run
bash amarel/submit.sh --resume --submit    # continue the same run after a failure
```

`CMP_ENV` selects another environment (default: the active conda environment, else `~/envs/cross-market`) and `CMP_DATA_DIR` another data folder. A
failed job prevents all later ones; read `logs/`, then resume. Files of another run are never reused or overwritten.

## Outputs (`results/`)

| Path | Content |
|---|---|
| `run.json`, `settings.json`, `features.json` | identity and settings of the run |
| `selection.json` | model selected on validation, locked before any test forecast |
| `checkpoints/` | models and one record per run (settings, seed, history, time, software, checksums) |
| `predictions/` | validation and test forecasts (private) |
| `tables/` | aggregate tables and figures; start with `validation_leaderboard.csv`, `test_comparisons.csv`, `final_comparison.csv` |

Job logs and executed notebooks are in `logs/<run id>/` (private).
