# Cross-Market Transfer Learning for Stock Return Prediction

Can a model pretrained on U.S. stocks help predict next-month stock returns in the UK and Japan, especially with little local data? The
project uses Jensen–Kelly–Pedersen (JKP) data and cross-sectional Transformers, with Elastic Net and XGBoost as baselines.

Everything is in one notebook, [Cross_Market_Project.ipynb](Cross_Market_Project.ipynb), which explains the design. Training ends in December 2017, tuning uses
development folds before 2018, selection uses 2018–2019. The 2020–2025 test period has been examined in earlier versions and is exploratory.

| Version | Where |
|---|---|
| 1 | tag `v1.0` |
| 2 (24 variants) | tag `v2.0-experiment`; audit and reports on branch `v2-finalization` |
| 3 | branch `v3` (not trained yet; no improvement is claimed) |

## Data (not included)

Place `USA.parquet` (1,604,094 rows), `GBR.parquet` (534,295) and `JPN.parquet` (1,175,157) in `data/`. Source: WRDS `contrib.global_factor`, January 2000 –
December 2025, with `common = exch_main = primary_sec = obs_main = 1`; 165 columns. The data are licensed and must not be redistributed.
Git ignores `data/`, `results_v3/` and `logs_v3/`; the results contain stock-level forecasts and must stay private.

## Run in Jupyter

1. `pip install -r requirements.txt` (Python 3.12), then open the notebook.
2. **Run All with the defaults:** defines everything; reads no data and trains nothing.
3. Set one switch in Section 1 to `True`, restart the kernel and Run All, in this order. Stage 1: `RUN_DATA_CHECKS`, `RUN_SCREENING`,
   `RUN_FINALIST_SELECTION`. Stage 2, after review: `RUN_BASELINES`, `RUN_FINAL_TRAINING`, `RUN_PREDICTIONS`, `RUN_FINAL_EVALUATION`.
   A stage stops with a clear message if an earlier stage is missing.

For a short trial use `MAX_EPOCHS = 2` and `ONLY_CONFIGS = ["small"]`; its outputs go to `results_v3_2epochs/`. Screening and final training are meant for a GPU.

## Run on Amarel

```bash
git clone -b v3 https://github.com/MattaRithik/MachineLearningProject.git MachineLearningProject_v3
cd MachineLearningProject_v3
bash amarel/submit_v3.sh screening             # Stage 1 plan; launches nothing
bash amarel/submit_v3.sh screening --submit    # preflight -> screening (8 GPU tasks) -> finalist choice, then stops
```

Review `results_v3/tables/screening_leaderboard.csv` and `screening_runs.csv`. Stage 2 is never submitted by Stage 1:

```bash
bash amarel/submit_v3.sh final                 # Stage 2 plan; launches nothing
bash amarel/submit_v3.sh final --submit        # finalist training and baselines -> predictions -> evaluation
```

Set `CMP_ENV` if your environment is not `~/envs/cross-market`, and `CMP_DATA_DIR` if the data are neither in `data/` nor in
`../MachineLearningProject/data`.

If a job fails, read `logs_v3/`, then submit again: complete runs are verified and skipped. Runs saved with other settings raise an error and are never
overwritten; move `results_v3/` first. For a first
installation of the environment, see the README of `v1.0`.

## Outputs (`results_v3/`)

`tables/validation_leaderboard.csv` is the model-selection table. `tables/` holds aggregates only. `runs/` and `test_forecasts/` hold models, records and
stock-level forecasts. `selection.json` is locked before any test forecast.
