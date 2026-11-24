# Cross-Market Transfer Learning for Stock Return Prediction

Can a model pretrained on U.S. stocks help predict next-month stock returns in the UK and Japan, especially with little local data? The
project uses Jensen–Kelly–Pedersen (JKP) data and cross-sectional Transformers, with Elastic Net and XGBoost as baselines.

Everything is in one notebook, [Cross_Market_Project.ipynb](Cross_Market_Project.ipynb), which explains the design. Training ends in December 2017; 2018–2019 chooses
checkpoints and the final model. The 2020–2025 test period has been examined in earlier versions and is exploratory.

| Version | Where |
|---|---|
| 1 | tag `v1.0` |
| 2 (24 variants) | tag `v2.0-experiment`; audit and reports on branch `v2-finalization` |
| 3 | branch `v3`: configurations frozen after Stage 1; full run not executed yet, no improvement is claimed |

## Data (not included)

Place `USA.parquet` (1,604,094 rows), `GBR.parquet` (534,295) and `JPN.parquet` (1,175,157) in `data/`. Source: WRDS `contrib.global_factor`, January 2000 –
December 2025, with `common = exch_main = primary_sec = obs_main = 1`; 165 columns. The data are licensed and must not be redistributed.
Git ignores `data/` and `results_*/`; the results contain stock-level forecasts and must stay private.

The full run also reads the saved Version 2 `rank` forecasts (`results_v2/rank/predictions/`) for the comparison. They are verified and never changed.

## Run in Jupyter

1. `pip install -r requirements.txt` (Python 3.12), then open the notebook.
2. **Run All with the defaults:** defines everything; reads no data and trains nothing.
3. Set one stage switch in Section 1 to `True`, restart the kernel and Run All, in the order of the table at the top of the notebook, or use
   `RUN_FULL_PIPELINE = True` for everything. A stage stops with a clear message if an earlier stage is missing. Set `CMP_V2_DIR` if the
   Version 2 results are not in `../MachineLearningProject/results_v2`.

For a short trial use `MAX_EPOCHS = 2`; its outputs go to `results_v3_full_2epochs/`. Training is meant for a GPU.

## Run on Amarel

```bash
bash amarel/submit_v3.sh full             # complete plan; launches nothing
bash amarel/submit_v3.sh full --submit    # data checks -> baselines, pretraining, local -> transfer -> selection -> test forecasts -> evaluation
```

Set `CMP_ENV` if your environment is not `~/envs/cross-market`, `CMP_DATA_DIR` if the data are neither in `data/` nor in `../MachineLearningProject/data`, and
`CMP_V2_DIR` as above. A failed job prevents all later ones. Read its log, then submit again: complete runs are verified and
skipped, and runs saved with other settings raise an error and are never overwritten. For a first installation, see the README of `v1.0`.

## Outputs (`results_v3_full/`)

| Folder | Content |
|---|---|
| `checkpoints/` | models and one JSON record per run (settings, seed, history, time, Git revision, software, checksums) |
| `predictions/validation/`, `predictions/test/` | stock-level forecasts (private) |
| `tables/` | aggregate tables and figures; start with `validation_leaderboard.csv`, `test_comparisons.csv` and `v2_v3_comparison.csv` |
| `logs/` | job logs and executed notebooks (private) |

`selection.json` is locked before any test forecast.
