# Cross-Market Transfer Learning for Stock Return Prediction

Can a model trained on U.S. stocks help predict next-month stock returns in the UK and Japan, especially with little
local data? The project compares Elastic Net, XGBoost and cross-sectional neural models (local-only, direct transfer,
fine-tuned, joint and ensemble) on the Jensen–Kelly–Pedersen (JKP) data.

| | Version 1 | Version 2 |
|---|---|---|
| Code | tag `v1.0` (branch `main`) | tag `v2.0-experiment` (branch `v2`); audit on `v2-finalization` |
| Notebook sections | 1–40 (frozen, shared code) | 41–51; audit 52–60 |
| Outputs | `results/`, `logs/` | `results_v2/<variant>/`, `logs_v2/` |
| Status | full run completed on Amarel | training, evaluation and comparison of all 24 variants completed on Amarel |

**Version 2 is an exploratory follow-up designed after the Version 1 test results were seen.** It reuses the same test
period (January 2020 – December 2025), which is therefore not an untouched holdout. Training ends in December 2017, and
all selection uses 2018–2019. No test month is changed or removed.

In this branch the Version 1 switches raise an error, so Version 1 results cannot be changed. To rerun Version 1, check
out `v1.0` in a separate folder.

## Results and audit

Read [FINAL_RESULTS.md](FINAL_RESULTS.md) (provisional), [MODEL_SELECTION.md](MODEL_SELECTION.md), [V2_AUDIT_REPORT.md](V2_AUDIT_REPORT.md),
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) and [FINAL_CHANGELOG.md](FINAL_CHANGELOG.md). The results of `v2` are frozen; the audit on this branch reads them
and writes only to `audit_v2/`. It trains nothing and has not yet been run on the full data.

```bash
ssh NETID@amarel.rutgers.edu
cd ~ && git clone -b v2-finalization https://github.com/MattaRithik/MachineLearningProject.git MachineLearningProject_audit
cd MachineLearningProject_audit
bash amarel/submit_audit.sh             # show the plan (reads ../MachineLearningProject_v2/results_v2)
bash amarel/submit_audit.sh --submit    # one CPU job
```

Set `CMP_V2_RESULTS`, `CMP_DATA_DIR` or `CMP_ENV` if your folders differ. `audit_v2/private/` contains licensed stock-level records: keep it private.
Synthetic tests (no data): `python tests/test_audit.py`.

## Version 2 experiments

Every variant changes one setting of its reference (Section 41 of the notebook checks this), with the same seeds, search
size and architecture size. `combined` is the only variant with several changes.

| Variant | Reference | Change | Question |
|---|---|---|---|
| `control` | – | Version 1 Transformer and loss, small search | Baseline of the ablations |
| `huber` | `control` | Huber loss | Robust loss |
| `clip` | `control` | targets clipped at training quantiles | Robust targets |
| `clip_month` | `control` | targets clipped within each training month | Robust targets |
| `robust` | `huber` | adds `clip` | Base of the variants below |
| `select_clip`, `select_ic` | `robust` | select by clipped MSE or rank IC | Selection rule |
| `rank` | `robust` | ranking objective | Ranking-aware loss |
| `missing` | `robust` | missing-value indicators | Features |
| `mlp`, `gated` | `robust` | stockwise MLP; MLP with gated attention | Architecture |
| `dropout`, `gradclip`, `schedule`, `earlystop` | `robust` | one each | Regularization |
| `joint50`, `joint75` | `robust` | 50% or 75% local months (U.S./local 50/50 and 25/75) | Joint sampling |
| `combined` | `robust` | seven changes, fixed in advance | Joint effect |
| `b_control` | – | Version 1 baselines | Baseline control |
| `b_clip` | `b_control` | clipped training targets | Baselines |
| `b_grid` | `b_clip` | data-scaled Elastic Net, regularized XGBoost search | Baselines |
| `b_ic`, `b_missing`, `b_huber` | `b_grid` | rank-IC selection; indicators; pseudo-Huber XGBoost | Baselines |

Each Transformer variant also builds four ensembles with the same `b_grid` baselines: equal (Version 1 rule), inverse
MSE, rank IC and shrunken nonnegative stacking. Weights use validation data only.

**How variants are compared.** `compare` writes to `results_v2/comparison/`:

- `validation.csv`: validation scores next to the reference. Use this table to choose between variants.
- `tests.csv`: paired block-bootstrap tests on test months against the reference, against Version 1, and of each ensemble
  weighting against equal weights, with Holm adjustment. Exploratory.
- `summary.csv`: the original metrics, three drawdown definitions and the sensitivity scenarios (quintiles, 25 bps,
  returns limited to ±100%, no micro or nano stocks).
- `month_influence.csv`, `outliers.csv`, and a figure.

## Run on a laptop

Everything is off by default, so **Run All trains nothing**. Switches are in the first settings cell:

1. Run All with the defaults. Sections 1–11 read the data; configuration checks run in every section.
2. `RUN_SMOKE_TESTS = True` and Run All: short code checks on artificial months; nothing is saved.
3. Short real run, kept apart in `results_v2_2epochs/`: `RESEARCH_EPOCHS = 2`, `RUN_V2 = True`,
   `V2_STAGES = ["train", "predict", "evaluate"]`, `V2_VARIANTS = ["mlp"]`, `V2_STRATEGIES = ["localmatched"]`.
   Other strategies also train on U.S. history and take much longer.

Reset the switches afterwards. Clear outputs that show stock-level data before committing.

## Run on Amarel

### 1. Get the code

Use a separate clone next to the Version 1 folder, so Version 1 stays untouched. Off campus, connect to the VPN first.

```bash
ssh NETID@amarel.rutgers.edu
cd ~
git clone -b v2 https://github.com/MattaRithik/MachineLearningProject.git MachineLearningProject_v2
cd MachineLearningProject_v2
```

The existing environment (`~/envs/cross-market`), data (`~/MachineLearningProject/data`) and Version 1 results
(`~/MachineLearningProject/results`, read only) are found automatically. Otherwise set `CMP_ENV`, `CMP_DATA_DIR` or
`CMP_V1_RESULTS` before submitting. No new packages are needed. For a first installation, see the README of `v1.0`.

### 2. Check the environment and data

```bash
srun --partition=main --cpus-per-task=2 --mem=8G --time=00:10:00 \
    ~/envs/cross-market/bin/python amarel/check_data.py ~/MachineLearningProject/data
srun --partition=gpu --gres=gpu:1 --cpus-per-task=2 --mem=8G --time=00:10:00 \
    ~/envs/cross-market/bin/python amarel/verify_environment.py
```

### 3. Small jobs

```bash
source amarel/git_state.sh && resolve_git_state     # records the commit for direct sbatch calls
sbatch --time=01:00:00 amarel/gpu_job.slurm preflight
sbatch --time=01:00:00 amarel/gpu_job.slurm smoke
CMP_EPOCHS=2 CMP_V2_STRATEGIES=localmatched bash amarel/submit_v2.sh mlp --submit     # results_v2_2epochs/
```

### 4. Submit Version 2

Nothing is submitted without `--submit`.

```bash
bash amarel/submit_v2.sh all             # show the plan
bash amarel/submit_v2.sh all --submit    # 24 variants, then evaluation, then comparison
bash amarel/submit_v2.sh robust --submit # or one variant (needs no other variant)
bash amarel/submit_v2.sh evaluate --submit    # afterwards: evaluate what is available, then compare
```

**Version 2 has already run; these commands are for a new, separate reproduction.** Variants are independent and can run in any order; SLURM queues them within your GPU limit. Evaluation starts only if
all variants succeeded. Single stages: `sbatch amarel/gpu_job.slurm train:robust`, `predict:robust`, and
`sbatch amarel/cpu_job.slurm evaluate:robust`. Never run one variant twice at the same time; the submit script refuses this.

**Resuming.** Resubmit the same command. Completed experiments are verified and skipped; unfinished ones start again. Every
saved model carries a fingerprint of its full configuration, and an incompatible or Version 1 model raises an error. To start a
variant over, move its folder away first.

Each variant folder holds `config.json` (settings, seeds, predictors, data rows, Git commit, packages), `models/`,
`validation.csv`, `predictions/` and `evaluation/`. Job logs are `logs_v2/cmp2-<variant>_<job>.out`, and executed notebooks are in
`results_v2/executed/`.

### 5. Monitor and retrieve

```bash
squeue -u $USER
tail -f logs_v2/cmp2-robust_JOBID.out
sacct -u $USER --starttime=today --format=JobID,JobName%24,State,ExitCode,Elapsed,MaxRSS
```

On your own computer:

```bash
mkdir -p ~/Desktop/Cross_Market_Versions/Version_2
rsync -av NETID@amarel.rutgers.edu:MachineLearningProject_v2/results_v2 \
    NETID@amarel.rutgers.edu:MachineLearningProject_v2/logs_v2 ~/Desktop/Cross_Market_Versions/Version_2/
```

### 6. Update the code

Run `git pull` only when no jobs are waiting, because jobs read the notebook when they start. After any change to Version 2
training code, increase `V2_CODE_VERSION` in Section 42.

## Data and publication

The data are licensed through WRDS and are never committed; see `data/README.md`. Git also ignores models, predictions,
results, logs and ZIP archives. Predictions and executed notebooks contain stock-level values, so keep them private.
