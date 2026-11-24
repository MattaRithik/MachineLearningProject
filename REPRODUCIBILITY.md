# Reproducibility

## Two separate revisions

| | Trained models and original evaluation | Audit and reports |
|---|---|---|
| Code | commit `69a628c08888b375621b02aca3c10e2070d20b9a`, tag `v2.0-experiment` | branch `v2-finalization` |
| Version field | `V2_CODE_VERSION = 1` | `AUDIT_CODE_VERSION = 1` |
| Outputs | `results_v2/`, `logs_v2/` (frozen) | `audit_v2/` (new) |

Version 1: commit `2f376f3a20b4c48440efbe87ab0dd7fe08a27d6d`, tag `v1.0`, `main` unchanged.

## Historical limitation

All 24 `config.json` files record `git_commit: unknown`, and the job logs say the same. What is known separately: the review package contains the
notebook and job scripts of `69a628c`, and `git_commit.txt` in the package names it. This supports, but does not prove, that every job read that
exact version. The fingerprints prove which settings were used. The historical files are not edited.

## Recorded for the Version 2 run

Python 3.12.16; NumPy 2.3.2; pandas 2.2.3; PyArrow 22.0.0; scikit-learn 1.7.1; XGBoost 3.4.1; PyTorch 2.14.1+cu126 (CUDA 12.6 build); seeds 42, 43, 44;
at most 100 epochs; 126 ordered predictors, SHA-256 `4554f15c73df38c77792dfa6133207fa63512863cba408cfd80ff070e6071e5c`; data rows U.S.
1,604,094, UK 534,295, Japan 1,175,157; training to December 2017, validation 2018–2019, test January 2020 – December 2025; one SLURM job per variant (see
each `config.json`). GPU model, CUDA driver and data checksums were not recorded.

## Recorded from now on

Commit, uncommitted-change flag and notebook checksum (resolved at submission), SLURM job and node, GPU, CUDA and cuDNN, package versions, ordered
predictor hash, and a fingerprint of each Parquet footer, in every `config.json` and checkpoint. These fields are outside the configuration
fingerprint, so existing checkpoints stay compatible. The audit stores full SHA-256 checksums of the data and of every Version 2 result file, before
and after. Compute nodes must be started with `submit_v2.sh`, `submit_audit.sh`, or after `source amarel/git_state.sh && resolve_git_state`.

## Environment lock

`python amarel/capture_environment.py environment-lock.txt` on a compute node records every installed package. It describes the environment *when
captured*; compare its six versions with those above. **PENDING** (needs Amarel).

## Checks without licensed data

```bash
python tests/test_audit.py                                   # synthetic tests
CMP_V2_RESULTS=/path/to/results_v2 python tests/test_audit.py   # also reproduces the 24 fingerprints
python amarel/audit_reports.py /path/to/results_v2 /new/folder   # tables from saved results
```

## Read-only audit on Amarel

`bash amarel/submit_audit.sh --submit` (see README) runs one CPU job. It refuses to write inside a results folder, trains nothing, and fails if any
Version 2 result file changes.
