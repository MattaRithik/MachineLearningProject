# Changelog: `v2-finalization` (from `v2.0-experiment`)

| File | Change | Reason |
|---|---|---|
| `Cross_Market_Project.ipynb`, cell 0 | Title text mentions Sections 52–60 | Orientation |
| cells 4–5 | `RUN_AUDIT = False`, `AUDIT_STAGES`, environment overrides | Audit off by default |
| cell 185 | `git_commit` no longer trusts `unknown` and reads packed refs; `environment_details` records more; new `feature_hash`, `dataset_fingerprint` | Provenance (D) |
| cells 222–246 (new, Phase 9) | Sections 52–60: safeguards, selection, test hierarchy, portfolio sensitivity, data audit, accounting, common clipping bounds, provenance | A, B, C, E, F, model selection |
| `amarel/git_state.sh` (new) | Resolve commit, dirty flag and notebook checksum on the submission host | D |
| `amarel/run_experiment.sh` | Use the submitted commit; stop if it is missing or the notebook changed; `audit` step | D |
| `amarel/submit_v2.sh` | Resolve and pass the Git state | D |
| `amarel/submit_audit.sh` (new) | Submit the read-only audit; plan only without `--submit` | Audit |
| `amarel/run_notebook.py` | Executed audit notebooks go to `audit_v2/private/` | Protect `results_v2/` |
| `amarel/audit_reports.py` (new) | Tables from saved results without data | Reports |
| `amarel/capture_environment.py` (new) | Package lock file | D |
| `tests/` (new) | Synthetic tests; loads functions from the notebook without running it | Verification |
| `.gitignore` | `audit_v2*/` | Licensed and large outputs |
| `README.md` and five reports | Status, audit commands, findings | Documentation |

Unchanged: every training, prediction, evaluation and comparison function of Versions 1 and 2, `V2_CODE_VERSION`, all configuration fingerprints, `main`,
`v1.0`, and all saved results.
