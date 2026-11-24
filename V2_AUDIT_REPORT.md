# Version 2 Audit Report

**Not final: items marked PENDING need the Amarel run of the read-only audit.** Version 2 = tag `v2.0-experiment`, commit `69a628c`.
Original metrics stay available under their original names; corrected or additional quantities have new names in `audit_v2/`.

## Checked

- The review package is identical to its ZIP (163 files). Its notebook equals the tagged notebook cell by cell.
- All 24 variant configurations reproduce their fingerprints, also when recomputed with the finalization code. Executed-notebook messages
  are present in all logs of the 24 variants, evaluation and comparison.
- The notebook cells that train, predict and evaluate Version 2 are unchanged (a test compares them with the tag). Only the title, switches
  and provenance functions differ; `V2_CODE_VERSION` stays 1, so existing checkpoints remain valid.

## A. Extreme returns

| Fact (from the saved comparison) | Value |
|---|---|
| Largest positive target: U.S. training / UK test / Japan test | about 15,349 / 331 / 11.9 (decimal) |
| Top-10 share of *squared returns*: U.S. training / UK test | 99.9% / 95.8% |
| April 2020 share of UK zero-forecast MSE (monthly-equal) | about 86% |

- **Labeling bug (fixed in new outputs):** `outliers.csv` calls the maximum `largest`; it ignores negative returns. Audit tables
  report the largest positive, negative and absolute returns separately. The original table and function are unchanged.
- **Two concentrations are separate:** squared *returns* (zero-forecast loss) and squared *errors* of a model, each pooled and with the monthly
  weights of the reported MSE.
- **Alignment:** logs of the completed run report 100.00% agreement between the target and the next `ret_1_0` for consecutive months in all
  countries. This shows internal consistency; it does not show that a return is economically correct.
- **Cause unresolved. PENDING.** The files hold no delisting return, local return, exchange rate, split or adjustment factor, so units,
  currency, corporate actions and delisting cannot be confirmed from them. The audit collects evidence (market-equity and price growth against
  the return, concentration on one day, reversal, identifier continuity, price level, size). An exchange rate alone cannot plausibly explain a 331x
  monthly return; a price-scale or adjustment discontinuity is a candidate, not a finding. Confirming it needs the WRDS source tables.
- Nothing was removed, capped or replaced in any original metric.

## B. Portfolio accounting with missing returns

**Confirmed issue (kept as `legacy`):** selection has no look-ahead, but each side's return is the mean over stocks whose return was later
observed, so the missing weight is reassigned after the fact. `rank` fine-tuned and joint:

| | Months with at least one unobserved position | Share of positions unobserved | Largest month |
|---|---|---|---|
| UK | 92–94% | 0.9–1.0% | 2.9% |
| Japan | 62–67% | 0.1% | 0.6% |

An exact intended-weight P&L can therefore be measured in few UK months. Section 57 adds `exact` (undefined when not measurable),
`assume_zero`, `assume_minus30` and `assume_adverse`, with exposure shares, static and drifted turnover, and costs of 10 and 25 bps. The
imputed scenarios are assumptions. Legacy returns and turnover are reproduced to 1e-12 by the same code on synthetic data.
**Values PENDING.** Delisting-adjusted returns need WRDS source data.

Ignored by the original turnover: drift of weights between rebalancing dates (documented in Section 36). Both rules are now reported.

## C. Validation clipping bounds

**Confirmed:** `train_variant` scores direct transfer with U.S. `train_full` bounds, while local, fine-tuned and joint models use the local
window. Their clipped MSEs are not comparable. Direct clipped MSE was not used to select any hyperparameter, but `validation.csv` shows it
next to the others. Section 58 recomputes all scores from saved forecasts with destination-window bounds, next to the originals, with a
reproduction check of raw MSE and rank IC. No retraining is needed. **Values PENDING.** Ensemble weights already used common destination bounds.

## D. Provenance

**Confirmed cause of `git_commit: unknown`:** `run_experiment.sh` ran `git` on the compute node, exported `unknown` when that failed, and the
notebook trusted the exported value over its own `.git` fallback. A failed `git status` also counted as clean.
**Fixed for future jobs:** the commit, dirty flag and notebook checksum are resolved on the submission host and passed on; a job stops if
the commit is missing or the notebook changed after submission. See `REPRODUCIBILITY.md`. Historical configurations are left as they are.

## E. Statistical testing

| Scope of adjustment (9,792 comparisons; 9,735 with p-values) | Significant at 5% |
|---|---|
| None | 2,518 |
| Holm within original groups (as published) | 1,445 |
| Holm over all comparisons | 0 |

The last row is not evidence of no effects: with 2,000 resamples the smallest p-value is 1/2,001, so adjustment over about 9,700 tests cannot
reject. **No global family-wise control is claimed.** Two new families (transfer against local; candidates against baselines) use 20,000
resamples and Holm over their own rows. They were designed after the results were known. Confidence intervals are pointwise.

An earlier informal review quoted unadjusted intervals for the transfer gains; the Holm-adjusted values here replace them. One result changed:
UK 1-year fine-tuning has adjusted p = 0.086.

## F. Portfolio sensitivity

`rank` joint in the UK: net Sharpe −0.27 (base) and +1.25 without micro and nano stocks, as reported; both reproduced. Removing one month
(April 2020 for most series) changes −0.27 to 0.64, so most of the gap is one event. Japan moves the other way: 1.19 to 0.59 without micro and nano
stocks, so part of the full-universe Japanese result depends on small stocks. Section 57 crosses universes, quantiles (10%, 20%, 30%), costs
(10, 25 bps) and weighting (equal, value, capped value), all using formation-date information only. **Values PENDING.**
Additive P&L is distinguished from compounded wealth, with capital base, leverage and bankruptcy rule stated in Section 55.

## Outstanding questions

1. Origin of the extreme returns (WRDS source tables), and the treatment recommended by Prof. Dinc.
2. Delisting returns for missing targets.
3. Whether clipped-MSE comparisons of direct transfer change with common bounds.
4. Package versions: six are recorded per run; the rest are requirements pins until the lock file is captured.
5. Characteristic definitions are not stored in the notebook.
6. Validation months were used for early stopping, hyperparameters and ensembles; no untouched holdout exists before 2026 data.
