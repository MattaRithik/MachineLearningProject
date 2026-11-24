# Final Results (Version 2, pending real-data audit)

**Status: provisional.** Numbers come from the saved Version 2 tables of the completed Amarel run and are reproduced by
`amarel/audit_reports.py`. The data-dependent audit (Sections 56–58 of the notebook) has not been run yet, so
portfolio results and clipped-MSE comparisons may still be qualified; see `V2_AUDIT_REPORT.md`.

**All 2020–2025 results are exploratory.** Version 1 already used this test period before Version 2 was designed.

## Question and design

Can a model trained on U.S. stocks help predict next-month stock returns in the UK and Japan when local history is short? Local
histories are 1, 3 and 5 years ending December 2017; validation is 2018–2019; the test has 72 months (January 2020 – December 2025).
Transfer is measured against a local model (`localmatched`) with the same architecture and training rules.

## Findings

**1. Robust training targets are the largest improvement.** Average test rank IC of the Transformer strategies is near zero for
`control` (squared error, raw returns) and 0.10–0.11 for `huber`. All 144 rank-IC comparisons of `huber`, `clip` and `clip_month`
with `control` are positive, also after the original Holm adjustment within groups (exploratory).

**2. Transfer adds ranking skill, mainly with short local histories.** Test rank IC of the `rank` variant minus its own
local model, with p-values Holm-adjusted over these 18 tests:

| | UK 1y | UK 3y | UK 5y | Japan 1y | Japan 3y | Japan 5y |
|---|---|---|---|---|---|---|
| Fine-tuned | +0.013 (0.086) | +0.007 (0.004) | +0.008 (0.001) | +0.012 (0.032) | +0.013 (1.00) | +0.007 (1.00) |
| Joint | +0.013 (0.019) | +0.009 (0.005) | +0.011 (0.001) | +0.015 (0.016) | +0.012 (0.28) | +0.001 (1.00) |
| Direct | +0.007 (1.00) | −0.001 (1.00) | −0.004 (1.00) | +0.022 (0.002) | +0.005 (1.00) | −0.009 (1.00) |

The gains are small (about 0.01). Over all 16 variants with a local counterpart (288 tests), 38 differences are significantly positive
and 11 significantly negative. Positive ones are more frequent with one year of local data (18) than with five (10). The variants
are highly correlated and share the test months, so these counts are descriptive.

**3. The networks rank better than the baselines.** `rank` fine-tuned and joint beat XGBoost with Huber loss (`b_huber`) in all six
country-history cells (Holm over the family of baseline tests, all p ≤ 0.015). The Elastic Net comparison with `b_clip` has an undefined
cell (Japan, 1 year: constant forecast) and is not counted there. Elastic Net was never trained with a robust or ranking
objective, so it is a weaker comparison than XGBoost.

**4. There is no evidence of accurate return levels.** Out-of-sample R² is about zero or negative for the Transformers in both
validation and test. The results concern cross-sectional ranking only.

**5. Cross-stock attention is not needed for these results.** The stockwise `mlp` is as good as or better than the Transformer `robust`
in most strategies, and the average gate of the attention in `gated` is small (0.016). These comparisons use three seeds; seed variation
is of similar size to many differences between variants.

**6. The UK portfolio results are fragile and should not be a headline.** Net Sharpe, average over histories (exploratory):

| | Base | Without the single largest month | No micro/nano stocks | 25 bps | Quintiles |
|---|---|---|---|---|---|
| UK, `rank` joint | −0.27 | 0.64 | 1.25 | −0.30 | −0.32 |
| UK, `rank` fine-tuned | −0.29 | 0.55 | 1.05 | −0.32 | −0.35 |
| Japan, `rank` joint | 1.19 | 1.33 | 0.59 | 0.94 | 1.23 |
| Japan, `rank` fine-tuned | 1.56 | 1.60 | 0.52 | 1.27 | 1.64 |

- The UK full-universe series of both models lose more than their whole capital in one month (bankrupt at $1 and at $2 of capital per
  $1 sides). An additive P&L is defined in that case; a compounded one is total loss.
- The exclusion of micro and nano stocks was chosen after the results were known. It shows sensitivity, not a better strategy.
- Results with returns capped at ±100% are diagnostics of outlier influence and not investable.
- `control` has the best uncapped UK Sharpe ratios despite almost no ranking skill: the metric is dominated by chance.
- Holding unobservable positions is a further limitation (`V2_AUDIT_REPORT.md`, B).

## Conclusion

U.S. data improve cross-sectional ranking in the UK and Japan modestly, most consistently by joint training and fine-tuning and with short
local histories. The evidence does not support claims about return-level accuracy, investable profits, or a unique best model, and needs
confirmation on months after December 2025.
