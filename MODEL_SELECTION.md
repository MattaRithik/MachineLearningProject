# Model Selection

Only 2018–2019 validation results are used. The policy was written after the test results were known to the project, so it cannot
be a pre-registration; it is the primary metric that was already recorded for each run.

## Policy

1. **Primary:** mean monthly rank IC, averaged over seeds in each country × history cell, then equally over the six cells.
2. **Secondary:** raw MSE as a percentage gain over the zero forecast *within* a country (raw MSE is never averaged across countries),
   and MSE clipped at training-window bounds. The saved clipped MSE of direct transfer used U.S. bounds and is not comparable; Section 58
   rescores it with destination-country bounds (pending run).
3. A candidate with an undefined score in any cell is listed but not ranked (`b_clip` Elastic Net: 1 cell; `b_control` Elastic Net: 3).
4. Candidates are retained together unless paired validation evidence separates them.
5. Test Sharpe, rank IC and R² are not used.

## Results (validation rank IC, averages of histories)

| Candidate | Mean | UK | Japan | 1y | 3y | 5y | Seeds 43–44 only | Seed SD | MSE gain UK / Japan (%) |
|---|---|---|---|---|---|---|---|---|---|
| `rank` fine-tuned | 0.1273 | 0.1640 | 0.0907 | 0.1128 | 0.1307 | 0.1384 | 0.1261 | 0.0070 | −0.08 / −0.38 |
| `select_ic` joint | 0.1268 | 0.1635 | 0.0901 | 0.1238 | 0.1261 | 0.1305 | 0.1271 | 0.0041 | −0.20 / −2.64 |
| `rank` joint | 0.1242 | 0.1615 | 0.0869 | 0.1142 | 0.1185 | 0.1399 | 0.1253 | 0.0062 | −0.02 / −0.07 |
| `combined` joint | 0.1228 | 0.1611 | 0.0845 | 0.1133 | 0.1258 | 0.1292 | 0.1227 | 0.0023 | −0.76 / −1.91 |
| `mlp` fine-tuned | 0.1215 | 0.1578 | 0.0853 | 0.1155 | 0.1223 | 0.1268 | 0.1214 | 0.0047 | −0.08 / −0.66 |
| `b_huber` XGBoost | 0.1026 | 0.1518 | 0.0534 | 0.0963 | 0.1006 | 0.1108 | one seed | – | −0.07 / −1.77 |

The four candidates given in the audit request were verified; all eight country values agree. The table is in
`selection_candidates.csv`, with one row per cell in `selection_cells.csv`.

## What the difference means

- `rank` fine-tuned leads `select_ic` joint by 0.0005. Its three seeds give 0.1298, 0.1298 and 0.1224 (`select_ic` joint: 0.1263, 0.1237,
  0.1304), so the difference is far below seed variation.
- **The order depends on the aggregation.** `rank` fine-tuned leads by mean, UK, Japan and 3-year averages; `select_ic` joint by average
  cell rank, worst cell, 1-year history and seeds 43–44; `rank` joint by median and 5-year history.
- **Selection bias.** Hyperparameters were chosen with seed 42; epochs separately per seed. Both used the same months as these
  scores. `select_ic` and `combined` also chose epochs by rank IC, which flatters this metric. `rank` chose by raw MSE, so its rank IC is
  somewhat less directly optimized.
- **Level errors.** `select_ic` joint has much worse raw MSE in Japan (−2.64% against zero, versus −0.07% for `rank` joint).

## Shortlist

| | Role |
|---|---|
| **`rank` fine-tuned** | highest mean validation rank IC |
| **`rank` joint** | within 0.003 of the best, smallest MSE loss, same training loss as above |
| **`select_ic` joint** | statistically indistinguishable so far, but selected on the reported metric and with poor MSE in Japan |
| `rank` `localmatched` | required local reference |
| `b_huber` XGBoost | strongest baseline on validation |

No single winner is claimed. A one-model summary may use `rank` fine-tuned, reported next to `rank` joint. The paired bootstrap of 24
monthly rank ICs (`validation_paired_tests.csv`) is implemented and pending; it can show that candidates differ, not remove the bias above.
