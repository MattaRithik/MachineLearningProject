"""Synthetic tests of the audit calculations. Usage: python tests/test_audit.py

No licensed data, training or GPU. Expected values are worked out by hand or by separate plain-Python code.
Set CMP_V2_RESULTS to a Version 2 results folder to also check that this code reproduces the saved fingerprints.
"""
import ast
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notebook_loader import NOTEBOOK, REPORT_NAMES, load_names

ROOT = NOTEBOOK.parent
PORTFOLIO_NAMES = ["long_short_weights", "portfolio_returns", "traded_notional", "squared_error_metrics", "rank_ic",
                   "evaluate_forecast", "independent_gross_returns", "ACCOUNTING", "weighted_sides", "month_accounting",
                   "account_portfolio", "accounting_summary"]
DATA_NAMES = ["TARGET", "COUNTRIES", "PERIODS", "in_period", "return_profile", "return_profiles", "squared_return_concentration",
              "error_concentration", "add_neighbors", "alignment_checks", "extreme_records", "extreme_summary"]
SCORE_NAMES = ["monthly_lists", "score_forecasts", "rescore", "monthly_rank_ic", "validation_shortlist_tests"]
DIRECTORY = tempfile.TemporaryDirectory()
N = {"np": np, "pd": pd, "json": json, "hashlib": hashlib, "os": os, "Path": Path, "product": product,
     "AUDIT_DIR": Path(DIRECTORY.name) / "audit"}
load_names(REPORT_NAMES + PORTFOLIO_NAMES + DATA_NAMES + SCORE_NAMES + ["git_commit", "dataset_fingerprint"], N)


def close(a, b):
    assert np.allclose(a, b, atol=1e-12, equal_nan=True), (a, b)


def month_table(predictions, returns, month="2020-01-31", me=None, size="small"):
    n = len(predictions)
    return pd.DataFrame({
        "id": np.arange(1, n + 1), "predicted_month": pd.Timestamp(month), "prediction": np.asarray(predictions, dtype=float),
        "actual_return": np.asarray(returns, dtype=float), "me": np.ones(n) if me is None else np.asarray(me, dtype=float),
        "size_grp": size, "in_common": True, "has_actual": np.isfinite(np.asarray(returns, dtype=float)),
    })


def test_accounting_by_hand():
    # Quantile 0.5 of 4 stocks: short stocks 1, 2 and long stocks 3, 4, half a dollar each. Stock 4 is unobserved.
    table = month_table([1, 2, 3, 4], [0.10, -0.20, 0.30, np.nan])
    row = N["account_portfolio"](table, 0.5).iloc[0]
    close(row["legacy"], 0.30 - (-0.05))          # weight of stock 4 moved to stock 3
    close(row["assume_zero"], 0.15 + 0.05)        # 0.5 * 0.30 + 0 - 0.5 * (0.10 - 0.20)
    close(row["assume_minus30"], 0.15 - 0.15 + 0.05)
    close(row["assume_adverse"], 0.15 - 0.15 + 0.05)
    close([row["long_unobserved"], row["short_unobserved"]], [0.5, 0.0])
    assert np.isnan(row["exact"]) and not row["exactly_measured"]
    # A missing short: -30% helps it, the adverse bound hurts it
    row = N["account_portfolio"](month_table([1, 2, 3, 4], [np.nan, -0.20, 0.30, 0.10]), 0.5).iloc[0]
    close([row["assume_zero"], row["assume_minus30"], row["assume_adverse"]], [0.20 + 0.10, 0.45, 0.15])


def test_selection_never_uses_labels():
    weights = N["long_short_weights"](month_table([1, 2, 3, 4], [0, 0, 0, 0]), 0.5)
    missing = N["long_short_weights"](month_table([1, 2, 3, 4], [np.nan] * 4), 0.5)
    assert weights.equals(missing)


def test_legacy_matches_version_2_and_plain_python():
    rng = np.random.default_rng(1)
    months = []
    for i, month in enumerate(pd.date_range("2020-01-31", periods=6, freq="ME")):
        n = 60 + 5 * i
        returns = rng.normal(0, 0.1, n)
        returns[rng.random(n) < 0.15] = np.nan
        months.append(month_table(rng.normal(size=n), returns, month))
    stocks = pd.concat(months, ignore_index=True)
    ours, old = N["account_portfolio"](stocks), N["evaluate_forecast"](stocks)
    close(ours["legacy"], old["gross_return"])
    close(ours["turnover_static"], old["turnover"])
    close(N["independent_gross_returns"](stocks).to_numpy(), old["gross_return"])
    previous, previous_return = {}, {}
    for (_, part), (_, row) in zip(stocks.groupby("predicted_month"), ours.iterrows()):
        ranked = sorted(part[["prediction", "id", "actual_return"]].itertuples(index=False))
        n = int(0.10 * len(ranked))
        weights = {s.id: -1 / n for s in ranked[:n]} | {s.id: 1 / n for s in ranked[-n:]}
        outcome = {s.id: 0.0 if np.isnan(s.actual_return) else s.actual_return for s in ranked}
        close(row["assume_zero"], sum(w * outcome[i] for i, w in weights.items()))
        held = {i: w * (1 + previous_return[i]) for i, w in previous.items()}
        close(row["turnover_drift_assume_zero"], sum(abs(weights.get(i, 0) - held.get(i, 0)) for i in set(weights) | set(held)))
        previous, previous_return = weights, outcome
        if row["exactly_measured"]:
            close([row["exact"]] * 4, [row["legacy"], row["assume_zero"], row["assume_minus30"], row["assume_adverse"]])


def test_turnover_and_costs_by_hand():
    first = month_table([1, 2, 3, 4], [0.0, 0.0, 1.0, 0.0])
    second = month_table([1, 2, 3, 4], [0.0, 0.0, 0.0, 0.0], "2020-02-29")
    months = N["account_portfolio"](pd.concat([first, second]), 0.5)
    close(months["turnover_static"], [2.0, 0.0])             # opening $1 long and $1 short, then no change
    close(months["turnover_drift_assume_zero"], [2.0, 0.5])  # stock 3 doubled from 0.5 to 1.0 and is cut back
    table = N["accounting_summary"](months, 10).set_index(["accounting", "turnover_rule"])
    close(table.loc[("assume_zero", "static"), "additive_net_pnl"], 0.5 - 2.0 * 0.001)
    close(table.loc[("assume_zero", "drift"), "additive_net_pnl"], 0.5 - 2.5 * 0.001)


def test_unmeasured_months_are_reported():
    first = month_table([1, 2, 3, 4], [0.0, 0.1, 0.2, np.nan])
    second = month_table([1, 2, 3, 4], [0.0, 0.0, 0.1, 0.1], "2020-02-29")
    table = N["accounting_summary"](N["account_portfolio"](pd.concat([first, second]), 0.5), 10)
    row = table[(table["accounting"] == "exact") & (table["turnover_rule"] == "static")].iloc[0]
    assert row["months_measured"] == 1 and row["months_with_unobserved"] == 1 and "not a 72-month" in row["note"]


def test_weighting_uses_formation_size_only():
    table = month_table([1, 2, 3, 4], [0.1, 0.1, np.nan, 0.3], me=[1, 3, 1, 9])
    weights = N["weighted_sides"](table, 0.5, "value")
    close(weights.loc[[1, 2, 3, 4]], [-0.25, -0.75, 0.1, 0.9])
    for weighting in ["equal", "value", "value_capped"]:
        weights = N["weighted_sides"](table, 0.5, weighting)
        close([weights[weights > 0].sum(), weights[weights < 0].sum()], [1, -1])
    unchanged = N["weighted_sides"](table.assign(actual_return=np.nan), 0.5, "value_capped")
    assert unchanged.equals(weights)
    no_size = N["weighted_sides"](table.assign(me=np.nan), 0.5, "value")  # falls back to equal weights
    close(no_size.abs(), 0.5)


def test_constant_forecast_takes_no_position():
    row = N["account_portfolio"](month_table([1, 1, 1, 1], [0.1, 0.2, 0.3, np.nan]), 0.5).iloc[0]
    close([row["legacy"], row["assume_zero"], row["turnover_static"], row["n_side"]], [0, 0, 0, 0])


def test_universe_exclusion():
    table = pd.concat([month_table([1, 2, 3, 4], [0.1] * 4), month_table([9, 9.5], [5.0, 5.0], size="nano").assign(id=[8, 9])])
    row = N["account_portfolio"](table, 0.5, exclude=("micro", "nano")).iloc[0]
    close([row["n_side"], row["legacy"]], [2, 0.0])


def test_identical_forecasts_and_bounds_give_identical_scores():
    rng = np.random.default_rng(2)
    prediction, actual = rng.normal(0, 0.02, 80), rng.normal(0, 0.1, 80)
    actual[0] = 50.0
    table = pd.DataFrame({"eom": np.repeat(pd.to_datetime(["2018-01-31", "2018-02-28"]), 40), "prediction": prediction,
                          "seed_42": prediction, "actual_return": actual})
    scores = N["rescore"](table, (-0.2, 0.2))
    for name in ["mse", "clipped_mse", "rank_ic"]:
        assert scores[name].nunique() == 1
    other = N["rescore"](table, (-0.5, 0.8))  # only the clipped score depends on the bounds
    close(other["mse"], scores["mse"])
    close(other["rank_ic"], scores["rank_ic"])
    assert (other["clipped_mse"] != scores["clipped_mse"]).all()
    # Worked out separately; the raw MSE keeps the unmodified 5,000% return
    clipped = np.minimum(np.maximum(actual, -0.2), 0.2)
    close(scores["clipped_mse"], ((clipped - prediction) ** 2).reshape(2, 40).mean(axis=1).mean())
    close(scores["mse"], ((actual - prediction) ** 2).reshape(2, 40).mean(axis=1).mean())


def test_unlabeled_stocks_do_not_enter_scores():
    table = pd.DataFrame({"eom": pd.Timestamp("2018-01-31"), "prediction": [0.1, 0.2, 0.3, 9.0], "actual_return": [0.0, 0.1, 0.2, np.nan]})
    close(N["rescore"](table, (-1, 1)).loc[0, ["mse", "rank_ic"]].astype(float), [0.01, 1.0])


def test_return_profile_labels():
    profile = N["return_profile"](pd.Series([3.0, -5.0, 0.5, -0.99, np.nan, 1.0]))
    close([profile["largest_positive"], profile["largest_negative"], profile["largest_absolute"]], [3, -5, 5])
    assert (profile["above_plus_100pct"], profile["below_minus_100pct"], profile["missing"]) == (1, 1, 1)
    assert profile["at_or_below_minus_99pct"] == 2


def test_squared_return_and_squared_error_concentration_differ():
    stocks = pd.concat([month_table([0.0, 0.0], [3.0, 1.0]), month_table([0.0] * 4, [2.0, 0, 0, 0], "2020-02-29")], ignore_index=True)
    by_month, by_stock = N["squared_return_concentration"](
        stocks.rename(columns={"actual_return": N["TARGET"]}))
    close(by_month["pooled_share"], [10 / 14, 4 / 14])
    close(by_month["share_of_zero_forecast_mse"], [5 / 6, 1 / 6])  # monthly means 5 and 1
    close(by_stock["pooled_share"].sum(), 1)
    perfect_large = stocks.assign(prediction=[3.0, 0.0, 0.0, 0, 0, 0])
    result = N["error_concentration"](perfect_large, {})
    close(result["mse"], (0.5 + 1) / 2)
    close(result["top_month_share_of_mse"], 1 / 1.5)  # February, although January dominates squared returns
    assert result["top_month"] == pd.Timestamp("2020-02-29")
    close(result["oos_r2"], 1 - 0.75 / 3)


def test_mse_contributions_add_up():
    rng = np.random.default_rng(3)
    parts = [month_table(rng.normal(0, .1, n), rng.normal(0, .1, n), m) for n, m in [(30, "2020-01-31"), (90, "2020-02-29")]]
    stocks = pd.concat(parts, ignore_index=True)
    expected = np.mean([((p["actual_return"] - p["prediction"]) ** 2).mean() for p in parts])
    close(N["error_concentration"](stocks, {})["mse"], expected)


def synthetic_panel():
    """Stock 1 complete; stock 2 skips a month; stock 3 leaves early; one target is inconsistent on purpose."""
    dates = pd.date_range("2020-01-31", periods=4, freq="ME")
    rows = [(1, d, r) for d, r in zip(dates, [0.0, 0.5, 4.0, 0.1])] + [(2, dates[0], 0.1), (2, dates[2], 0.2)] + [(3, dates[0], 0.1)]
    df = pd.DataFrame(rows, columns=["id", "eom", "ret_1_0"])
    df["me"] = [100, 150, 750, 825, 10, 12, 5]
    df["prc"] = [1, 1.5, 1.5, 1.65, 2, 2, 0.5]
    df["rmax1_21d"], df["gvkey"], df["permno"], df["size_grp"] = [0, .1, 3.9, .1, 0, 0, 0], "001", np.nan, "nano"
    df[N["TARGET"]] = [0.499, 3.999, 0.3, np.nan, 0.05, np.nan, np.nan]
    df["predicted_month"] = df["eom"] + pd.offsets.MonthEnd(1)
    return N["add_neighbors"](df)


def test_alignment_and_extreme_evidence():
    df = synthetic_panel()
    row = N["alignment_checks"]({"GBR": df}).iloc[0]
    assert (row["pairs"], row["mismatches"], row["extreme_target_pairs"], row["extreme_target_mismatches"]) == (3, 1, 1, 0)
    assert row["target_without_next_month_row"] == 1            # stock 2 in January (its February row is missing)
    assert row["missing_target_final_data_month"] == 1 and row["missing_target_last_row_of_stock"] == 2
    records = N["extreme_records"](df, top=1).iloc[0]
    assert records["id"] == 1 and records["me_growth"] == 5 and bool(records["me_growth_matches_return"])
    assert records["price_growth"] == 1 and not records["price_growth_matches_return"]  # price flat: needs an adjustment
    assert bool(records["mostly_one_day"]) and not records["reversal_next_month"]
    summary = N["extreme_summary"](N["extreme_records"](df, top=1).assign(country="GBR", period="test"))
    assert summary.loc[0, "me_growth_matches_return"] == 1 and not {"id", "gvkey", "permno"} & set(summary.columns)


def validation_rows(ic_by_window, mse=1.0, seeds=(42, 43, 44)):
    return pd.DataFrame([
        {"variant": "a", "strategy": "joint", "country": c, "window": w, "seed": s, "rank_ic": ic, "mse": mse,
         "clipped_mse": 0.1, "clipping_bounds": "x"}
        for (c, w), ic in ic_by_window.items() for s in seeds])


def test_selection_uses_relative_mse_and_keeps_undefined():
    windows = ["train_1y", "train_3y", "train_5y"]
    ic = {(c, w): 0.1 * (i + 1) for i, (c, w) in enumerate(product(["GBR", "JPN"], windows))}
    rows = validation_rows(ic)
    cells = N["validation_cells"](rows, pd.Series({"GBR": 2.0, "JPN": 1.0}))
    table = N["selection_table"](cells).iloc[0]
    close([table["mean_ic"], table["ic_GBR"], table["ic_JPN"], table["worst_cell_ic"]], [0.35, 0.2, 0.5, 0.1])
    close([table["mse_gain_pct_GBR"], table["mse_gain_pct_JPN"]], [50, 0])
    rows.loc[rows.index[0], "rank_ic"] = np.nan  # one undefined seed makes the cell and the mean undefined
    table = N["selection_table"](N["validation_cells"](rows, pd.Series({"GBR": 2.0, "JPN": 1.0}))).iloc[0]
    assert np.isnan(table["mean_ic"]) and table["undefined_cells"] == 1


def test_validation_tests_prefer_the_best_and_skip_incomplete():
    rng = np.random.default_rng(4)
    rows = []
    for (variant, level), country, window in product([("good", 0.2), ("bad", 0.0)], ["GBR", "JPN"], ["train_1y", "train_3y", "train_5y"]):
        for month in pd.date_range("2018-01-31", periods=24, freq="ME"):
            rows.append({"predicted_month": month, "rank_ic": level + rng.normal(0, .02),
                         "variant": variant, "strategy": "joint", "country": country, "window": window})
    rows += [{**r, "variant": "partial"} for r in rows if r["variant"] == "good" and r["country"] == "GBR"]
    table = N["validation_shortlist_tests"](pd.DataFrame(rows))
    assert table["variant"].tolist() == ["good", "bad"]
    close(table.loc[1, "best_minus_candidate"], table.loc[0, "mean_ic"] - table.loc[1, "mean_ic"])
    assert table.loc[1, "p_holm"] < 0.001 and table.loc[1, "ci_low"] > 0.15


def test_family_adjustment_and_global_limit():
    p = pd.Series([0.01, 0.04, np.nan, 0.03])
    close(N["holm_adjust"](p), [0.03, 0.06, np.nan, 0.06])
    tests = pd.DataFrame({"p_value": [1 / 2001] * 9792, "p_adjusted": 0.01})
    summary = N["existing_test_summary"](tests).set_index("quantity")["value"]
    assert summary["Holm over all comparisons, p < 0.05"] == 0  # the resampling limit alone prevents rejection
    indices = N["audit_block_indices"](72, np.random.default_rng(0), 50)
    assert indices.shape == (50, 72) and indices.max() < 72 and (np.diff(indices[:, :6]) == 1).all()


def test_drawdown_and_bankruptcy():
    months = pd.date_range("2020-01-31", periods=3, freq="ME")
    table = pd.DataFrame({"variant": "a", "strategy": "joint", "country": "GBR", "window": "train_1y", "scenario": "base",
                          "seed": "main", "predicted_month": months, "gross_return": [0.1, -2.5, 0.2],
                          "net_return": [0.1, -2.5, 0.2], "turnover": 1.0, "n_side": 5})
    row = N["portfolio_sensitivity"](table, [("a", "joint")]).iloc[0]
    close([row["additive_net_pnl"], row["compounded_drawdown_capital_1"]], [-2.2, -1])
    assert row["bankrupt_capital_1"] and row["bankrupt_capital_2"] and row["largest_abs_month"] == months[1]


def test_read_only_outputs_and_manifest():
    folder = Path(DIRECTORY.name) / "source"
    folder.mkdir(exist_ok=True)
    (folder / "a.csv").write_text("1")
    before = N["tree_manifest"](folder)
    N["save_audit"](pd.DataFrame({"x": [1]}), "table")
    assert N["tree_manifest"](folder).equals(before) and (N["AUDIT_DIR"] / "aggregate" / "table.csv").exists()
    (folder / "a.csv").write_text("2")
    assert not N["tree_manifest"](folder).equals(before)


def test_data_dependent_stages_run_on_synthetic_files():
    """The data, validation and portfolio stages, end to end, on artificial Parquet files and forecasts."""
    rng = np.random.default_rng(5)
    root = Path(DIRECTORY.name) / "run"
    source, files, data = root / "results_v2", {}, {}
    root.mkdir()
    names = {**N, "AUDIT_DIR": root / "audit", "V2_SOURCE": source, "AUDIT_BOOTSTRAP": 200, "KEYS": ["country", "id", "eom", "predicted_month"],
             "AUDIT_CANDIDATES": [("rank", "direct"), ("b_huber", "xgboost"), ("rank", "joint")], "FILES": files, "data": data,
             "expected_months": {"validation": 24, "test": 72}}
    for country in ["USA", "GBR", "JPN"]:
        eom = pd.date_range("2012-12-31", "2025-11-30", freq="ME")
        df = pd.MultiIndex.from_product([range(1, 41), eom], names=["id", "eom"]).to_frame(index=False)
        df = df[rng.random(len(df)) > 0.05].reset_index(drop=True)
        n = len(df)
        target = np.where(rng.random(n) < 0.03, np.nan, rng.normal(0, 0.1, n))
        size = pd.Series(rng.choice(["nano", "micro", "small", "large"], n), dtype="category")
        df = df.assign(gvkey="1", permno=np.nan, size_grp=size, me=rng.lognormal(size=n), prc=1.0, rmax1_21d=0.1, ret_1_0=rng.normal(0, .1, n))
        df[N["TARGET"]] = target
        df.assign(eom=df["eom"].dt.date).to_parquet(root / f"{country}.parquet")  # dates are stored as in the real files
        files[country] = root / f"{country}.parquet"
        data[country] = df.assign(predicted_month=df["eom"] + pd.offsets.MonthEnd(1))
    for variant, strategy, columns in [("rank", "direct", 3), ("b_huber", "xgboost", 0)]:
        (source / variant / "predictions").mkdir(parents=True)
        for country, window in product(["GBR", "JPN"], ["train_1y", "train_3y", "train_5y"]):
            df = data[country]
            parts = []
            for period in ["validation", "test"]:
                part = df.loc[N["in_period"](df["predicted_month"], period), ["id", "eom", "predicted_month"]].assign(country=country)
                for seed in range(42, 42 + columns):
                    part[f"seed_{seed}"] = rng.normal(0, 0.01, len(part))
                part["prediction"] = part[[c for c in part if c.startswith("seed_")]].mean(axis=1) if columns else rng.normal(0, .01, len(part))
                parts.append(part.assign(period=period, sources="run-1"))
            name = f"{country}_{'all' if strategy == 'direct' else window}_{strategy}.parquet"
            pd.concat(parts).to_parquet(source / variant / "predictions" / name)
    rows = []
    for strategy, seeds in [("direct", [42, 43, 44])]:
        for country, seed in product(["GBR", "JPN"], seeds):
            rows.append({"variant": "rank", "strategy": strategy, "country": country, "window": "all", "seed": seed, "stage": "seed",
                         "mse": 0.01, "clipped_mse": 0.01, "rank_ic": 0.1, "selected": np.nan})
    pd.DataFrame(rows).to_csv(source / "rank" / "validation.csv", index=False)
    load_names(["clip_limits", "add_actuals", "read_source_forecast", "run_data_audit", "run_validation_audit", "run_portfolio_audit", "report_missing",
                "validation_strategies", "load_audit_data", "AUDIT_UNIVERSES", "TARGET_QUANTILES"] + DATA_NAMES + SCORE_NAMES
               + REPORT_NAMES + PORTFOLIO_NAMES, names)
    names.update(AUDIT_BOOTSTRAP=200, AUDIT_CANDIDATES=[("rank", "direct"), ("b_huber", "xgboost"), ("rank", "joint")])
    before = names["tree_manifest"](source)
    for stage in ["run_data_audit", "run_validation_audit", "run_portfolio_audit"]:
        with contextlib.redirect_stdout(io.StringIO()):
            names[stage]()
    assert names["tree_manifest"](source).equals(before)
    folder = root / "audit" / "aggregate"
    assert len(pd.read_csv(folder / "missing_forecasts_portfolio.csv")) == 6  # rank + joint was never created
    scores = pd.read_csv(folder / "validation_rescored.csv")
    direct = scores[(scores["strategy"] == "direct") & (scores["model"] == "average") & (scores["country"] == "GBR")]
    assert direct["upper_bound"].nunique() == 3 and direct["mse"].nunique() == 1  # bounds follow the local history
    assert scores.loc[scores["model"] == "seed_42", "mse_original"].notna().all()
    accounting = pd.read_csv(folder / "portfolio_accounting.csv")
    assert len(accounting) == 12 * 18 * 2 * 10 and (accounting["months_with_unobserved"] > 0).any()
    for path in folder.glob("*.csv"):  # no identifiers outside the private folder
        assert not {"id", "gvkey", "permno"} & set(pd.read_csv(path, nrows=1).columns), path
    assert {"id", "gvkey"} <= set(pd.read_csv(root / "audit" / "private" / "extreme_records.csv", nrows=1).columns)


def test_commit_resolution():
    root = Path(DIRECTORY.name) / "repo"
    (root / ".git").mkdir(parents=True, exist_ok=True)
    namespace = {"os": os, "PROJECT_ROOT": root}
    load_names(["git_commit"], namespace)
    saved = os.environ.pop("CMP_GIT_COMMIT", None)
    (root / ".git" / "HEAD").write_text("ref: refs/heads/v2\n")
    (root / ".git" / "packed-refs").write_text("# pack-refs\nabc123 refs/heads/v2\n")
    assert namespace["git_commit"]() == "abc123"
    os.environ["CMP_GIT_COMMIT"] = "unknown"  # the Version 2 failure: this value must not hide the real commit
    assert namespace["git_commit"]() == "abc123"
    os.environ["CMP_GIT_COMMIT"] = "def456"
    assert namespace["git_commit"]() == "def456"
    del os.environ["CMP_GIT_COMMIT"]
    if saved is not None:
        os.environ["CMP_GIT_COMMIT"] = saved


def test_dataset_fingerprint_changes_with_the_footer():
    folder = Path(DIRECTORY.name)
    paths = []
    for i, footer in enumerate([b"footer-a", b"footer-b"]):
        path = folder / f"{i}.parquet"
        path.write_bytes(b"PAR1data" + footer + len(footer).to_bytes(4, "little") + b"PAR1")
        paths.append(path)
    assert N["dataset_fingerprint"](paths[0]) != N["dataset_fingerprint"](paths[1])


def test_git_state_script():
    with tempfile.TemporaryDirectory() as name:
        run = lambda command: subprocess.run(["bash", "-c", command], cwd=name, capture_output=True, text=True)
        tool = 'sha256sum() { shasum -a 256 "$@"; }; ' if sys.platform == "darwin" else ""
        source = f'{tool}source "{ROOT}/amarel/git_state.sh"; resolve_git_state && echo "$CMP_GIT_COMMIT $CMP_GIT_DIRTY $CMP_NOTEBOOK_SHA256"'
        assert run(source).returncode != 0  # no repository: an error, never "unknown"
        run("git init -q . && echo 1 > Cross_Market_Project.ipynb && git add . && git -c user.name=t -c user.email=t@t commit -qm t")
        commit, dirty, digest = run(source).stdout.split()
        assert len(commit) == 40 and dirty == "0" and len(digest) == 64
        run("echo 2 > Cross_Market_Project.ipynb")
        assert run(source).stdout.split()[1] == "1"


def test_notebook_structure():
    notebook = json.loads(NOTEBOOK.read_text())
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for number, source in enumerate(code):
        tree = ast.parse(source)
        imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
        assert not imports or number == 0, "imports belong in the first code cell"
    for cell in notebook["cells"]:  # aggregate outputs may stay; stock identifiers may not
        shown = json.dumps(cell.get("outputs", []))
        assert "gvkey" not in shown and "permno" not in shown
    settings = "\n".join(code[1:2])
    for switch in ["RUN_V2", "RUN_AUDIT", "RUN_SMOKE_TESTS", "RUN_PREFLIGHT", "RUN_PHASE6_PREDICTIONS", "RUN_FINAL_EVALUATION"]:
        assert any(line.startswith(f"{switch} = False") for line in settings.splitlines()), switch
    assert 'RUN_MODE = "off"' in settings


def test_training_code_is_unchanged():
    """Everything that trains, predicts or evaluated Version 2 equals the tagged experiment; only provenance may differ."""
    old = subprocess.run(["git", "show", "v2.0-experiment:Cross_Market_Project.ipynb"], cwd=ROOT, capture_output=True, text=True)
    if old.returncode != 0:
        print("  skipped: tag v2.0-experiment not available")
        return
    sources = [["".join(c["source"]) for c in json.loads(text)["cells"]] for text in [old.stdout, NOTEBOOK.read_text()]]
    changed = [i for i, (a, b) in enumerate(zip(*sources)) if a != b]
    assert changed == [0, 4, 5, 185], changed  # title, switches, switch reading, environment_details/git_commit
    def definitions(source):
        return {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    a, b = definitions(sources[0][185]), definitions(sources[1][185])
    assert {name for name in b if a.get(name) != b[name]} == {
        "git_commit", "environment_details", "feature_hash", "dataset_fingerprint"}


def test_saved_fingerprints_are_reproduced_by_this_code():
    source = os.environ.get("CMP_V2_RESULTS")
    if not source:
        print("  skipped: set CMP_V2_RESULTS to check the saved configurations")
        return
    names = {"np": np, "pd": pd, "json": json, "hashlib": hashlib, "product": product, "RESEARCH_EPOCHS": 100, "COUNTRIES": ["USA", "GBR", "JPN"]}
    load_names(["SEED", "PERIODS", "TARGET_QUANTILES", "MONTH_QUANTILES", "V2_EMBEDDING_DIM", "V2_BLOCKS", "V2_GRID", "V2_FINE_TUNING",
                "V2_ELASTIC_NET", "V2_XGBOOST_GRID", "ELASTIC_NET_GRID", "XGBOOST_GRID", "PATIENCE", "EARLY_STOPPING",
                "TRANSFORMER_SEEDS", "N_HEADS", "FEEDFORWARD_DIM", "V2_CODE_VERSION", "BASE_SPEC", "V2_TRANSFORMER_STRATEGIES",
                "VARIANTS", "BASE_BASELINE_SPEC", "BASELINE_VARIANTS", "ALL_VARIANTS", "variant_spec", "variant_config",
                "fingerprint"], names)
    checked = 0
    for path in Path(source).glob("*/config.json"):
        saved = json.loads(path.read_text())
        names["selected_features"] = saved["features"]
        names["data"] = {country: range(rows) for country, rows in saved["data_rows"].items()}
        assert names["fingerprint"](saved["variant"]) == saved["fingerprint"], path
        checked += 1
    assert checked == 24
    print(f"  {checked} fingerprints reproduced by the current notebook code")


if __name__ == "__main__":
    tests = [(name, test) for name, test in list(globals().items()) if name.startswith("test_")]
    failed = 0
    for name, test in tests:
        try:
            test()
            print(f"ok      {name}")
        except Exception as error:  # report every failure, then fail the run
            failed += 1
            print(f"FAILED  {name}: {type(error).__name__}: {error}")
    print(f"{len(tests) - failed} of {len(tests)} tests passed")
    sys.exit(1 if failed else 0)
