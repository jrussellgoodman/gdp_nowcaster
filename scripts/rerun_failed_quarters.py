"""
Re-run only the failed backtest quarters and merge into the main results CSV.

This is faster than re-running all 44 quarters because:
  - The ALFRED cache already has data for the 30 successful quarters.
  - Only the 14 failed quarters need new API calls.
  - The retry-with-backoff logic in fetch_vintage_series handles rate limits.

Expected runtime: ~7-10 min (14 quarters × ~20s DFM + API calls for
  uncached series with 0.6s throttle between each call).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from src.backtest.runner import (
    _fetch_current_gdp_actuals,
    compute_rmse_stats,
    plot_backtest_results,
    print_backtest_summary,
    run_single_quarter,
    BACKTEST_RESULTS_PATH,
)
from src.model.dfm import make_block_structure

FAILED = [
    ("2015-01-01", "2015-03-31"),
    ("2017-04-01", "2017-06-30"),
    ("2017-07-01", "2017-09-30"),
    ("2020-01-01", "2020-03-31"),
    ("2020-04-01", "2020-06-30"),
    ("2020-07-01", "2020-09-30"),
    ("2020-10-01", "2020-12-31"),
    ("2021-01-01", "2021-03-31"),
    ("2021-04-01", "2021-06-30"),
    ("2024-01-01", "2024-03-31"),
    ("2024-04-01", "2024-06-30"),
    ("2024-07-01", "2024-09-30"),
    ("2024-10-01", "2024-12-31"),
    ("2025-01-01", "2025-03-31"),
]

if __name__ == "__main__":
    existing = pd.read_csv(BACKTEST_RESULTS_PATH, parse_dates=["target_quarter"])
    actual_gdp = _fetch_current_gdp_actuals(start="2000-01-01")

    new_rows = []
    n = len(FAILED)
    print(f"\nRe-running {n} failed quarters with retry logic...\n")

    for i, (tq_str, eval_date) in enumerate(FAILED):
        target_q   = pd.Timestamp(tq_str)
        qtr_num    = (target_q.month - 1) // 3 + 1
        qtr_label  = f"{target_q.year} Q{qtr_num}"
        print(f"  [{i+1:2d}/{n}] {qtr_label}  eval={eval_date}  ", end="", flush=True)

        res = run_single_quarter(
            target_quarter=target_q,
            evaluation_date=eval_date,
            n_factors=1,
            start_data="2000-01-01",
            use_cache=True,
        )

        actual_qtr = actual_gdp.get(target_q, float("nan"))
        if pd.isna(actual_qtr):
            nearby = [d for d in actual_gdp.index if abs((d - target_q).days) <= 10]
            actual_qtr = float(actual_gdp.loc[nearby[0]]) if nearby else float("nan")

        actual_ann = float(actual_qtr) * 4 if not pd.isna(actual_qtr) else float("nan")
        dfm_ann    = res["dfm_nowcast_qtr"] * 4 if not pd.isna(res["dfm_nowcast_qtr"]) else float("nan")
        ar1_ann    = res["ar1_nowcast_qtr"] * 4 if not pd.isna(res["ar1_nowcast_qtr"]) else float("nan")
        dfm_error  = dfm_ann - actual_ann if not (pd.isna(dfm_ann) or pd.isna(actual_ann)) else float("nan")
        ar1_error  = ar1_ann - actual_ann if not (pd.isna(ar1_ann) or pd.isna(actual_ann)) else float("nan")

        status = "✓" if res["converged"] else "FAILED"
        print(f"DFM={dfm_ann:>+6.1f}%  actual={actual_ann:>+6.1f}%  {status}")

        new_rows.append({
            "quarter_label":   qtr_label,
            "target_quarter":  target_q,
            "evaluation_date": eval_date,
            "dfm_nowcast_ann": dfm_ann,
            "ar1_nowcast_ann": ar1_ann,
            "actual_ann":      actual_ann,
            "dfm_error_ann":   dfm_error,
            "ar1_error_ann":   ar1_error,
            "is_2020":         target_q.year == 2020,
            "converged":       res["converged"],
        })

    # Merge: replace failed rows in the existing DataFrame
    new_df    = pd.DataFrame(new_rows)
    merged    = existing[~existing["quarter_label"].isin(new_df["quarter_label"])].copy()
    full      = pd.concat([merged, new_df], ignore_index=True)
    full      = full.sort_values("target_quarter").reset_index(drop=True)

    full.to_csv(BACKTEST_RESULTS_PATH, index=False)
    print(f"\nSaved merged results to {BACKTEST_RESULTS_PATH}")

    print_backtest_summary(full)
    plot_backtest_results(full, save_path="docs/figures/phase4_backtest.png")
    print("\nPlot updated → docs/figures/phase4_backtest.png")

    still_nan = full["dfm_nowcast_ann"].isna().sum()
    if still_nan:
        print(f"\n⚠  Still {still_nan} NaN quarters after re-run:")
        print(full.loc[full["dfm_nowcast_ann"].isna(), "quarter_label"].tolist())
