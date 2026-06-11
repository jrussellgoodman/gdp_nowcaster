"""
Run the Phase 4 ALFRED vintage backtest.

Usage:
  source .venv/bin/activate
  python scripts/run_phase4_backtest.py

What this does:
  Loops over every quarter from Q1 2015 through Q4 2025 (44 quarters).
  For each quarter, fetches ALFRED vintage data (data as it existed on the
  last day of that quarter), fits the DFM, and records the nowcast.
  Compares to current-vintage actual GDP. Reports RMSE.

Runtime: ~20-30 min total (ALFRED fetches + 44 DFM fits × ~20 sec each).
First run fetches 44 × 12 = 528 ALFRED files. Subsequent runs use cache
and only need the DFM fitting (~15 min).
"""
import logging
import sys
from pathlib import Path

# Make sure src/ is on the path when run from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — works in script/no-display context

from src.backtest.runner import (
    compute_rmse_stats,
    plot_backtest_results,
    print_backtest_summary,
    run_backtest,
)

logging.basicConfig(
    level=logging.WARNING,       # suppress debug noise; runner prints its own progress
    format="%(levelname)s  %(name)s  %(message)s",
)

FIGURE_PATH = "docs/figures/phase4_backtest.png"

if __name__ == "__main__":
    df = run_backtest(
        start_qs="2015-01-01",
        end_qs="2025-10-01",     # Q4 2025 is the last completed quarter
        n_factors=1,             # 1-factor for first run (faster, ~20 min)
        start_data="2000-01-01",
        use_cache=True,
        save_results=True,       # saves to data/backtest_results.csv
    )

    print_backtest_summary(df)

    plot_backtest_results(df, save_path=FIGURE_PATH)
    print(f"\nPlot saved → {FIGURE_PATH}")

    # Quick sanity outputs
    stats = compute_rmse_stats(df)
    n_failed = df["converged"].value_counts().get(False, 0)
    n_nan    = df["dfm_nowcast_ann"].isna().sum()
    if n_failed > 0 or n_nan > 0:
        print(f"\n⚠  Non-convergences: {n_failed}  |  NaN nowcasts: {n_nan}")
        failed_qtrs = df.loc[~df["converged"] | df["dfm_nowcast_ann"].isna(), "quarter_label"].tolist()
        print(f"   Affected quarters: {failed_qtrs}")
    else:
        print(f"\nAll {len(df)} quarters converged successfully.")
