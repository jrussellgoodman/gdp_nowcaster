"""
Phase 4 backtest using the 2-factor BLOCKED DFM (global + real-activity).

Same 44 quarters as the 1-factor run (Q1 2015 – Q4 2025), same ALFRED vintage
data (already fully cached from the 1-factor run — no API calls needed).
The only difference is the model: instead of one global factor loading on all
series, we use make_block_structure() which assigns:
  - global factor : all 11 (or 10) series
  - real factor   : INDPRO, PAYEMS, UNRATE, PCEC96, RSAFS, HOUST, DGORDER,
                    DSPIC96, GDPC1

Why bother?
  Phase 3 showed the blocked 2-factor model has LLF −3510 vs 1-factor −3682,
  a +172 log-likelihood-point improvement. Better fit does not guarantee better
  nowcast accuracy, but it's worth checking whether the extra factor captures
  real-activity variation that tightens RMSE or corrects the −2 pp bias.

Results saved to data/backtest_results_2factor.csv.
A side-by-side comparison with the 1-factor run is printed at the end.

Expected runtime: ~25-35 min (44 fits; ALFRED data all cached from 1-factor run).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")

import pandas as pd

from src.backtest.runner import (
    compute_rmse_stats,
    plot_backtest_results,
    print_backtest_summary,
    run_backtest,
    BACKTEST_RESULTS_PATH,
)

RESULTS_2F = Path("data/backtest_results_2factor.csv")
FIGURE_2F  = "docs/figures/phase4_backtest_2factor.png"

if __name__ == "__main__":
    df2 = run_backtest(
        start_qs="2015-01-01",
        end_qs="2025-10-01",
        n_factors=2,          # triggers make_block_structure() per vintage
        start_data="2000-01-01",
        use_cache=True,
        save_results=True,
        results_path=str(RESULTS_2F),
    )

    print("\n\n" + "=" * 65)
    print("  2-FACTOR BLOCKED DFM RESULTS")
    print("=" * 65)
    print_backtest_summary(df2)

    plot_backtest_results(df2, save_path=FIGURE_2F)
    print(f"Plot saved → {FIGURE_2F}")

    # ── Side-by-side comparison ──────────────────────────────────────────────
    df1    = pd.read_csv(BACKTEST_RESULTS_PATH)
    stats1 = compute_rmse_stats(df1)
    stats2 = compute_rmse_stats(df2)

    print("\n\n" + "=" * 72)
    print("  COMPARISON: 1-factor DFM vs 2-factor blocked DFM (ex-2020)")
    print("=" * 72)
    fmt = "  {:<35}  {:>10}  {:>10}  {:>10}"
    print(fmt.format("Metric", "1-factor", "2-factor", "Δ (2f-1f)"))
    print(fmt.format("-" * 35, "-" * 10, "-" * 10, "-" * 10))

    def row(label, k1, k2, invert=False):
        v1 = stats1[k1]; v2 = stats2[k2]
        delta = v2 - v1
        sign  = "+" if delta > 0 else ""
        better = ("✓ 2f" if (delta < 0) != invert else "✓ 1f") if abs(delta) > 0.01 else "≈ tie"
        print(fmt.format(label, f"{v1:.2f}", f"{v2:.2f}", f"{sign}{delta:.2f} {better}"))

    row("RMSE full (pp ann)",      "dfm_full_rmse",    "dfm_full_rmse")
    row("RMSE ex-2020 (primary)",  "dfm_ex2020_rmse",  "dfm_ex2020_rmse")
    row("RMSE 2020 only (crisis)", "dfm_covid_rmse",   "dfm_covid_rmse")
    row("AR(1) RMSE ex-2020",      "ar1_ex2020_rmse",  "ar1_ex2020_rmse")
    row("Skill vs AR(1) ex-2020",  "dfm_skill_ex2020", "dfm_skill_ex2020", invert=True)
    row("Mean error / bias (pp)",  "dfm_mean_error",   "dfm_mean_error")
    print("=" * 72)

    still_nan2 = df2["dfm_nowcast_ann"].isna().sum()
    if still_nan2:
        print(f"\n⚠  {still_nan2} NaN quarters in 2-factor run:")
        print(df2.loc[df2["dfm_nowcast_ann"].isna(), "quarter_label"].tolist())
    else:
        print(f"\nAll 44 quarters converged in the 2-factor run.")
