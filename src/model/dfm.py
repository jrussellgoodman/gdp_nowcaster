"""
dfm.py — Dynamic Factor Model (DFM) for GDP nowcasting.

What this does, in plain English:
  The core idea of the NY Fed nowcasting model is that the economy has a small
  number of hidden "factors" — like "overall economic momentum" — that drive
  all the individual indicators we observe (jobs growth, industrial production,
  retail sales, etc.). If we can estimate this hidden factor in real time from
  the monthly data, we can use it to predict GDP before the official quarterly
  number is released.

  statsmodels' DynamicFactorMQ implements this model with a Kalman filter and
  EM algorithm. We don't need to code the math ourselves — we just need to
  prepare the data in the right format and interpret the output.

  Data format DynamicFactorMQ expects:
    - endog           : monthly series (a DataFrame, one column per series)
    - endog_quarterly : quarterly GDP on a monthly date grid (values only at
                        March/June/September/December, NaN everywhere else)

  After fitting, the "nowcast" is the model's prediction for the current quarter
  where GDP hasn't been officially released yet.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ

from src.data.fred_loader import SERIES_CATALOG, build_panel

logger = logging.getLogger(__name__)


def prepare_dfm_data(
    start: str = "2000-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download, transform, and prepare the two DataFrames that DynamicFactorMQ
    expects — one for monthly series, one for quarterly GDP.

    IMPORTANT — two separate frequency grids:
      DynamicFactorMQ does NOT want GDP on a monthly grid with NaN gaps.
      It wants two DataFrames on *separate* frequency grids:
        monthly_df   : monthly DatetimeIndex (freq='MS')
        quarterly_df : quarterly DatetimeIndex (freq='QS')
      The model's Kalman filter internally handles the alignment between them.

    What this function does:
      1. Builds the monthly panel via build_panel() for the monthly indicators.
      2. Fetches GDPC1 separately, applies log_diff, and returns it on its
         natural quarterly-frequency index — no monthly-grid alignment needed.
      3. Fetches one extra period before 'start' for both so the first
         log-difference is well-defined (not NaN).
      4. Drops any leading all-NaN rows from the monthly panel.

    Parameters
    ----------
    start : str
        First period to include in the model (default "2000-01-01").
    end : str or None
        Last period (default: today).
    use_cache : bool
        Use locally cached raw data (default True). Set False to force a fresh
        download from FRED — necessary on first run or after a long gap.

    Returns
    -------
    (monthly_df, quarterly_df)
        monthly_df   : shape (n_months, 11), monthly indicators, freq='MS'
        quarterly_df : shape (n_quarters, 1), GDPC1 log-diffs, freq='QS'
    """
    from src.data.fred_loader import fetch_series, transform_series

    # Pull data starting one period before 'start' so the first log-diff is defined
    pre_start_m = (pd.Timestamp(start) - pd.DateOffset(months=2)).strftime("%Y-%m-%d")
    pre_start_q = (pd.Timestamp(start) - pd.DateOffset(months=4)).strftime("%Y-%m-%d")

    # ── Monthly panel ──────────────────────────────────────────────────────────
    panel = build_panel(start=pre_start_m, end=end, use_cache=use_cache)

    monthly_ids = [s for s, m in SERIES_CATALOG.items()
                   if m["freq"] == "M" and s in panel.columns]

    monthly_df = panel[monthly_ids].loc[start:]
    # Drop leading rows where every single monthly series is NaN
    first_valid = monthly_df.dropna(how="all").index[0]
    monthly_df  = monthly_df.loc[first_valid:]

    # ── Quarterly GDP — on its natural quarterly-frequency index ───────────────
    # DynamicFactorMQ requires endog_quarterly to have freq starting with 'Q'.
    # We fetch GDPC1 raw (quarterly dates from FRED: 2000-01-01, 2000-04-01…),
    # apply log_diff, then set the quarterly frequency explicitly.
    gdp_raw    = fetch_series("GDPC1", start=pre_start_q, end=end, use_cache=use_cache)
    gdp_growth = transform_series(gdp_raw, "log_diff").dropna()

    # Trim to the requested start date
    gdp_growth = gdp_growth.loc[start:]

    # Convert to a proper quarterly-frequency DatetimeIndex (freq='QS').
    # FRED dates quarterly series at the quarter start (Jan, Apr, Jul, Oct).
    # to_timestamp() with no argument returns the first day of each period,
    # which is exactly quarter-start. ("QS" is invalid as a to_timestamp arg
    # in pandas 3.x — pass nothing instead.)
    gdp_growth.index = pd.DatetimeIndex(gdp_growth.index).to_period("Q").to_timestamp()
    gdp_growth = gdp_growth.asfreq("QS")   # sets freq attribute on the index

    # Extend to the current quarter so the unreleased period appears as NaN.
    # Without this, extract_nowcast would find no missing quarter to predict.
    today = pd.Timestamp.today()
    current_qtr_start = today.to_period("Q").to_timestamp()
    if gdp_growth.index[-1] < current_qtr_start:
        extension = pd.Series(
            [float("nan")],
            index=pd.DatetimeIndex([current_qtr_start]),
            name="GDPC1",
        )
        gdp_growth = pd.concat([gdp_growth, extension])
        # pd.concat drops the freq attribute; restore it explicitly
        gdp_growth.index.freq = pd.tseries.frequencies.to_offset("QS")

    quarterly_df = pd.DataFrame({"GDPC1": gdp_growth})

    logger.info(
        "Panel: %d months (%s→%s), %d monthly series | "
        "%d quarters (%s→%s), 1 quarterly series",
        len(monthly_df),
        monthly_df.index[0].strftime("%Y-%m"),
        monthly_df.index[-1].strftime("%Y-%m"),
        len(monthly_ids),
        len(quarterly_df),
        quarterly_df.index[0].strftime("%Y-%m"),
        quarterly_df.index[-1].strftime("%Y-%m"),
    )

    return monthly_df, quarterly_df


def fit_dfm(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
    n_factors: int = 1,
    factor_orders: int = 1,
    maxiter: int = 1000,
) -> Any:
    """
    Fit a Dynamic Factor Model using the EM algorithm.

    What "EM algorithm" means:
      EM stands for Expectation-Maximization. It iterates two steps:
        E-step: given the current model parameters, use the Kalman filter to
                estimate the hidden factor at each time period.
        M-step: given those factor estimates, re-fit all the loading coefficients
                (how much each indicator responds to the factor) to maximize
                the likelihood.
      This repeats until the parameters stop changing (convergence).

    What the model looks like mathematically:
      Observation equation (for each monthly indicator i):
          x_{it} = λ_i × F_t + ε_{it},    ε_{it} ~ AR(1)
      State equation (factor dynamics):
          F_t = φ × F_{t-1} + η_t,         η_t ~ N(0, 1)
      Quarterly GDP (temporal aggregation over 3 monthly factor values):
          GDPC1_q ≈ sum over months in quarter of λ_GDP × F_t + ε_{GDP,t}

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly indicators from prepare_dfm_data().
    quarterly_df : pd.DataFrame
        Quarterly GDP from prepare_dfm_data().
    n_factors : int
        Number of latent factors (default 1 for the Phase 2 baseline).
        Phase 3 will use multiple blocked factors.
    factor_orders : int
        AR order of the latent factor (1 = AR(1), standard choice).
    maxiter : int
        Maximum EM iterations (default 1000). The algorithm stops early if
        the log-likelihood improvement is smaller than the tolerance.

    Returns
    -------
    DynamicFactorMQResults
        Fitted results object. Key attributes:
          .fittedvalues           DataFrame of smoothed fitted values
          .factors.smoothed       Time series of smoothed factor estimates
          .llf                    Log-likelihood at convergence
          .summary()              Full parameter table
    """
    logger.info(
        "Fitting DFM: %d factor(s), AR(%d) factor dynamics, maxiter=%d",
        n_factors, factor_orders, maxiter,
    )

    mod = DynamicFactorMQ(
        endog=monthly_df,
        endog_quarterly=quarterly_df,
        factors=n_factors,
        factor_orders=factor_orders,
        idiosyncratic_ar1=True,   # Each series has its own AR(1) idiosyncratic noise
        standardize=True,         # Internally z-score all series (zero mean, unit variance)
    )

    results = mod.fit_em(disp=False, maxiter=maxiter, tolerance=1e-6)

    logger.info("EM finished. Log-likelihood: %.2f", results.llf)

    return results


def extract_nowcast(
    results: Any,
    quarterly_df: pd.DataFrame,
) -> dict | None:
    """
    Extract the current-quarter GDP growth nowcast from a fitted DFM.

    What "nowcast" means here:
      We look at the last quarter in the panel where actual GDP hasn't been
      released yet (GDPC1 is NaN). The model's smoothed prediction for that
      quarter-end month is the nowcast: the model's best estimate of GDP growth
      given all the monthly indicators released so far.

    How the model produces this:
      The Kalman smoother runs forward through all the monthly data, updating
      its estimate of the hidden factor as each new indicator arrives. Where
      GDP is missing (current quarter), the factor estimate is driven entirely
      by the monthly indicators. The observation equation then translates the
      factor back into a GDP growth prediction.

    Parameters
    ----------
    results : DynamicFactorMQResults
        Fitted DFM from fit_dfm().
    quarterly_df : pd.DataFrame
        The same quarterly DataFrame passed to fit_dfm(), so we know where
        GDP actuals are vs. where they are NaN (= "nowcast" periods).

    Returns
    -------
    dict or None
        None if there is no missing GDP quarter in the panel.
        Otherwise a dict with:
            'date'             : the quarter-end date of the nowcast (e.g., 2026-06-01)
            'quarter'          : human-readable label (e.g., "2026 Q2")
            'nowcast_qtr_pct'  : model's predicted quarterly log-diff (%)
            'nowcast_ann_pct'  : same × 4 (annualized, comparable to GDPNow)
    """
    gdp_col = "GDPC1"
    if gdp_col not in quarterly_df.columns:
        raise ValueError(f"Expected '{gdp_col}' in quarterly_df. Got: {list(quarterly_df.columns)}")

    gdp_actual = quarterly_df[gdp_col]

    # quarterly_df now has a proper quarterly DatetimeIndex (freq='QS'), so
    # every row IS a quarter — no need to filter for quarter-end months.
    missing_actuals = gdp_actual[gdp_actual.isna()]

    if missing_actuals.empty:
        logger.warning("No missing GDP quarters found — no nowcast to compute.")
        return None

    # The nowcast target is the LAST quarter with no released actual
    nowcast_date  = missing_actuals.index[-1]
    quarter_num   = (nowcast_date.month - 1) // 3 + 1
    quarter_label = f"{nowcast_date.year} Q{quarter_num}"

    # fittedvalues contains the model's smoothed predictions for all periods,
    # including quarters where the actual GDPC1 is NaN (the nowcast).
    fv = results.fittedvalues

    if gdp_col not in fv.columns:
        raise ValueError(
            f"'{gdp_col}' not found in fittedvalues. "
            f"Available columns: {list(fv.columns)}"
        )

    # fittedvalues may use a different date convention than quarterly_df
    # (e.g., quarter-end vs quarter-start). Try direct lookup first, then
    # search within a 3-month window.
    nowcast_val = fv[gdp_col].get(nowcast_date, float("nan"))

    if pd.isna(nowcast_val):
        # Fall back: look at the last non-NaN fitted GDP value
        last_valid = fv[gdp_col].dropna()
        if last_valid.empty:
            logger.error("No valid fitted GDP values found in fittedvalues")
            return None
        nowcast_val   = last_valid.iloc[-1]
        nowcast_date  = last_valid.index[-1]
        quarter_num   = (nowcast_date.month - 1) // 3 + 1
        quarter_label = f"{nowcast_date.year} Q{quarter_num} (last available)"
        logger.warning("Nowcast date mismatch; used last fittedvalue at %s", nowcast_date.date())

    return {
        "date":            nowcast_date,
        "quarter":         quarter_label,
        "nowcast_qtr_pct": float(nowcast_val),
        "nowcast_ann_pct": float(nowcast_val) * 4,
    }


def print_dfm_nowcast(nowcast: dict, ar1_nowcast: dict | None = None) -> None:
    """Print the DFM nowcast alongside the AR(1) baseline for comparison."""
    print()
    print("=" * 55)
    print(f"DFM Nowcast — {nowcast['quarter']}")
    print("=" * 55)
    print(f"  Quarterly  : {nowcast['nowcast_qtr_pct']:>+.4f}%")
    print(f"  Annualized : {nowcast['nowcast_ann_pct']:>+.2f}%")
    if ar1_nowcast:
        print()
        print(f"  AR(1) baseline (annualized): {ar1_nowcast['nowcast_ann_pct']:>+.2f}%")
        diff = nowcast["nowcast_ann_pct"] - ar1_nowcast["nowcast_ann_pct"]
        print(f"  DFM vs AR(1) difference    : {diff:>+.2f} pp")
    print()
    print("  Compare to Atlanta Fed GDPNow:")
    print("  https://www.atlantafed.org/cprcd/macroeconomic-research/gdpnow")
    print("=" * 55)
