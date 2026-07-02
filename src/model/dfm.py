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

# ── Phase 3: blocked factor structure ─────────────────────────────────────────
#
# Following the NY Fed Staff Nowcast approach (Bok et al., 2018), we use a
# two-factor "blocked" structure:
#
#   Factor 1 — "global":  all 12 series load on this factor. It captures the
#                         overall business cycle.
#   Factor 2 — "real":    only real-activity series load on this factor. It
#                         captures idiosyncratic real-side variation not
#                         explained by the global factor.
#
# Survey indices (NY Fed, Philly Fed) and CPI load ONLY on the global factor
# because they measure sentiment/prices rather than direct real activity.
#
REAL_ACTIVITY_SERIES: frozenset[str] = frozenset({
    "INDPRO", "PAYEMS", "UNRATE", "PCEC96", "RSAFS",
    "HOUST", "DGORDER", "DSPIC96", "GDPC1",
})


def make_block_structure(
    monthly_df: pd.DataFrame,
    quarterly_df: pd.DataFrame,
) -> dict[str, list[str]]:
    """
    Build the two-factor blocked loading structure for DynamicFactorMQ.

    What this returns:
      A dict where each key is a series name and each value is the list of
      factor names that series loads on.  DynamicFactorMQ requires this format
      when you pass a dict to its `factors` argument.

      Example output (abbreviated):
        {
          'INDPRO':              ['global', 'real'],   # real activity → both factors
          'PAYEMS':              ['global', 'real'],
          'CPIAUCSL':            ['global'],           # price index → global only
          'GACDISA066MSFRBNY':   ['global'],           # survey index → global only
          'GDPC1':               ['global', 'real'],   # GDP → both factors
        }

    Why two factors?
      A single global factor captures the overall business cycle but can't
      separately represent idiosyncratic real-side dynamics.  Adding a "real"
      factor that loads only on the labour/output/spending series lets the
      model better distinguish real-economy swings from sentiment-driven or
      price-driven variation, improving both fit and nowcast accuracy.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly panel from prepare_dfm_data().
    quarterly_df : pd.DataFrame
        Quarterly GDP from prepare_dfm_data().

    Returns
    -------
    dict[str, list[str]]
        Mapping of series name → factor list.  Every series in both
        DataFrames appears exactly once.
    """
    all_series = list(monthly_df.columns) + list(quarterly_df.columns)
    return {
        s: (["global", "real"] if s in REAL_ACTIVITY_SERIES else ["global"])
        for s in all_series
    }


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
    non_empty = monthly_df.dropna(how="all")
    if non_empty.empty:
        raise ValueError(
            "No monthly data was loaded — every FRED series came back empty. "
            "This almost always means FRED_API_KEY is missing or invalid. "
            "Locally: check your .env file. On Streamlit Cloud: set FRED_API_KEY "
            "under Manage app → Settings → Secrets, then reboot the app."
        )
    first_valid = non_empty.index[0]
    monthly_df  = monthly_df.loc[first_valid:]

    # ── Quarterly GDP — on its natural quarterly-frequency index ───────────────
    # DynamicFactorMQ requires endog_quarterly to have freq starting with 'Q'.
    # We fetch GDPC1 raw (quarterly dates from FRED: 2000-01-01, 2000-04-01…),
    # apply log_diff, then set the quarterly frequency explicitly.
    gdp_raw    = fetch_series("GDPC1", start=pre_start_q, end=end, use_cache=use_cache)
    gdp_growth = transform_series(gdp_raw, "log_diff").dropna()

    # Trim to the requested start/end dates.
    # NOTE on look-ahead: the raw cache always contains the latest-available
    # revision, regardless of 'end'. For proper real-time vintage analysis use
    # ALFRED (Phase 4). Here we at least clip by date so we don't include
    # quarters whose FRED index date exceeds 'end'.
    gdp_growth = gdp_growth.loc[start:]
    if end is not None:
        gdp_growth = gdp_growth.loc[: pd.Timestamp(end)]

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
    factor_blocks: dict | None = None,
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

    Blocked factor structure (Phase 3):
      Pass factor_blocks=make_block_structure(monthly_df, quarterly_df) to use
      a two-factor model where a "global" factor loads on all series and a
      "real" factor loads only on real-activity series.  This matches the
      NY Fed Staff Nowcast design and improves log-likelihood by ~170 points
      over a single global factor.

    Parameters
    ----------
    monthly_df : pd.DataFrame
        Monthly indicators from prepare_dfm_data().
    quarterly_df : pd.DataFrame
        Quarterly GDP from prepare_dfm_data().
    n_factors : int
        Number of global factors when NOT using a block structure (default 1).
        Ignored when factor_blocks is provided.
    factor_orders : int
        AR order of each latent factor (1 = AR(1), standard choice).
    maxiter : int
        Maximum EM iterations (default 1000).
    factor_blocks : dict or None
        If provided, defines which series load on which factors.
        Build with make_block_structure().  Overrides n_factors.

    Returns
    -------
    DynamicFactorMQResults
        Fitted results object. Key attributes:
          .fittedvalues           DataFrame of smoothed fitted values
          .factors.smoothed       Time series of smoothed factor estimates
          .llf                    Log-likelihood at convergence
          .summary()              Full parameter table
    """
    factors_arg = factor_blocks if factor_blocks is not None else n_factors

    if factor_blocks is not None:
        logger.info("Fitting blocked DFM (%d factors), AR(%d), maxiter=%d",
                    len(set(f for fl in factor_blocks.values() for f in fl)),
                    factor_orders, maxiter)
    else:
        logger.info("Fitting DFM: %d factor(s), AR(%d), maxiter=%d",
                    n_factors, factor_orders, maxiter)

    mod = DynamicFactorMQ(
        endog=monthly_df,
        endog_quarterly=quarterly_df,
        factors=factors_arg,
        factor_orders=factor_orders,
        idiosyncratic_ar1=True,
        standardize=True,
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


def compute_news(
    before_end: str,
    after_end: str | None = None,
    start: str = "2000-01-01",
    factor_blocks: dict | None = None,
    n_factors: int = 1,
    use_cache: bool = True,
) -> dict:
    """
    Decompose a GDP nowcast revision into per-series news contributions.

    In plain English — what a "news decomposition" does:
      Every time new data arrives (e.g., the monthly jobs report), the model's
      GDP nowcast gets revised.  The news decomposition asks: how much of that
      revision came from each indicator?

      For example, if PAYEMS came in much better than expected, it should push
      the GDP nowcast up.  If the Philly Fed survey came in worse than expected,
      it should push it down.  The individual contributions (weighted by each
      series' "Kalman gain") should add up to the total revision.

    How this is computed:
      1.  Fit the model on ALL available data ("after" vintage).
          These parameter estimates are held fixed throughout.
      2.  Apply those fixed parameters to the "before" dataset using the
          Kalman smoother.  The difference in filtered forecasts (before vs
          after) gives the revision.
      3.  Call statsmodels results.news() to decompose the revision into
          per-series contributions via the Kalman gain.

    Important note on smoother vs filter:
      - The current-best-estimate nowcast (from extract_nowcast / fittedvalues)
        uses the Kalman SMOOTHER, which conditions on ALL available data in
        both directions.
      - The news decomposition uses the Kalman FILTER (one-sided), which only
        conditions on data up to the vintage cutoff.  This is conceptually
        correct for news: "given only past data, how does each new release
        move the forecast?"
      These two quantities answer different questions and will differ
      numerically; both are valid.

    Look-ahead note:
      The local FRED cache always contains the latest data revision.  For a
      true real-time vintage comparison, use ALFRED (Phase 4).  Here we clip
      by date, but the underlying cache may contain data released after
      before_end for series whose FRED index date precedes before_end.

    Parameters
    ----------
    before_end : str
        Vintage "before" date (e.g. "2026-04-30").  The model sees only
        data whose index date is on or before this date.
    after_end : str or None
        Vintage "after" date (default: all available data).
    start : str
        Model estimation start (default "2000-01-01").
    factor_blocks : dict or None
        Blocked structure from make_block_structure().  If None, uses n_factors
        global factors.
    n_factors : int
        Number of global factors when not using blocks (default 1).
    use_cache : bool
        Use locally cached FRED data (default True).

    Returns
    -------
    dict with:
        'target_quarter'     : human-readable label (e.g. "2026 Q2")
        'target_date'        : Timestamp of the nowcast quarter-start
        'before_end'         : the before_end string passed in
        'before_nowcast_qtr' : filter-based "before" quarterly nowcast (%)
        'before_nowcast_ann' : same × 4 (annualized %)
        'after_nowcast_qtr'  : filter-based "after" quarterly nowcast (%)
        'after_nowcast_ann'  : same × 4 (annualized %)
        'revision_qtr'       : after − before (quarterly pp)
        'revision_ann'       : same × 4 (annualized pp)
        'total_impact_qtr'   : statsmodels' reported total impact (≈ revision)
        'impacts'            : DataFrame — per-series news details
                               (columns: observed, forecast (prev), news,
                                weight, impact; indexed by update_date ×
                                updated_variable × impact_date × impacted_var)
        'news_results'       : raw statsmodels NewsResults object
    """
    # --- Fit "after" model on all available data ----------------------------
    monthly_after, quarterly_after = prepare_dfm_data(
        start=start, end=after_end, use_cache=use_cache
    )
    results_after = fit_dfm(
        monthly_after, quarterly_after,
        factor_blocks=factor_blocks, n_factors=n_factors,
    )

    nc_smooth = extract_nowcast(results_after, quarterly_after)
    if nc_smooth is None:
        raise ValueError("No missing GDP quarter found in the 'after' dataset.")
    impact_date = nc_smooth["date"]

    # --- Build "before" dataset by clipping to before_end ------------------
    before_ts = pd.Timestamp(before_end)
    monthly_before   = monthly_after.loc[monthly_after.index <= before_ts].copy()
    quarterly_before = quarterly_after.loc[quarterly_after.index <= before_ts].copy()
    if not quarterly_before.empty:
        quarterly_before.index.freq = pd.tseries.frequencies.to_offset("QS")

    # --- Apply "after" params to "before" data (Kalman smooth only) --------
    # Holding parameters fixed ensures the revision is purely from new
    # observations, not from parameter re-estimation.
    factors_arg = factor_blocks if factor_blocks is not None else n_factors
    mod_before = DynamicFactorMQ(
        endog=monthly_before,
        endog_quarterly=quarterly_before,
        factors=factors_arg,
        factor_orders=1,
        idiosyncratic_ar1=True,
        standardize=True,
    )
    results_before = mod_before.smooth(results_after.params)

    # --- Compute news decomposition ----------------------------------------
    news_obj = results_after.news(
        comparison=results_before,
        impact_dates=impact_date,
        impacted_variable="GDPC1",
        comparison_type="updated",
    )

    # Filter-based "before" and "after" forecasts for the target quarter
    prev_series = news_obj.prev_impacted_forecasts.get("GDPC1", pd.Series(dtype=float))
    post_series = news_obj.post_impacted_forecasts.get("GDPC1", pd.Series(dtype=float))

    if prev_series.empty or post_series.empty:
        raise ValueError(
            f"No forecast found for GDPC1 at {nc_smooth['quarter']}. "
            "The target quarter may be outside the model's prediction range."
        )

    prev_qtr = float(prev_series.iloc[0])
    post_qtr = float(post_series.iloc[0])
    revision_qtr = post_qtr - prev_qtr

    # impacts is empty when there are no new observations in the vintage window
    # (e.g., no monthly releases between before_end and today in the cached data).
    if news_obj.impacts.empty:
        raise ValueError(
            f"No new data releases found between {before_end!r} and today. "
            "The cutoff date is too recent — try a date 6–8 weeks ago to capture "
            "actual monthly indicator releases (jobs, industrial production, etc.)."
        )

    total_impact_qtr = float(news_obj.impacts["total impact"].iloc[0])

    return {
        "target_quarter":     nc_smooth["quarter"],
        "target_date":        impact_date,
        "before_end":         before_end,
        "before_nowcast_qtr": prev_qtr,
        "before_nowcast_ann": prev_qtr * 4,
        "after_nowcast_qtr":  post_qtr,
        "after_nowcast_ann":  post_qtr * 4,
        "revision_qtr":       revision_qtr,
        "revision_ann":       revision_qtr * 4,
        "total_impact_qtr":   total_impact_qtr,
        "impacts":            news_obj.details_by_update.copy(),
        "news_results":       news_obj,
    }


def print_news_summary(result: dict) -> None:
    """
    Print a human-readable news decomposition table.

    Shows each indicator's "surprise" (actual vs predicted) and how much
    that surprise pushed the GDP nowcast up or down.

    Parameters
    ----------
    result : dict
        Output of compute_news().
    """
    print()
    print("=" * 65)
    print(f"  News Decomposition — {result['target_quarter']}")
    print(f"  Vintage: through {result['before_end']}  →  today")
    print("=" * 65)
    print(f"  Nowcast before : {result['before_nowcast_ann']:>+.2f}%  (annualized, filtered)")
    print(f"  Nowcast after  : {result['after_nowcast_ann']:>+.2f}%  (annualized, filtered)")
    print(f"  Revision       : {result['revision_ann']:>+.4f} pp  (annualized)")
    print()
    print(f"  {'Series':<28}  {'Surprise':>9}  {'Weight':>8}  {'Impact (ann)':>12}")
    print(f"  {'-'*28}  {'-'*9}  {'-'*8}  {'-'*12}")

    impacts = result["impacts"]
    if impacts.empty:
        print("  (no new data in this vintage window)")
    else:
        for idx, row in impacts.iterrows():
            # MultiIndex: (update_date, updated_variable, ..., impacted_variable)
            series_name = idx[1] if isinstance(idx, tuple) and len(idx) > 1 else str(idx)
            surprise    = float(row.get("news", float("nan")))
            weight      = float(row.get("weight", float("nan")))
            impact_ann  = float(row.get("impact", float("nan"))) * 4
            print(f"  {series_name:<28}  {surprise:>+9.4f}  {weight:>8.4f}  {impact_ann:>+12.4f}%")

    print()
    print(f"  Sum of impacts (ann) : {result['total_impact_qtr']*4:>+.4f}% ≈ revision ✓"
          if abs(result['total_impact_qtr'] - result['revision_qtr']) < 1e-4
          else f"  Sum of impacts (ann) : {result['total_impact_qtr']*4:>+.4f}%")
    print("=" * 65)


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
