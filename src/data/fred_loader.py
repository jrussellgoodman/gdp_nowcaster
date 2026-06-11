"""
fred_loader.py — Download, cache, and transform FRED series for the GDP nowcasting model.

What this module does, in plain English:
  Economic data like GDP or unemployment comes from the Federal Reserve's FRED
  database. Before we can plug this data into our statistical model, we need to:

    1. Download it (using the fredapi library and your FRED API key from .env).
    2. Save a local copy ("cache") so we don't re-download every time we run.
    3. Transform it to be "stationary" — meaning its statistical properties
       don't change over time. Raw GDP grows indefinitely, which would break
       our model. Taking log-differences (≈ percent changes) removes the trend.
    4. Arrange everything on a common monthly timeline. Quarterly GDP sits on
       that monthly timeline with NaN gaps — the Kalman filter in DynamicFactorMQ
       handles those gaps automatically.

The main entry point is build_panel(), which returns a DataFrame ready to pass
to statsmodels DynamicFactorMQ.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()

logger = logging.getLogger(__name__)

# ── Cache directory ────────────────────────────────────────────────────────────
# Raw (pre-transformation) data is saved here as Parquet files.
# This folder is gitignored; it's regenerated automatically on first run.
RAW_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# ── Series catalog ─────────────────────────────────────────────────────────────
#
# Defines every series we download, its frequency, and how to transform it.
#
# TRANSFORM TYPES, explained:
#
#   "log_diff"  100 × (ln(x_t) − ln(x_{t−1}))
#               Approximates the percent change from one period to the next.
#               Example: GDP grew from $20,000B to $20,100B → log_diff ≈ 0.50%
#               Used for series that trend upward over time (output, prices,
#               employment levels). This removes the trend and makes them
#               stationary.
#
#   "diff"      x_t − x_{t−1}
#               Simple first difference (subtract previous value).
#               Used for series already expressed as rates (e.g., unemployment
#               rate is already in %, so "percent change of a percent" is odd).
#
#   "level"     x_t  (no change)
#               Used for survey indices designed to be stationary by construction
#               (e.g., Empire State Manufacturing Survey — positive = expanding,
#               negative = contracting, mean near zero).
#
# DATA RULES ENFORCED HERE:
#   - GDPC1 is real GDP (chained 2017 $). "GDP" alone is nominal — wrong series.
#   - PCEC96 is monthly real PCE. PCECC96 is quarterly — we use the monthly one.
#   - ISM/PMI (NAPM*) series were removed from FRED in 2014; we use Fed survey
#     substitutes (GACDISA066MSFRBNY, GACDFSA066MSFRBPHI) instead.
#
SERIES_CATALOG: dict[str, dict] = {
    # ── Quarterly ──────────────────────────────────────────────────────────────
    "GDPC1": {
        "name": "Real GDP (billions, chained 2017 $)",
        "freq": "Q",
        "transform": "log_diff",
        # Quarterly log_diff ≈ quarterly % growth.
        # To annualize for display: multiply by 4. But the model uses raw
        # quarterly values — annualizing is only for human-readable output.
    },
    # ── Monthly: Output ────────────────────────────────────────────────────────
    "INDPRO": {
        "name": "Industrial Production Index",
        "freq": "M",
        "transform": "log_diff",
    },
    # ── Monthly: Labor market ─────────────────────────────────────────────────
    "PAYEMS": {
        "name": "Nonfarm Payroll Employment (thousands)",
        "freq": "M",
        "transform": "log_diff",
    },
    "UNRATE": {
        "name": "Unemployment Rate (%)",
        "freq": "M",
        "transform": "diff",  # Already in percent — take first differences
    },
    # ── Monthly: Consumption ───────────────────────────────────────────────────
    "PCEC96": {
        "name": "Real Personal Consumption Expenditures (monthly)",
        "freq": "M",
        "transform": "log_diff",
        # PCEC96 = monthly. Do NOT use PCECC96, which is quarterly.
    },
    "RSAFS": {
        "name": "Advance Retail and Food Services Sales (millions $)",
        "freq": "M",
        "transform": "log_diff",
    },
    # ── Monthly: Housing ───────────────────────────────────────────────────────
    "HOUST": {
        "name": "Housing Starts (thousands, seasonally adjusted annual rate)",
        "freq": "M",
        "transform": "log_diff",
    },
    # ── Monthly: Orders ────────────────────────────────────────────────────────
    "DGORDER": {
        "name": "Manufacturers' New Orders: Durable Goods (millions $)",
        "freq": "M",
        "transform": "log_diff",
    },
    # ── Monthly: Income ────────────────────────────────────────────────────────
    "DSPIC96": {
        "name": "Real Disposable Personal Income",
        "freq": "M",
        "transform": "log_diff",
    },
    # ── Monthly: Prices ────────────────────────────────────────────────────────
    "CPIAUCSL": {
        "name": "CPI All Urban Consumers (seasonally adjusted)",
        "freq": "M",
        "transform": "log_diff",
    },
    # ── Monthly: Survey indices ────────────────────────────────────────────────
    # ISM/PMI (NAPM*) series were removed from FRED in 2014 due to licensing.
    # The NY Fed and Philly Fed publish comparable business-conditions indices.
    "GACDISA066MSFRBNY": {
        "name": "NY Fed Empire State Manufacturing Survey (general conditions)",
        "freq": "M",
        "transform": "level",  # Diffusion index: 0 = neutral, ±100 = all expanding/contracting
    },
    "GACDFSA066MSFRBPHI": {
        "name": "Philly Fed Business Outlook Survey (general activity)",
        "freq": "M",
        "transform": "level",  # Same scale
    },
}


def _raw_cache_path(series_id: str) -> Path:
    """Return the local Parquet file path for a raw (pre-transform) series."""
    return RAW_CACHE_DIR / f"{series_id}.parquet"


def fetch_series(
    series_id: str,
    start: str = "1990-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.Series:
    """
    Download a FRED series and return it as a pandas Series.

    What this does:
      If a local cache file exists and use_cache=True, we read from disk instead
      of calling the API. This avoids unnecessary network requests and keeps the
      model fast to re-run. We cache the RAW data (before transformation) so we
      can change the transformation without re-downloading.

    Parameters
    ----------
    series_id : str
        FRED series ID, e.g. "GDPC1" or "PAYEMS".
    start : str
        Start date in "YYYY-MM-DD" format (default 1990-01-01).
    end : str or None
        End date (default: today).
    use_cache : bool
        Read from local cache if available (default True).

    Returns
    -------
    pd.Series
        Raw, untransformed series with DatetimeIndex.
    """
    cache_path = _raw_cache_path(series_id)
    req_start  = pd.Timestamp(start)
    req_end    = pd.Timestamp(end) if end else pd.Timestamp.today()

    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path).squeeze()
        # Validate the cache covers the requested window.
        # "Covers start": cache begins at or before the requested start date.
        # "Recent enough": cache ends within 60 days of the requested end
        #   (prevents using an old cache that's missing recent releases).
        covers_start  = cached.index[0] <= req_start + pd.DateOffset(days=31)
        covers_recent = cached.index[-1] >= req_end - pd.DateOffset(days=60)
        if covers_start and covers_recent:
            logger.info("Loading %s from cache", series_id)
            return cached.loc[req_start:]
        logger.info("Cache for %s is outdated or too narrow; re-downloading", series_id)

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not found. "
            "Make sure it is set in your .env file (never hard-code it)."
        )

    logger.info("Fetching %s from FRED API (start=%s, end=%s)", series_id, start, end)
    fred = Fred(api_key=api_key)
    raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    raw.name = series_id

    raw.to_frame().to_parquet(cache_path)
    logger.info("Cached %s to %s", series_id, cache_path)

    return raw


def transform_series(raw: pd.Series, transform: str) -> pd.Series:
    """
    Apply a stationarity transformation to a raw economic series.

    Why we transform:
      Statistical models like the DFM assume the data is "stationary" — that
      its mean and variance don't change over time. Most raw economic series
      are NOT stationary (e.g., real GDP has been growing for decades). If we
      fed a trending series into the model, the factor estimates would be
      dominated by the trend rather than the business-cycle signal we care about.

      The standard fix is to take growth rates:
        - Log-difference of GDP ≈ GDP growth rate (quarterly % change).
        - Log-difference of employment ≈ monthly % change in jobs.
        - First difference of unemployment rate ≈ monthly change in pp.

    Parameters
    ----------
    raw : pd.Series
        Untransformed series (e.g., level of GDP in billions of dollars).
    transform : str
        "log_diff", "diff", or "level". See SERIES_CATALOG docstring above.

    Returns
    -------
    pd.Series
        Transformed series. The first value is NaN after any differencing
        (there is no previous period to subtract from the first observation).
    """
    if transform == "log_diff":
        # 100 × (ln(x_t) − ln(x_{t−1})) ≈ percent change.
        # The 100× puts values in percent units (e.g., 0.30 = 0.30%/month).
        return 100.0 * np.log(raw).diff()

    elif transform == "diff":
        return raw.diff()

    elif transform == "level":
        return raw.copy()

    else:
        raise ValueError(
            f"Unknown transform '{transform}'. Choose 'log_diff', 'diff', or 'level'."
        )


def _align_to_monthly(
    series: pd.Series,
    freq_code: str,
    monthly_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Place a (possibly quarterly) transformed series onto a monthly date grid.

    What this does:
      DynamicFactorMQ needs a single DataFrame where every row is a month.
      Monthly series: straightforward — one value per month.
      Quarterly series: FRED dates Q1 at January 1st, but DynamicFactorMQ
        expects the value at the LAST month of the quarter (March = end of Q1).
        We shift the FRED dates forward by 2 months, then reindex to the monthly
        grid. The two preceding months within the quarter (January, February)
        are NaN — the Kalman filter will fill those in probabilistically.

    Shift logic (quarterly only):
      FRED date → panel date
      2020-01-01 (Q1) → 2020-03-01  (+2 months)
      2020-04-01 (Q2) → 2020-06-01  (+2 months)
      2020-07-01 (Q3) → 2020-09-01  (+2 months)
      2020-10-01 (Q4) → 2020-12-01  (+2 months)

    Parameters
    ----------
    series : pd.Series
        Transformed series (monthly or quarterly DatetimeIndex).
    freq_code : str
        "M" for monthly, "Q" for quarterly.
    monthly_index : pd.DatetimeIndex
        Target monthly grid (freq="MS").

    Returns
    -------
    pd.Series
        Series reindexed to monthly_index; NaN where no observation exists.
    """
    series = series.copy()
    # Normalize to month-start dates (handles any within-month date offsets FRED uses).
    # to_timestamp() with no argument returns the first day of each period (month start).
    # "MS" is not a valid freq argument here in pandas 3.x — use bare to_timestamp().
    series.index = series.index.to_period("M").to_timestamp()

    if freq_code == "Q":
        # Shift from quarter-start to quarter-end (add 2 months)
        series.index = series.index + pd.DateOffset(months=2)

    return series.reindex(monthly_index)


def build_panel(
    start: str = "1990-01-01",
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build the complete mixed-frequency data panel for the DFM.

    What you get back:
      A pandas DataFrame where:
        - Each row is one month (DatetimeIndex, freq='MS').
        - Each column is one transformed economic indicator.
        - Monthly series: a value every row.
        - GDPC1 (quarterly): a value only in March, June, September, December;
          NaN in the other months. DynamicFactorMQ handles those NaNs.

    Why mixed frequency matters:
      GDP is released only quarterly (~80 days after the quarter ends). But
      monthly data (jobs, retail sales, industrial production) arrives every
      month. The Kalman filter inside DynamicFactorMQ uses the monthly signals
      to infer GDP before the official release — that is the nowcast.

    Parameters
    ----------
    start : str
        First month of the panel (default "1990-01-01").
    end : str or None
        Last month (default: today).
    use_cache : bool
        If True, use locally cached raw data when available (default True).
        Set False to force a fresh download from FRED.

    Returns
    -------
    pd.DataFrame
        Shape: (n_months, n_series). Index is monthly DatetimeIndex (freq='MS').
        Column names are FRED series IDs (e.g., "GDPC1", "PAYEMS").
    """
    if end is None:
        end = pd.Timestamp.today().strftime("%Y-%m-%d")

    monthly_index = pd.date_range(start=start, end=end, freq="MS")
    columns: dict[str, pd.Series] = {}

    for series_id, meta in SERIES_CATALOG.items():
        logger.info("Processing %s — %s", series_id, meta["name"])
        try:
            raw = fetch_series(series_id, start=start, end=end, use_cache=use_cache)
            transformed = transform_series(raw, meta["transform"])
            aligned = _align_to_monthly(transformed, meta["freq"], monthly_index)
            columns[series_id] = aligned
        except Exception as exc:
            logger.warning("Could not load %s: %s", series_id, exc)
            columns[series_id] = pd.Series(np.nan, index=monthly_index, name=series_id)

    panel = pd.DataFrame(columns, index=monthly_index)
    panel.index.name = "date"
    return panel
