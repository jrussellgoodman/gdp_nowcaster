"""
vintage.py — ALFRED real-time vintage data for GDP nowcast backtesting.

What "ALFRED" means, and why it matters:
  The Federal Reserve's standard FRED database always shows the LATEST
  revision of each series. But in a real-time nowcasting model, you should
  only use data that was available AT THE TIME of the prediction.

  For example, January 2020 nonfarm payrolls published on February 7, 2020
  showed 225,000 jobs added. That number was later revised down to 185,000.
  If you backtest using today's revised data, you're cheating — your 2020
  model would have "seen" better data than was actually available.

  ALFRED (Archival FRED) stores every historical vintage of every series.
  By setting realtime_start = realtime_end = evaluation_date in the FRED API
  call, we get "what FRED showed on that exact date." This is the standard
  approach in real-time forecasting literature (Croushore & Stark, 2001).

  Without ALFRED vintages, your backtest has look-ahead bias. With them,
  each nowcast is a genuinely out-of-sample prediction — the same information
  constraint the model would have faced in practice.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from src.data.fred_loader import (
    SERIES_CATALOG,
    _align_to_monthly,
    transform_series,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Separate cache for ALFRED vintage data (date-stamped, never overwritten).
# The regular data/raw/ cache stores current-vintage data and can be refreshed.
# ALFRED vintages are immutable — what FRED showed on 2020-03-31 will not change.
ALFRED_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "alfred"
ALFRED_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Always fetch ALFRED data from this early date so the cache is comprehensive.
# The `start` argument to fetch_vintage_series only filters the return value —
# it does NOT change the API call or the cache contents. This prevents a bug
# where a smoke-test call with start="2023-01-01" creates a short cache file
# that then silently returns NaN for all pre-2023 rows in a later wider call.
_ALFRED_FETCH_FROM = "1990-01-01"


def _alfred_cache_path(series_id: str, as_of_date: str) -> Path:
    """Deterministic path for a single (series, vintage_date) pair."""
    return ALFRED_CACHE_DIR / f"{series_id}_{as_of_date}.parquet"


def fetch_vintage_series(
    series_id: str,
    as_of_date: str,
    start: str = "1990-01-01",
    use_cache: bool = True,
) -> pd.Series:
    """
    Fetch a FRED series as it was known on as_of_date, using ALFRED vintages.

    What this does under the hood:
      Calls the FRED API with realtime_start = realtime_end = as_of_date.
      This returns the observations that were valid on that exact date —
      the same values that would have appeared on fred.stlouisfed.org on that day.

    Look-ahead prevention:
      The local data/raw/ cache always contains the LATEST revision. If you used
      that for backtesting, you'd inadvertently use post-revision data. This
      function fetches from the separate ALFRED system instead.

    Parameters
    ----------
    series_id : str
        FRED series ID (e.g., "PAYEMS", "GDPC1").
    as_of_date : str
        The vintage date ("YYYY-MM-DD"). Returns data as it existed on this date.
    start : str
        Start date for observations (default "1990-01-01").
    use_cache : bool
        Cache results in data/alfred/ — ALFRED vintages are immutable, so
        caching them is always safe (default True).

    Returns
    -------
    pd.Series
        Raw (untransformed) series with DatetimeIndex, as known on as_of_date.
        May be shorter than a current-vintage fetch — later releases/revisions
        are excluded.

    Raises
    ------
    EnvironmentError
        If FRED_API_KEY is not set.
    ValueError
        If ALFRED returns an empty series (series may not have existed on as_of_date).
    """
    as_of_str = pd.Timestamp(as_of_date).strftime("%Y-%m-%d")
    cache_path = _alfred_cache_path(series_id, as_of_str)

    if use_cache and cache_path.exists():
        cached = pd.read_parquet(cache_path).squeeze()
        # The cache always holds data from _ALFRED_FETCH_FROM; slice to `start`.
        req_start = pd.Timestamp(start)
        logger.debug("ALFRED cache hit: %s @ %s", series_id, as_of_str)
        return cached.loc[cached.index >= req_start]

    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "FRED_API_KEY not set. Check your .env file. Never hard-code this value."
        )

    # Always fetch from _ALFRED_FETCH_FROM so the cache is wide enough for any
    # future call. The `start` arg only filters what we return — not what we cache.
    logger.info("ALFRED fetch: %s @ %s", series_id, as_of_str)
    fred = Fred(api_key=api_key)

    # Retry with exponential backoff on FRED rate-limit errors.
    # FRED allows 120 requests/minute. If we exceed that (12 series × many
    # uncached vintages in quick succession), FRED returns HTTP 429.
    # We wait 65 s on first hit (slightly over 1 minute to let the window
    # reset), then 90 s on the second, before giving up on the third.
    _RETRY_WAITS = [65, 90]
    for attempt, wait in enumerate([0] + _RETRY_WAITS):
        if wait:
            logger.warning(
                "FRED rate limit on %s @ %s — waiting %ds (attempt %d)",
                series_id, as_of_str, wait, attempt,
            )
            time.sleep(wait)
        try:
            raw = fred.get_series(
                series_id,
                observation_start=_ALFRED_FETCH_FROM,
                realtime_start=as_of_str,
                realtime_end=as_of_str,
            )
            break   # success — exit retry loop
        except Exception as exc:
            err = str(exc)
            if "Too Many Requests" in err or "Rate Limit" in err or "429" in err:
                if attempt < len(_RETRY_WAITS):
                    continue     # try again after wait
            raise               # non-rate-limit error or exhausted retries

    if raw is None or len(raw) == 0:
        raise ValueError(
            f"ALFRED returned empty series for {series_id} as of {as_of_str}. "
            "Series may not have existed on this date, or the date is in the future."
        )

    raw.name = series_id
    raw.to_frame().to_parquet(cache_path)
    logger.info("Cached ALFRED %s @ %s → %s", series_id, as_of_str, cache_path)

    # Throttle between successful API calls. FRED allows 120/min; 0.6 s spacing
    # keeps bursts of 12 series at ~2 req/sec = well under the limit.
    time.sleep(0.6)

    req_start = pd.Timestamp(start)
    return raw.loc[raw.index >= req_start]


def assert_no_lookahead(evaluation_date: str, target_quarter_start: str) -> None:
    """
    Assert that evaluation_date cannot contain the target quarter's GDP release.

    How look-ahead bias enters a backtest:
      The BEA advance GDP estimate arrives ~28 days after quarter end (e.g.,
      Q1 GDP is released in late April). If the backtest's evaluation_date
      falls AFTER this release, the ALFRED fetch for GDPC1 could include the
      actual Q1 GDP value — meaning the model sees the "answer" it's supposed
      to be predicting. This is the most dangerous form of look-ahead bias.

    Our convention:
      evaluation_date = last calendar day of the target quarter.
      This guarantees the GDP advance estimate (released ~30 days later)
      is NOT in the vintage data, while all 3 monthly releases for the
      quarter are in (monthly data is released 2–6 weeks after month end,
      so all 3 months are available by quarter end).

    Parameters
    ----------
    evaluation_date : str
        ALFRED vintage date — must be on or before the last day of the
        target quarter.
    target_quarter_start : str
        First day of the quarter being nowcast (e.g., "2020-01-01" for Q1 2020).

    Raises
    ------
    AssertionError
        If evaluation_date is after quarter end, i.e., GDP may already be released.
    """
    eval_ts   = pd.Timestamp(evaluation_date)
    qtr_start = pd.Timestamp(target_quarter_start)
    qtr_end   = qtr_start + pd.DateOffset(months=3) - pd.DateOffset(days=1)

    if eval_ts > qtr_end:
        raise AssertionError(
            f"LOOK-AHEAD DETECTED: evaluation_date={eval_ts.date()} is after "
            f"target quarter end={qtr_end.date()}. The GDP advance release for "
            f"this quarter may already be in the ALFRED data. "
            f"Set evaluation_date ≤ {qtr_end.date()} to prevent look-ahead bias."
        )


def build_vintage_dfm_data(
    as_of_date: str,
    start: str = "2000-01-01",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the DFM panel (monthly_df, quarterly_df) using ALFRED vintage data.

    This is the vintage-aware counterpart to prepare_dfm_data() in dfm.py.
    Every series is fetched as it existed on as_of_date — no post-vintage
    revisions or later releases are included.

    What you get back:
      monthly_df   : 11 monthly indicators on a monthly DatetimeIndex (freq='MS').
                     Series ends at the last month on or before as_of_date.
      quarterly_df : GDPC1 log-growth on a quarterly DatetimeIndex (freq='QS').
                     The current quarter (inferred from as_of_date) is NaN —
                     that is the nowcast target.

    How the "current quarter" NaN works:
      as_of_date falls within a quarter Q. The ALFRED GDPC1 vintage as of that
      date will only contain data through Q-1 (the GDP advance release for Q
      comes ~30 days after Q ends, after our evaluation_date). We extend the
      index by one NaN row for Q so the model knows to nowcast it.

    Parameters
    ----------
    as_of_date : str
        ALFRED vintage date. All data reflects what was known on this date.
    start : str
        Model estimation start (default "2000-01-01").
    use_cache : bool
        Use ALFRED cache in data/alfred/ (default True).

    Returns
    -------
    (monthly_df, quarterly_df)
        Same format as prepare_dfm_data() — ready to pass to fit_dfm().
    """
    as_of_ts  = pd.Timestamp(as_of_date)
    # Pull from one period before start so the first log-diff is defined
    pre_start_m = (pd.Timestamp(start) - pd.DateOffset(months=2)).strftime("%Y-%m-%d")
    pre_start_q = (pd.Timestamp(start) - pd.DateOffset(months=4)).strftime("%Y-%m-%d")

    # ── Monthly panel ─────────────────────────────────────────────────────────
    monthly_index = pd.date_range(
        start=pre_start_m,
        end=as_of_date,
        freq="MS",
    )
    columns: dict[str, pd.Series] = {}

    for series_id, meta in SERIES_CATALOG.items():
        if meta["freq"] != "M":
            continue
        try:
            raw = fetch_vintage_series(
                series_id, as_of_date, start=pre_start_m, use_cache=use_cache
            )
            transformed = transform_series(raw, meta["transform"])
            aligned = _align_to_monthly(transformed, "M", monthly_index)
            columns[series_id] = aligned
        except Exception as exc:
            logger.warning(
                "Could not load %s vintage %s: %s — filling with NaN",
                series_id, as_of_date, exc,
            )
            columns[series_id] = pd.Series(
                np.nan, index=monthly_index, name=series_id
            )

    monthly_df = pd.DataFrame(columns, index=monthly_index)
    monthly_df.index.name = "date"

    # Trim to requested start, drop leading all-NaN rows
    monthly_df = monthly_df.loc[start:]
    first_valid = monthly_df.dropna(how="all").index[0]
    monthly_df  = monthly_df.loc[first_valid:]

    # Drop columns where the entire history is NaN. This happens when a series
    # doesn't exist in ALFRED for this vintage date (e.g., GACDFSA066MSFRBPHI
    # is absent from ALFRED for 2015 Q1). An all-NaN column causes the DFM
    # standardize step to produce inf/NaN, failing the EM fit. Dropping it
    # here lets the model run with the remaining series — a small accuracy cost
    # vs. a hard failure. Logged as a warning so omissions are visible.
    all_nan_cols = [c for c in monthly_df.columns if monthly_df[c].isna().all()]
    if all_nan_cols:
        logger.warning(
            "Dropping %d fully-NaN series for vintage %s: %s",
            len(all_nan_cols), as_of_date, all_nan_cols,
        )
        monthly_df = monthly_df.drop(columns=all_nan_cols)

    # ── Quarterly GDP — on its natural quarterly-frequency index ─────────────
    gdp_raw    = fetch_vintage_series(
        "GDPC1", as_of_date, start=pre_start_q, use_cache=use_cache
    )
    gdp_growth = transform_series(gdp_raw, "log_diff").dropna()

    # Clip to requested window
    gdp_growth = gdp_growth.loc[start:]
    gdp_growth = gdp_growth.loc[:as_of_date]

    # Move to proper quarterly-frequency DatetimeIndex
    gdp_growth.index = (
        pd.DatetimeIndex(gdp_growth.index).to_period("Q").to_timestamp()
    )
    gdp_growth = gdp_growth.asfreq("QS")

    # Extend by one NaN row for the nowcast target quarter.
    # The target is the quarter that contains as_of_date — its GDP advance
    # release hasn't happened yet (eval_date is the last day of the quarter).
    current_qtr = as_of_ts.to_period("Q").to_timestamp()
    if gdp_growth.index[-1] < current_qtr:
        extension = pd.Series(
            [float("nan")],
            index=pd.DatetimeIndex([current_qtr]),
            name="GDPC1",
        )
        gdp_growth = pd.concat([gdp_growth, extension])
        gdp_growth.index.freq = pd.tseries.frequencies.to_offset("QS")

    quarterly_df = pd.DataFrame({"GDPC1": gdp_growth})

    logger.info(
        "Vintage panel (%s): %d months, %d quarters",
        as_of_date,
        len(monthly_df),
        len(quarterly_df),
    )
    return monthly_df, quarterly_df
