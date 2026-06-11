"""
Tests for src/model/dfm.py — DynamicFactorMQ wrapper.

Unit tests: use tiny synthetic panels (no API, no EM fitting).
Integration tests: fit the real model on a short window; marked @integration.
"""
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.model.dfm import extract_nowcast, prepare_dfm_data


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_monthly(n_months=36, n_series=3, start="2015-01-01", seed=0):
    """Synthetic monthly panel: small values around zero (like log_diffs)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_months, freq="MS")
    cols = [f"M{i}" for i in range(n_series)]
    return pd.DataFrame(rng.normal(0, 0.3, (n_months, n_series)), index=idx, columns=cols)


def _make_quarterly(n_quarters=12, start="2015-01-01", seed=1):
    """
    Synthetic quarterly GDP DataFrame with a proper quarterly DatetimeIndex
    (freq='QS'), as DynamicFactorMQ expects for endog_quarterly.
    The last 2 quarters are NaN to simulate unreleased data.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_quarters, freq="QS")
    values = list(rng.normal(0.5, 0.3, n_quarters - 2)) + [float("nan"), float("nan")]
    s = pd.Series(values, index=idx, name="GDPC1")
    return pd.DataFrame({"GDPC1": s})


# ── Unit tests: prepare_dfm_data ──────────────────────────────────────────────

def _fake_gdp_raw():
    """
    Return a synthetic raw GDPC1 Series with quarterly DatetimeIndex ending
    one quarter before today, so prepare_dfm_data will extend it to the current
    quarter (adding the NaN nowcast target row).
    """
    prev_qtr_end = (pd.Timestamp.today().to_period("Q") - 1).to_timestamp()
    idx = pd.date_range("2000-01-01", end=prev_qtr_end, freq="QS")
    return pd.Series(
        np.linspace(18_000, 20_000, len(idx)),
        index=idx,
        name="GDPC1",
    )


def _fake_monthly_panel(start="1999-11-01", n_months=30, n_cols=11):
    idx = pd.date_range(start, periods=n_months, freq="MS")
    cols = ["INDPRO", "PAYEMS", "UNRATE", "PCEC96", "RSAFS",
            "HOUST", "DGORDER", "DSPIC96", "CPIAUCSL",
            "GACDISA066MSFRBNY", "GACDFSA066MSFRBPHI"][:n_cols]
    data = np.random.default_rng(0).normal(0, 0.2, (n_months, n_cols))
    df = pd.DataFrame(data, index=idx, columns=cols)
    df.iloc[0, :] = np.nan   # first row is NaN from log_diff
    return df


class TestPrepareDFMData:

    @patch("src.data.fred_loader.fetch_series")
    @patch("src.model.dfm.build_panel")
    def test_splits_by_frequency(self, mock_build, mock_fetch):
        """
        prepare_dfm_data must return (monthly_df, quarterly_df) where:
          - quarterly_df contains only GDPC1 with a quarterly-frequency index
          - monthly_df contains all other series with a monthly-frequency index
        """
        mock_build.return_value = _fake_monthly_panel()  # no GDPC1 in monthly panel
        mock_fetch.return_value = _fake_gdp_raw()        # raw GDPC1 for GDP branch

        monthly_df, quarterly_df = prepare_dfm_data(
            start="2000-01-01", use_cache=False
        )

        assert "GDPC1" not in monthly_df.columns, \
            "GDPC1 (quarterly) should not appear in the monthly DataFrame"
        assert "GDPC1" in quarterly_df.columns, \
            "GDPC1 should appear in the quarterly DataFrame"

    @patch("src.data.fred_loader.fetch_series")
    @patch("src.model.dfm.build_panel")
    def test_no_all_nan_rows_in_monthly(self, mock_build, mock_fetch):
        """
        After prepare_dfm_data, no row of the monthly DataFrame should be
        entirely NaN — the first log-diff row (which is all-NaN) is dropped.
        """
        mock_build.return_value = _fake_monthly_panel()
        mock_fetch.return_value = _fake_gdp_raw()

        monthly_df, _ = prepare_dfm_data(start="2000-01-01", use_cache=False)

        all_nan_rows = monthly_df.isna().all(axis=1).sum()
        assert all_nan_rows == 0, (
            f"{all_nan_rows} fully-NaN rows remain in monthly_df; "
            f"the leading all-NaN row from log_diff should have been dropped."
        )

    @patch("src.data.fred_loader.fetch_series")
    @patch("src.model.dfm.build_panel")
    def test_different_frequencies(self, mock_build, mock_fetch):
        """
        monthly_df must have monthly freq ('MS') and quarterly_df must have
        quarterly freq (starting with 'Q'). DynamicFactorMQ uses these
        separate grids — they should NOT share the same index.
        """
        mock_build.return_value = _fake_monthly_panel()
        mock_fetch.return_value = _fake_gdp_raw()

        monthly_df, quarterly_df = prepare_dfm_data(
            start="2000-01-01", use_cache=False
        )

        monthly_freq = monthly_df.index.freqstr
        quarterly_freq = quarterly_df.index.freqstr
        assert monthly_freq == "MS", (
            f"monthly_df should have freq='MS', got '{monthly_freq}'"
        )
        assert quarterly_freq.startswith("Q"), (
            f"quarterly_df should have freq starting with 'Q', got '{quarterly_freq}'"
        )


# ── Unit tests: extract_nowcast ────────────────────────────────────────────────

class TestExtractNowcast:
    """
    Test the nowcast extraction logic with a mock results object.
    We stub out results.fittedvalues so we don't need to run the Kalman filter.
    """

    class _MockResults:
        """Minimal stand-in for DynamicFactorMQResults."""
        def __init__(self, fv: pd.DataFrame):
            self.fittedvalues = fv

    def _make_quarterly_df(self, n_quarters=8, n_missing=2):
        """Build a quarterly DataFrame (freq='QS') with the last n_missing as NaN."""
        idx = pd.date_range("2020-01-01", periods=n_quarters, freq="QS")
        values = [0.5] * (n_quarters - n_missing) + [float("nan")] * n_missing
        s = pd.Series(values, index=idx, name="GDPC1")
        return pd.DataFrame({"GDPC1": s})

    def test_returns_last_missing_quarter(self):
        """
        extract_nowcast should report the LAST quarter where GDP is NaN.
        If Q3 2021 and Q4 2021 are both missing, the nowcast should be for Q4 2021.
        """
        quarterly_df = self._make_quarterly_df(n_quarters=8, n_missing=2)

        idx = quarterly_df.index
        fv_vals = pd.Series(0.42, index=idx, name="GDPC1")
        mock_results = self._MockResults(pd.DataFrame({"GDPC1": fv_vals}))

        out = extract_nowcast(mock_results, quarterly_df)

        assert out is not None
        assert out["date"] == idx[-1], (
            f"Expected nowcast at last index entry {idx[-1].date()}, "
            f"got {out['date'].date()}"
        )

    def test_annualized_is_4x_quarterly(self):
        """nowcast_ann_pct must equal nowcast_qtr_pct × 4."""
        quarterly_df = self._make_quarterly_df()
        idx = quarterly_df.index
        fv = pd.DataFrame({"GDPC1": pd.Series(0.55, index=idx)})
        mock_results = self._MockResults(fv)

        out = extract_nowcast(mock_results, quarterly_df)
        assert abs(out["nowcast_ann_pct"] - 4 * out["nowcast_qtr_pct"]) < 1e-12

    def test_returns_none_when_no_missing(self):
        """If all GDP quarters have actual data, there is nothing to nowcast."""
        idx = pd.date_range("2020-01-01", periods=8, freq="QS")
        s = pd.Series(0.5, index=idx, name="GDPC1")   # All quarters have actuals
        quarterly_df = pd.DataFrame({"GDPC1": s})

        fv = pd.DataFrame({"GDPC1": pd.Series(0.5, index=idx)})
        mock_results = self._MockResults(fv)

        result = extract_nowcast(mock_results, quarterly_df)
        assert result is None

    def test_nowcast_value_matches_fittedvalues(self):
        """
        The reported nowcast_qtr_pct must equal the fittedvalues entry
        at the nowcast date. This verifies we're reading the right cell.
        """
        quarterly_df = self._make_quarterly_df(n_quarters=8, n_missing=1)
        idx = quarterly_df.index

        fv_series = pd.Series(0.0, index=idx)
        fv_series.iloc[-1] = 0.1234        # distinctive value at the nowcast quarter

        mock_results = self._MockResults(pd.DataFrame({"GDPC1": fv_series}))
        out = extract_nowcast(mock_results, quarterly_df)

        assert abs(out["nowcast_qtr_pct"] - 0.1234) < 1e-10, (
            f"Reported nowcast {out['nowcast_qtr_pct']:.6f} ≠ "
            f"fittedvalues entry 0.1234 at {idx[-1].date()}"
        )


# ── Integration tests (require FRED API + model fit) ─────────────────────────

@pytest.mark.integration
class TestDFMIntegration:
    """
    Fits a single-factor DFM on a 5-year window and checks convergence and
    nowcast plausibility. This takes ~30–90 seconds.
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.getenv("FRED_API_KEY"):
            pytest.skip("FRED_API_KEY not set")

    def test_dfm_fits_and_converges(self):
        """
        The EM algorithm should produce a finite log-likelihood and
        the factor estimates should be a well-shaped array.

        We use a short window (2015–2020) to keep runtime under ~60s.
        """
        from src.model.dfm import fit_dfm

        # Monthly and quarterly DataFrames are on SEPARATE grids — do NOT
        # align them to the same index. DynamicFactorMQ handles alignment.
        monthly_df   = _make_monthly(n_months=72, n_series=4, start="2015-01-01")
        quarterly_df = _make_quarterly(n_quarters=24, start="2015-01-01")
        monthly_df.columns = ["INDPRO", "PAYEMS", "UNRATE", "PCEC96"]

        results = fit_dfm(monthly_df, quarterly_df, n_factors=1, maxiter=200)

        assert np.isfinite(results.llf), \
            f"Log-likelihood is not finite: {results.llf}"
        factors = results.factors.smoothed
        assert factors.shape[0] == len(monthly_df), \
            f"Factor array has {factors.shape[0]} rows, expected {len(monthly_df)}"

    def test_nowcast_in_plausible_range(self):
        """
        Run the full pipeline on recent real data and verify the annualized
        nowcast is in a plausible range for the US economy.

        We use a short estimation window (2018–present) for speed. The nowcast
        should be between −20% and +15% annualized — this range would catch
        catastrophic failures while accommodating historical outliers (e.g.,
        COVID-19 Q2 2020 was −31%).
        """
        monthly_df, quarterly_df = prepare_dfm_data(
            start="2018-01-01", use_cache=True
        )

        from src.model.dfm import fit_dfm

        results  = fit_dfm(monthly_df, quarterly_df, n_factors=1, maxiter=300)
        nowcast  = extract_nowcast(results, quarterly_df)

        assert nowcast is not None, "No nowcast produced — check for missing GDP quarters"
        assert -20.0 < nowcast["nowcast_ann_pct"] < 15.0, (
            f"Annualized nowcast {nowcast['nowcast_ann_pct']:.2f}% is outside "
            f"the plausible range (−20%, +15%). This may indicate a data or "
            f"model setup error."
        )
