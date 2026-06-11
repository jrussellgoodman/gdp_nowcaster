"""
Tests for src/data/fred_loader.py

How these tests are organized:
  - UNIT TESTS  (no API key needed): test transform math and alignment logic
    using synthetic data we create ourselves. These run in milliseconds.
  - INTEGRATION TESTS  (need FRED_API_KEY): download a small slice of real
    data and sanity-check its values against known benchmarks. Automatically
    skipped if the API key isn't in the environment.

Run unit tests only:        pytest -q
Run integration tests too:  pytest -q -m integration
"""
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.fred_loader import (
    SERIES_CATALOG,
    _align_to_monthly,
    build_panel,
    fetch_series,
    transform_series,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _monthly_series(values, start="2020-01-01"):
    """Build a monthly pd.Series with given values."""
    idx = pd.date_range(start, periods=len(values), freq="MS")
    return pd.Series(values, index=idx, dtype=float)


def _quarterly_series(values, start="2020-01-01"):
    """Build a quarterly pd.Series (FRED convention: dates at quarter start)."""
    idx = pd.date_range(start, periods=len(values), freq="QS")
    return pd.Series(values, index=idx, dtype=float)


# ── Unit tests: transform_series ───────────────────────────────────────────────

class TestTransformSeries:
    """
    Test each transform type with synthetic data.
    No FRED API needed — we construct the data ourselves.
    """

    def test_log_diff_constant_growth(self):
        """
        If the series grows by exactly factor e each period, every log_diff
        value should equal 100 (since 100 × ln(e) = 100).

        This verifies the formula: 100 × (ln(x_t) − ln(x_{t−1})).
        """
        # e^0, e^1, e^2, e^3 — geometric growth with ratio e
        values = np.exp([0.0, 1.0, 2.0, 3.0])
        s = _monthly_series(values)
        result = transform_series(s, "log_diff")

        assert np.isnan(result.iloc[0]), "First obs must be NaN (nothing to diff against)"
        np.testing.assert_allclose(
            result.iloc[1:].values, 100.0, atol=1e-10,
            err_msg="log_diff of e^n series should equal 100 exactly",
        )

    def test_log_diff_approximates_percent_change(self):
        """
        For small changes, log_diff ≈ actual percent change.
        100 → 102 is a 2% change; log_diff gives ≈ 1.98% (a close approximation).
        """
        s = _monthly_series([100.0, 102.0])
        result = transform_series(s, "log_diff")
        actual_pct = 100.0 * (102 / 100 - 1)          # 2.000%
        log_diff_val = result.iloc[1]                   # ≈ 1.980%
        assert abs(log_diff_val - actual_pct) < 0.05, (
            f"log_diff {log_diff_val:.4f}% should be close to true % change {actual_pct:.4f}%"
        )

    def test_log_diff_first_obs_is_nan(self):
        """There is no previous period for the first obs, so it must be NaN."""
        s = _monthly_series([10.0, 11.0, 12.0])
        result = transform_series(s, "log_diff")
        assert np.isnan(result.iloc[0])
        assert result.notna().sum() == 2

    def test_diff_simple(self):
        """First differences: each value minus the one before it."""
        s = _monthly_series([3.0, 5.0, 2.0, 7.0])
        result = transform_series(s, "diff")
        assert np.isnan(result.iloc[0])
        np.testing.assert_allclose(result.iloc[1:].values, [2.0, -3.0, 5.0])

    def test_level_is_unchanged(self):
        """Level transform should return the series completely unchanged."""
        values = [10.0, -5.0, 0.0, 100.0]
        s = _monthly_series(values)
        result = transform_series(s, "level")
        np.testing.assert_array_equal(result.values, values)

    def test_invalid_transform_raises_valueerror(self):
        s = _monthly_series([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown transform"):
            transform_series(s, "pct_change_annualized")  # not a valid key


# ── Unit tests: _align_to_monthly ──────────────────────────────────────────────

class TestAlignToMonthly:
    """
    Test that series are placed on the monthly grid at the correct positions.

    The critical case is quarterly data: DynamicFactorMQ requires quarterly values
    to appear at the LAST month of each quarter (March, June, September, December),
    with NaN in the preceding two months. FRED dates quarterly data at the FIRST
    month of the quarter, so we shift by +2 months.
    """

    def _monthly_idx(self, start="2020-01-01", periods=12):
        return pd.date_range(start, periods=periods, freq="MS")

    def test_monthly_series_placed_correctly(self):
        """A monthly series should appear at the correct month-start dates."""
        s = _monthly_series([1.0, 2.0, 3.0], start="2020-01-01")
        monthly_idx = self._monthly_idx("2020-01-01", periods=6)
        result = _align_to_monthly(s, "M", monthly_idx)

        assert result["2020-01-01"] == 1.0
        assert result["2020-02-01"] == 2.0
        assert result["2020-03-01"] == 3.0

    def test_quarterly_values_at_quarter_end_months(self):
        """
        Q1 2020 (FRED date: 2020-01-01) must appear at 2020-03-01.
        Q2 2020 (FRED date: 2020-04-01) must appear at 2020-06-01.

        DynamicFactorMQ interprets the quarterly value as an aggregate over the
        full quarter, so it must sit at the quarter-END month.
        """
        # Two quarters: Q1 and Q2 2020
        s = _quarterly_series([10.0, 20.0], start="2020-01-01")
        monthly_idx = self._monthly_idx("2020-01-01", periods=8)
        result = _align_to_monthly(s, "Q", monthly_idx)

        # Values should be at the last month of each quarter
        assert result["2020-03-01"] == 10.0, "Q1 2020 should be at March"
        assert result["2020-06-01"] == 20.0, "Q2 2020 should be at June"

        # The other months of each quarter should be NaN
        for m in ["2020-01-01", "2020-02-01", "2020-04-01", "2020-05-01"]:
            assert np.isnan(result[m]), f"{m} should be NaN (within-quarter gap)"

    def test_quarterly_nan_ratio(self):
        """Over a full year, exactly 4 of 12 monthly slots should have GDP values."""
        s = _quarterly_series([1.0, 2.0, 3.0, 4.0], start="2020-01-01")
        monthly_idx = self._monthly_idx("2020-01-01", periods=12)
        result = _align_to_monthly(s, "Q", monthly_idx)

        assert result.notna().sum() == 4, (
            f"Expected 4 quarterly values in 12 months, got {result.notna().sum()}"
        )
        assert result.isna().sum() == 8

    def test_quarterly_non_nan_at_correct_months(self):
        """All non-NaN GDP values must fall on months 3, 6, 9, or 12."""
        s = _quarterly_series([1.0, 2.0, 3.0, 4.0], start="2019-01-01")
        monthly_idx = self._monthly_idx("2019-01-01", periods=12)
        result = _align_to_monthly(s, "Q", monthly_idx)

        for date, val in result.items():
            if pd.notna(val):
                assert date.month in {3, 6, 9, 12}, (
                    f"GDP value at {date.strftime('%Y-%m')} (month={date.month}), "
                    f"but only months 3, 6, 9, 12 are quarter-ends"
                )


# ── Unit tests: SERIES_CATALOG structure ──────────────────────────────────────

class TestSeriesCatalog:
    """
    Validate the catalog that defines which series we download and how we treat them.
    These tests catch mis-configurations before we ever hit the API.
    """

    def test_required_fields_present(self):
        """Every entry must have 'name', 'freq', and 'transform'."""
        for series_id, meta in SERIES_CATALOG.items():
            for field in ("name", "freq", "transform"):
                assert field in meta, f"'{series_id}' is missing field '{field}'"

    def test_valid_freq_values(self):
        for series_id, meta in SERIES_CATALOG.items():
            assert meta["freq"] in ("M", "Q"), (
                f"'{series_id}' has freq='{meta['freq']}'; must be 'M' or 'Q'"
            )

    def test_valid_transform_values(self):
        valid = {"log_diff", "diff", "level"}
        for series_id, meta in SERIES_CATALOG.items():
            assert meta["transform"] in valid, (
                f"'{series_id}' has transform='{meta['transform']}'; "
                f"must be one of {valid}"
            )

    def test_gdpc1_is_quarterly(self):
        """GDPC1 is a quarterly series — if marked monthly it would misalign the panel."""
        assert "GDPC1" in SERIES_CATALOG, "GDPC1 (real GDP) must be in the catalog"
        assert SERIES_CATALOG["GDPC1"]["freq"] == "Q", (
            "GDPC1 freq must be 'Q' — it is released quarterly, not monthly"
        )

    def test_no_nominal_gdp(self):
        """'GDP' alone is nominal GDP — we must use GDPC1 (real). Catch the mistake."""
        assert "GDP" not in SERIES_CATALOG, (
            "Found 'GDP' (nominal) in catalog. Use 'GDPC1' (real GDP, chained 2017 $)."
        )

    def test_monthly_pce_not_quarterly(self):
        """PCEC96 is monthly real PCE. PCECC96 is quarterly. We need monthly."""
        assert "PCECC96" not in SERIES_CATALOG, (
            "Found PCECC96 (quarterly PCE). Use PCEC96 (monthly real PCE) instead."
        )
        if "PCEC96" in SERIES_CATALOG:
            assert SERIES_CATALOG["PCEC96"]["freq"] == "M", (
                "PCEC96 should be marked 'M' (monthly)"
            )

    def test_no_napm_ism_series(self):
        """ISM/PMI (NAPM*) were removed from FRED in 2014 — must not be in catalog."""
        for series_id in SERIES_CATALOG:
            assert not series_id.startswith("NAPM"), (
                f"'{series_id}' starts with 'NAPM' — these series were removed from "
                f"FRED in 2014. Use GACDISA066MSFRBNY or GACDFSA066MSFRBPHI instead."
            )

    def test_uses_fed_survey_substitutes(self):
        """
        Verify the NY Fed and Philly Fed survey series are present as ISM substitutes.
        These are the series the CLAUDE.md data rules specifically require.
        """
        assert "GACDISA066MSFRBNY" in SERIES_CATALOG, (
            "NY Fed survey series GACDISA066MSFRBNY should be in catalog"
        )
        assert "GACDFSA066MSFRBPHI" in SERIES_CATALOG, (
            "Philly Fed survey series GACDFSA066MSFRBPHI should be in catalog"
        )


# ── Unit tests: build_panel with mocked fetch_series ──────────────────────────

class TestBuildPanelMocked:
    """
    Test build_panel() structure without hitting the FRED API.
    We replace fetch_series with a function that returns synthetic data.
    """

    def test_panel_columns_match_catalog(self):
        """Panel should have one column per series in SERIES_CATALOG."""

        def fake_fetch(series_id, start="1990-01-01", end=None, use_cache=True):
            meta = SERIES_CATALOG[series_id]
            if meta["freq"] == "Q":
                idx = pd.date_range("2020-01-01", periods=4, freq="QS")
            else:
                idx = pd.date_range("2020-01-01", periods=12, freq="MS")
            return pd.Series(np.exp(np.linspace(0, 0.05, len(idx))), index=idx, name=series_id)

        with patch("src.data.fred_loader.fetch_series", side_effect=fake_fetch):
            panel = build_panel(start="2020-01-01", end="2020-12-31", use_cache=False)

        assert set(panel.columns) == set(SERIES_CATALOG.keys()), (
            "Panel columns should exactly match SERIES_CATALOG keys"
        )

    def test_panel_has_monthly_index(self):
        """Panel rows should be monthly (freq='MS')."""

        def fake_fetch(series_id, start="1990-01-01", end=None, use_cache=True):
            meta = SERIES_CATALOG[series_id]
            if meta["freq"] == "Q":
                idx = pd.date_range("2020-01-01", periods=4, freq="QS")
            else:
                idx = pd.date_range("2020-01-01", periods=12, freq="MS")
            return pd.Series(np.ones(len(idx)) * 100.0, index=idx, name=series_id)

        with patch("src.data.fred_loader.fetch_series", side_effect=fake_fetch):
            panel = build_panel(start="2020-01-01", end="2020-12-31", use_cache=False)

        assert len(panel) == 12, f"Expected 12 months, got {len(panel)}"
        assert isinstance(panel.index, pd.DatetimeIndex)

    def test_gdp_column_nan_pattern(self):
        """In the mocked panel, GDPC1 should have values only at quarter-end months."""

        def fake_fetch(series_id, start="1990-01-01", end=None, use_cache=True):
            meta = SERIES_CATALOG[series_id]
            if meta["freq"] == "Q":
                idx = pd.date_range("2020-01-01", periods=4, freq="QS")
            else:
                idx = pd.date_range("2020-01-01", periods=12, freq="MS")
            return pd.Series(np.exp(np.linspace(0, 0.05, len(idx))), index=idx, name=series_id)

        with patch("src.data.fred_loader.fetch_series", side_effect=fake_fetch):
            panel = build_panel(start="2020-01-01", end="2020-12-31", use_cache=False)

        gdp = panel["GDPC1"]
        for date, val in gdp.items():
            if pd.notna(val):
                assert date.month in {3, 6, 9, 12}, (
                    f"GDPC1 non-NaN at {date.strftime('%Y-%m')} (month {date.month}), "
                    f"but should only appear at quarter-end months 3, 6, 9, 12"
                )


# ── Integration tests (require live FRED API) ──────────────────────────────────

@pytest.mark.integration
class TestFREDIntegration:
    """
    These tests download real data from FRED.
    They are skipped automatically when FRED_API_KEY is not set.

    Each test includes a comment explaining the benchmark it checks against
    so you can verify the logic independently if something looks wrong.
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.getenv("FRED_API_KEY"):
            pytest.skip("FRED_API_KEY not set")

    def test_fetch_gdpc1_level(self):
        """
        Sanity check: GDPC1 (real GDP) should be in the range $15,000B–$30,000B
        for the period 2018–2020. The actual value for Q4 2019 was ~$19,254B
        (2017 chained dollars). We check a wide range to be robust to revisions.
        """
        raw = fetch_series("GDPC1", start="2018-01-01", end="2020-12-31", use_cache=False)

        assert isinstance(raw, pd.Series), "Expected a pd.Series"
        assert len(raw) >= 8, f"Expected at least 8 quarters, got {len(raw)}"
        assert raw.min() > 15_000, f"GDP suspiciously low: {raw.min():.0f}"
        assert raw.max() < 30_000, f"GDP suspiciously high: {raw.max():.0f}"

    def test_gdp_growth_rate_long_run_average(self):
        """
        Benchmark: average quarterly GDP growth (log_diff) over 2000–2019 should
        correspond to roughly 1.5%–3.5% annualized growth.

        Annualized growth = 4 × quarterly_log_diff (both in percent units).
        So quarterly log_diff should be in [0.375%, 0.875%].

        We exclude 2020 because COVID caused a historic contraction/rebound that
        would distort the average.

        Source: BEA / FRED historical data; consistent with published estimates
        of US trend GDP growth (~2-3% per year over this period).
        """
        raw = fetch_series("GDPC1", start="2000-01-01", end="2019-12-31", use_cache=False)
        growth = transform_series(raw, "log_diff").dropna()

        avg_quarterly_pct = growth.mean()
        assert 0.2 < avg_quarterly_pct < 1.0, (
            f"Average quarterly GDP growth = {avg_quarterly_pct:.3f}% "
            f"(= {avg_quarterly_pct * 4:.2f}% annualized). "
            f"Expected roughly 0.375–0.875% per quarter (1.5–3.5% annualized). "
            f"This may indicate a transform or series ID error."
        )

    def test_gdp_quarterly_placement_on_monthly_grid(self):
        """
        After aligning to a monthly grid, GDPC1 values must sit at quarter-end
        months (March, June, September, December), not quarter-start months.

        DynamicFactorMQ requires this convention. If values were at January/April/
        July/October instead, the temporal aggregation constraint would be wrong
        and the nowcast would be meaningless.
        """
        # Fetch from Q4 2018 so Q1 2019's log_diff (vs Q4 2018) is defined.
        # Without this, Q1 2019 would be NaN (nothing to difference against).
        raw = fetch_series("GDPC1", start="2018-10-01", end="2020-12-31", use_cache=False)
        growth = transform_series(raw, "log_diff")
        monthly_idx = pd.date_range("2019-01-01", "2020-12-31", freq="MS")
        aligned = _align_to_monthly(growth, "Q", monthly_idx)

        # 24 months, 8 quarters; all 8 should be non-NaN because we fetched the
        # prior quarter as the base for the first log_diff
        non_nan = aligned.dropna()
        assert len(non_nan) == 8, (
            f"Expected 8 quarterly GDP values over 2019–2020, got {len(non_nan)}"
        )

        # Every non-NaN value must be at a quarter-end month
        for date in non_nan.index:
            assert date.month in {3, 6, 9, 12}, (
                f"GDP value at {date.strftime('%Y-%m')} — month {date.month} is not "
                f"a quarter-end month. Check _align_to_monthly."
            )

    def test_payems_growth_2019_sanity(self):
        """
        Benchmark: nonfarm payrolls in 2019 averaged ~170k–200k new jobs per month,
        on a base of ~150M jobs, giving monthly log_diff ≈ 0.11–0.14%.

        We check the range [0.05%, 0.30%] to be robust to revisions.
        A value outside this range likely indicates a transform error (e.g.,
        accidentally annualizing the monthly series).

        Source: BLS via FRED. Consistent with 2019 Beige Book descriptions of
        "solid" labor market growth.
        """
        raw = fetch_series("PAYEMS", start="2019-01-01", end="2019-12-31", use_cache=False)
        growth = transform_series(raw, "log_diff").dropna()

        avg_monthly_pct = growth.mean()
        assert 0.05 < avg_monthly_pct < 0.30, (
            f"Average monthly PAYEMS log_diff = {avg_monthly_pct:.4f}% "
            f"(≈ {avg_monthly_pct * 12:.2f}% annualized). "
            f"Expected ≈ 0.11–0.14% per month. "
            f"Outside [0.05, 0.30] suggests a transform or unit error."
        )
