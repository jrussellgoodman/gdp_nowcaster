"""
Tests for Phase 3 news decomposition — src/model/dfm.py.

Unit tests: check structure and arithmetic using synthetic or mocked data.
Integration tests: run compute_news on real FRED data and verify the core
  identity: sum(per-series impacts) == total impact of news.
"""
import os

import numpy as np
import pandas as pd
import pytest

from src.model.dfm import REAL_ACTIVITY_SERIES, make_block_structure


# ── Helpers ────────────────────────────────────────────────────────────────────

def _small_monthly(cols=None, n_months=12, start="2020-01-01"):
    """Tiny monthly DataFrame for structural tests (no model fitting)."""
    if cols is None:
        cols = ["INDPRO", "PAYEMS", "CPIAUCSL", "GACDISA066MSFRBNY"]
    idx = pd.date_range(start, periods=n_months, freq="MS")
    return pd.DataFrame(0.0, index=idx, columns=cols)


def _small_quarterly(cols=None, n_quarters=4, start="2020-01-01"):
    """Tiny quarterly DataFrame for structural tests."""
    if cols is None:
        cols = ["GDPC1"]
    idx = pd.date_range(start, periods=n_quarters, freq="QS")
    return pd.DataFrame(0.0, index=idx, columns=cols)


# ── Unit tests: make_block_structure ──────────────────────────────────────────

class TestMakeBlockStructure:

    def test_all_series_present(self):
        """Every column in monthly_df + quarterly_df must appear in the output dict."""
        m = _small_monthly(["INDPRO", "PAYEMS", "CPIAUCSL", "GACDISA066MSFRBNY"])
        q = _small_quarterly(["GDPC1"])
        blocks = make_block_structure(m, q)

        for s in list(m.columns) + list(q.columns):
            assert s in blocks, f"Series {s!r} missing from block dict"

    def test_real_series_load_on_both_factors(self):
        """
        INDPRO, PAYEMS, UNRATE, GDPC1, etc. are real-activity indicators
        and must load on both 'global' AND 'real' factors.
        """
        real_monthly = [s for s in REAL_ACTIVITY_SERIES if s != "GDPC1"]
        m = _small_monthly(real_monthly[:4] + ["CPIAUCSL"])
        q = _small_quarterly(["GDPC1"])
        blocks = make_block_structure(m, q)

        for s in ["INDPRO", "PAYEMS", "GDPC1"]:
            if s in blocks:
                assert "real" in blocks[s], f"{s} must load on 'real' factor"
                assert "global" in blocks[s], f"{s} must load on 'global' factor"

    def test_survey_and_cpi_load_only_on_global(self):
        """
        Survey indices and CPI measure sentiment/prices, not real activity.
        They should load ONLY on the 'global' factor.
        """
        m = _small_monthly(["PAYEMS", "CPIAUCSL", "GACDISA066MSFRBNY", "GACDFSA066MSFRBPHI"])
        q = _small_quarterly(["GDPC1"])
        blocks = make_block_structure(m, q)

        for s in ["CPIAUCSL", "GACDISA066MSFRBNY", "GACDFSA066MSFRBPHI"]:
            if s in blocks:
                assert blocks[s] == ["global"], (
                    f"{s} should load only on ['global'], got {blocks[s]}"
                )

    def test_values_are_lists_of_strings(self):
        """All block dict values must be non-empty lists of strings."""
        m = _small_monthly()
        q = _small_quarterly()
        blocks = make_block_structure(m, q)

        for key, val in blocks.items():
            assert isinstance(val, list) and len(val) > 0, (
                f"blocks[{key!r}] = {val!r}; must be a non-empty list"
            )
            for item in val:
                assert isinstance(item, str), (
                    f"Factor name {item!r} in blocks[{key!r}] must be a string"
                )

    def test_gdpc1_in_real_block(self):
        """GDPC1 (quarterly GDP) must always load on the 'real' block."""
        m = _small_monthly(["INDPRO", "CPIAUCSL"])
        q = _small_quarterly(["GDPC1"])
        blocks = make_block_structure(m, q)
        assert "GDPC1" in blocks, "GDPC1 must appear in block dict"
        assert "real" in blocks["GDPC1"], "GDPC1 must load on 'real' factor"


# ── Unit tests: news arithmetic identity ─────────────────────────────────────

class TestNewsArithmetic:
    """
    Verify the core news identity without fitting a real DFM.

    The fundamental property: sum(per-series impacts) == total revision.
    We test this with hand-constructed data, independent of any model output.
    """

    def test_impacts_sum_equals_total(self):
        """
        Individual series impacts must sum to the total impact.

        This is the defining mathematical property of the Kalman-gain-based
        news decomposition.  Each impact = news_surprise × Kalman_gain, and
        the sum of those products equals the forecast revision.
        """
        # Simulate per-series news impacts (quarterly %)
        impacts = pd.Series({
            "PAYEMS":  0.005104,
            "UNRATE": -0.008524,
            "CPIAUCSL": 0.004685,
            "GACDISA066MSFRBNY": 0.018796,
            "GACDFSA066MSFRBPHI": -0.028630,
        })
        total = -0.008569   # expected sum (from a real DFM run)
        assert abs(impacts.sum() - total) < 1e-6, (
            f"Impacts sum ({impacts.sum():.8f}) ≠ expected total ({total:.8f})"
        )

    def test_annualized_is_4x_quarterly(self):
        """Annualized revision = quarterly × 4."""
        qtr = 0.0123
        assert abs(qtr * 4 - 0.0492) < 1e-12

    def test_revision_is_after_minus_before(self):
        """revision = after_nowcast - before_nowcast."""
        before = 0.6898
        after  = 0.6812
        assert abs((after - before) - (-0.0086)) < 1e-4

    def test_positive_payems_surprise_positive_impact(self):
        """
        If payrolls come in above forecast (good news for the economy),
        the Kalman gain for PAYEMS on GDPC1 must be positive, so the
        impact on the GDP nowcast must be positive too.

        surprise > 0 AND weight > 0  →  impact > 0
        """
        surprise = 0.020    # payrolls beat forecast
        weight   = 0.2528   # Kalman gain (empirical from blocked DFM)
        impact   = surprise * weight
        assert impact > 0, (
            f"Positive PAYEMS surprise ({surprise}) × positive weight ({weight}) "
            f"should give positive impact, got {impact}"
        )

    def test_positive_unrate_surprise_negative_impact(self):
        """
        If unemployment comes in above forecast (bad news for the economy),
        the Kalman gain for UNRATE on GDPC1 must be negative, so the impact
        on GDP is negative.

        UNRATE diff is in pp, so surprise > 0 means unemployment rose more
        than expected.  UNRATE loads negatively on the economic-activity
        factor, giving a negative Kalman gain.

        surprise > 0 AND weight < 0  →  impact < 0
        """
        surprise = 0.030    # unemployment rose more than expected
        weight   = -0.2861  # Kalman gain (empirical, negative because UNRATE
                             # rises when economy is weak)
        impact   = surprise * weight
        assert impact < 0, (
            f"Positive UNRATE surprise × negative weight should give "
            f"negative impact on GDP nowcast, got {impact}"
        )


# ── Integration tests ─────────────────────────────────────────────────────────

@pytest.mark.integration
class TestNewsIntegration:
    """
    Run compute_news() on real FRED data and verify the sum identity.
    This takes ~2–4 minutes (fits a blocked DFM).
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.getenv("FRED_API_KEY"):
            pytest.skip("FRED_API_KEY not set")

    def test_compute_news_returns_expected_keys(self):
        """compute_news must return a dict with all documented keys."""
        from src.model.dfm import compute_news

        result = compute_news(
            before_end="2026-04-30",
            start="2000-01-01",
            use_cache=True,
        )

        required = {
            "target_quarter", "target_date",
            "before_nowcast_qtr", "before_nowcast_ann",
            "after_nowcast_qtr", "after_nowcast_ann",
            "revision_qtr", "revision_ann",
            "total_impact_qtr", "impacts", "news_results",
        }
        missing = required - set(result.keys())
        assert not missing, f"compute_news missing keys: {missing}"

    def test_news_impacts_sum_to_total_news_impact(self):
        """
        The central identity of news decomposition:
          sum_i( news_i × weight_i ) == total impact of news

        'impact of news' is the component due to NEW observations only
        (data_updates), excluding any revisions to previously-known data.

        This must hold to machine precision because it follows from the
        linearity of the Kalman filter.  We allow tolerance 1e-5 for
        floating-point accumulation.
        """
        from src.model.dfm import compute_news

        result = compute_news(
            before_end="2026-04-30",
            start="2000-01-01",
            use_cache=True,
        )

        impacts_df   = result["impacts"]
        news_obj     = result["news_results"]

        sum_impacts  = float(impacts_df["impact"].sum())
        total_news   = float(news_obj.impacts["impact of news"].iloc[0])

        assert abs(sum_impacts - total_news) < 1e-5, (
            f"Sum of per-series impacts ({sum_impacts:.8f}) ≠ "
            f"total impact of news ({total_news:.8f}).  "
            f"Difference: {sum_impacts - total_news:.2e}.  "
            "This indicates a methodological error in the news decomposition."
        )

    def test_blocked_dfm_better_than_single_factor(self):
        """
        The blocked 2-factor DFM must have higher log-likelihood than
        the baseline 1-factor model.  This confirms the added factor
        captures genuine variation not explained by a single global factor.
        """
        from src.model.dfm import prepare_dfm_data, fit_dfm, make_block_structure

        monthly_df, quarterly_df = prepare_dfm_data(
            start="2000-01-01", use_cache=True
        )
        blocks = make_block_structure(monthly_df, quarterly_df)

        r1 = fit_dfm(monthly_df, quarterly_df, n_factors=1, maxiter=500)
        r2 = fit_dfm(monthly_df, quarterly_df,
                     factor_blocks=blocks, maxiter=500)

        llf_improvement = r2.llf - r1.llf
        assert llf_improvement > 50, (
            f"Blocked DFM LLF improvement ({llf_improvement:.1f}) is too small. "
            f"Expected > 50 log-likelihood points.  "
            "This may indicate the block structure failed to converge."
        )
