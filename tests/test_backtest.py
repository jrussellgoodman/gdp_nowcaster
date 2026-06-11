"""
Tests for Phase 4 — ALFRED vintage backtest.

Unit tests: verify look-ahead assertion, quarter range generation, and
  RMSE statistics using synthetic data. No FRED API calls.
Integration test: run a 2-quarter real-time backtest and check that:
  (a) the look-ahead assertion passes for every evaluated quarter,
  (b) nowcasts are in a plausible range,
  (c) the ALFRED vintage panel ends before the advance GDP release.

Run unit tests only (fast):    pytest tests/test_backtest.py -k "not integration" -q
Run all including integration: pytest tests/test_backtest.py -m integration -q
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from src.backtest.runner import (
    backtest_quarter_range,
    compute_rmse_stats,
)
from src.backtest.vintage import assert_no_lookahead


# ── Unit tests: assert_no_lookahead ──────────────────────────────────────────

class TestAssertNoLookahead:
    """
    The look-ahead assertion is the most critical safeguard in the backtest.
    If it fails to catch an invalid evaluation_date, the entire backtest is
    potentially contaminated with look-ahead bias.
    """

    def test_passes_on_last_day_of_quarter(self):
        """
        evaluation_date = last day of the quarter → safe, must NOT raise.

        Q1 2020 ends March 31, 2020. GDP advance release: April 29, 2020.
        Using March 31 as evaluation_date means GDP for Q1 2020 is definitely
        not in the ALFRED data.
        """
        assert_no_lookahead("2020-03-31", "2020-01-01")  # Q1 2020

    def test_passes_on_day_within_quarter(self):
        """Mid-quarter evaluation dates are also safe."""
        assert_no_lookahead("2020-02-15", "2020-01-01")

    def test_passes_on_first_day_of_quarter(self):
        """First day of the quarter is safe (advance release is 3 months away)."""
        assert_no_lookahead("2020-01-01", "2020-01-01")

    def test_raises_one_day_after_quarter_end(self):
        """
        evaluation_date = quarter end + 1 day → MUST raise.

        April 1, 2020 is after Q1 2020 ends. The GDP advance release
        (April 29) hasn't happened yet, but we're already in territory
        where revisions to prior data could have leaked in.
        Our assertion uses a conservative cutoff: any date after quarter end.
        """
        with pytest.raises(AssertionError, match="LOOK-AHEAD DETECTED"):
            assert_no_lookahead("2020-04-01", "2020-01-01")

    def test_raises_on_advance_release_day(self):
        """
        evaluation_date = actual advance GDP release date → MUST raise.
        Q1 2020 advance was April 29, 2020; this is after quarter end.
        """
        with pytest.raises(AssertionError, match="LOOK-AHEAD DETECTED"):
            assert_no_lookahead("2020-04-29", "2020-01-01")

    def test_raises_on_next_quarter_start(self):
        """Q2 start (April 1) is after Q1 end — must raise for Q1 target."""
        with pytest.raises(AssertionError):
            assert_no_lookahead("2021-04-01", "2021-01-01")

    @pytest.mark.parametrize("quarter_start,last_day", [
        ("2020-01-01", "2020-03-31"),  # Q1
        ("2020-04-01", "2020-06-30"),  # Q2
        ("2020-07-01", "2020-09-30"),  # Q3
        ("2020-10-01", "2020-12-31"),  # Q4
        ("2021-01-01", "2021-03-31"),  # Q1 leap year / non-leap
    ])
    def test_boundary_is_last_day_of_quarter(self, quarter_start, last_day):
        """
        Exact quarter-end date must pass; the day after must fail.
        This parameterized test checks all four quarters.
        """
        assert_no_lookahead(last_day, quarter_start)      # should pass
        next_day = (pd.Timestamp(last_day) + pd.DateOffset(days=1)).strftime("%Y-%m-%d")
        with pytest.raises(AssertionError):
            assert_no_lookahead(next_day, quarter_start)   # should raise


# ── Unit tests: backtest_quarter_range ───────────────────────────────────────

class TestBacktestQuarterRange:

    def test_returns_correct_number_of_quarters(self):
        """Q1 2020 through Q4 2020 = 4 quarters."""
        pairs = backtest_quarter_range("2020-01-01", "2020-10-01")
        assert len(pairs) == 4

    def test_evaluation_dates_are_last_days_of_quarters(self):
        """
        Each evaluation_date must be the last calendar day of the quarter.
        Q1 = March 31, Q2 = June 30, Q3 = September 30, Q4 = December 31.
        """
        expected = ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]
        pairs = backtest_quarter_range("2020-01-01", "2020-10-01")
        eval_dates = [p[1] for p in pairs]
        assert eval_dates == expected

    def test_target_quarters_are_timestamps(self):
        """First element of each pair must be a pd.Timestamp."""
        pairs = backtest_quarter_range("2019-01-01", "2019-04-01")
        for tq, _ in pairs:
            assert isinstance(tq, pd.Timestamp)

    def test_no_lookahead_for_all_generated_pairs(self):
        """
        Every (target_quarter, evaluation_date) pair produced by
        backtest_quarter_range must pass the look-ahead assertion.
        This is the meta-test: if the generator is correct, no individual
        backtest point should ever need a look-ahead check to fail.
        """
        pairs = backtest_quarter_range("2015-01-01", "2025-10-01")
        for target_q, eval_date in pairs:
            # Should not raise
            assert_no_lookahead(eval_date, str(target_q.date()))

    def test_single_quarter(self):
        """Edge case: start == end gives exactly one quarter."""
        pairs = backtest_quarter_range("2022-04-01", "2022-04-01")
        assert len(pairs) == 1
        assert pairs[0][1] == "2022-06-30"

    def test_chronological_order(self):
        """Quarters must be returned in chronological order."""
        pairs = backtest_quarter_range("2020-01-01", "2022-10-01")
        target_quarters = [p[0] for p in pairs]
        assert target_quarters == sorted(target_quarters)


# ── Unit tests: compute_rmse_stats ───────────────────────────────────────────

class TestComputeRmseStats:
    """
    Verify RMSE computation on synthetic data with known properties.
    """

    def _make_results(
        self,
        n_normal: int = 10,
        n_covid: int = 4,
        dfm_error_normal: float = 1.0,
        dfm_error_covid: float = 20.0,
        ar1_error_normal: float = 2.0,
    ) -> pd.DataFrame:
        """
        Build a synthetic results DataFrame with controlled error values.
        All non-2020 quarters have constant error for predictable RMSE.
        """
        rows = []
        for i in range(n_normal):
            rows.append({
                "quarter_label":   f"2018 Q{i % 4 + 1}",
                "target_quarter":  pd.Timestamp("2018-01-01"),
                "evaluation_date": "2018-03-31",
                "dfm_nowcast_ann": 2.0,
                "ar1_nowcast_ann": 3.0,
                "actual_ann":      2.0 - dfm_error_normal,
                "dfm_error_ann":   dfm_error_normal,
                "ar1_error_ann":   ar1_error_normal,
                "is_2020":         False,
                "converged":       True,
            })
        for i in range(n_covid):
            rows.append({
                "quarter_label":   f"2020 Q{i+1}",
                "target_quarter":  pd.Timestamp(f"2020-{['01','04','07','10'][i]}-01"),
                "evaluation_date": "2020-03-31",
                "dfm_nowcast_ann": -5.0,
                "ar1_nowcast_ann": 2.0,
                "actual_ann":      -5.0 - dfm_error_covid,
                "dfm_error_ann":   dfm_error_covid,
                "ar1_error_ann":   20.0,
                "is_2020":         True,
                "converged":       True,
            })
        return pd.DataFrame(rows)

    def test_ex2020_rmse_excludes_covid_quarters(self):
        """
        dfm_ex2020_rmse must equal RMSE of the non-2020 rows only.
        With constant normal-period error = 1.0, RMSE = 1.0.
        """
        df = self._make_results(dfm_error_normal=1.0, dfm_error_covid=20.0)
        stats = compute_rmse_stats(df)
        assert abs(stats["dfm_ex2020_rmse"] - 1.0) < 1e-10, (
            f"ex-2020 RMSE should be 1.0, got {stats['dfm_ex2020_rmse']}"
        )

    def test_full_rmse_is_dominated_by_covid(self):
        """
        With large COVID errors (20 pp) and small normal errors (1 pp),
        the full-sample RMSE must be substantially larger than the ex-2020 RMSE.
        """
        df = self._make_results(dfm_error_normal=1.0, dfm_error_covid=20.0)
        stats = compute_rmse_stats(df)
        assert stats["dfm_full_rmse"] > stats["dfm_ex2020_rmse"] * 2, (
            "COVID quarters (20pp errors) should more than double the RMSE"
        )

    def test_skill_is_positive_when_dfm_wins(self):
        """
        dfm_skill_ex2020 = ar1_rmse − dfm_rmse.
        Positive = DFM beats AR(1); we want the DFM to win (positive skill).
        """
        df = self._make_results(dfm_error_normal=1.0, ar1_error_normal=2.0)
        stats = compute_rmse_stats(df)
        assert stats["dfm_skill_ex2020"] > 0, (
            "DFM (RMSE=1.0) should have positive skill vs AR(1) (RMSE=2.0)"
        )

    def test_n_counts_correct(self):
        """n_full, n_ex_2020, n_covid must sum correctly."""
        df = self._make_results(n_normal=10, n_covid=4)
        stats = compute_rmse_stats(df)
        assert stats["n_full"] == 14
        assert stats["n_ex_2020"] == 10
        assert stats["n_covid"] == 4

    def test_nan_errors_excluded_from_rmse(self):
        """NaN errors (failed fits) must be dropped before computing RMSE."""
        df = self._make_results(n_normal=4, n_covid=0, dfm_error_normal=1.0)
        # Introduce two NaN errors
        df.loc[df.index[:2], "dfm_error_ann"] = float("nan")
        stats = compute_rmse_stats(df)
        assert stats["n_full"] == 2  # only 2 valid errors
        assert np.isfinite(stats["dfm_full_rmse"])

    def test_rmse_is_nonnegative(self):
        """RMSE is always ≥ 0 by definition (sqrt of a mean of squares)."""
        df = self._make_results()
        stats = compute_rmse_stats(df)
        for key in ["dfm_full_rmse", "dfm_ex2020_rmse", "dfm_covid_rmse",
                    "ar1_full_rmse", "ar1_ex2020_rmse"]:
            val = stats[key]
            if not np.isnan(val):
                assert val >= 0, f"{key} = {val} < 0"


# ── Integration tests (require FRED API key + ~5 min runtime) ────────────────

@pytest.mark.integration
class TestBacktestIntegration:
    """
    Run a real 2-quarter backtest using ALFRED vintage data and verify:
      - No look-ahead bias (assertion passes)
      - ALFRED vintage panel ends before GDP advance release
      - Nowcasts are in a plausible range (−20% to +15% annualized)
      - AR(1) and DFM nowcasts are both produced
      - RMSE stats can be computed without error

    Evaluates Q4 2023 and Q1 2024 only (small window for test speed).
    Each DFM fit takes ~15–20 seconds, so this test runs in ~1 minute.
    """

    @pytest.fixture(autouse=True)
    def require_api_key(self):
        if not os.getenv("FRED_API_KEY"):
            pytest.skip("FRED_API_KEY not set — skipping integration test")

    def test_two_quarter_backtest_runs_without_error(self):
        """End-to-end backtest for Q4 2023 and Q1 2024."""
        from src.backtest.runner import run_backtest

        df = run_backtest(
            start_qs="2023-10-01",
            end_qs="2024-01-01",
            n_factors=1,     # 1-factor for speed in tests
            start_data="2015-01-01",
            use_cache=True,
            save_results=False,
        )

        assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
        assert "dfm_nowcast_ann" in df.columns
        assert "ar1_nowcast_ann" in df.columns
        assert "actual_ann" in df.columns

    def test_nowcasts_in_plausible_range(self):
        """DFM and AR(1) nowcasts must be in a plausible historical range."""
        from src.backtest.runner import run_backtest

        df = run_backtest(
            start_qs="2023-10-01",
            end_qs="2024-01-01",
            n_factors=1,
            start_data="2015-01-01",
            use_cache=True,
            save_results=False,
        )

        for _, row in df.iterrows():
            if row["converged"] and not np.isnan(row["dfm_nowcast_ann"]):
                assert -20.0 < row["dfm_nowcast_ann"] < 15.0, (
                    f"DFM nowcast {row['dfm_nowcast_ann']:.2f}% out of plausible range "
                    f"for {row['quarter_label']}"
                )

    def test_alfred_vintage_ends_before_gdp_release(self):
        """
        For each backtest quarter, the ALFRED vintage panel for GDPC1 must
        NOT contain GDP data for the target quarter — that would be look-ahead.

        We check this by verifying that the last non-NaN row in quarterly_df
        is the quarter BEFORE the target quarter.
        """
        from src.backtest.vintage import build_vintage_dfm_data

        pairs = backtest_quarter_range("2023-10-01", "2024-01-01")
        for target_q, eval_date in pairs:
            _, quarterly_df = build_vintage_dfm_data(
                as_of_date=eval_date,
                start="2015-01-01",
                use_cache=True,
            )

            last_actual_q = quarterly_df["GDPC1"].dropna().index[-1]
            expected_last = target_q - pd.DateOffset(months=3)

            assert last_actual_q == expected_last, (
                f"For target quarter {target_q.date()}, ALFRED vintage should "
                f"end at {expected_last.date()}, but last actual GDP is "
                f"{last_actual_q.date()}. LOOK-AHEAD BIAS DETECTED."
            )

    def test_compute_rmse_stats_works_on_real_results(self):
        """RMSE stats can be computed on real backtest output without error."""
        from src.backtest.runner import compute_rmse_stats, run_backtest

        df     = run_backtest(
            start_qs="2023-10-01",
            end_qs="2024-01-01",
            n_factors=1,
            start_data="2015-01-01",
            use_cache=True,
            save_results=False,
        )
        stats  = compute_rmse_stats(df)

        assert np.isfinite(stats["dfm_full_rmse"]) or np.isnan(stats["dfm_full_rmse"])
        assert stats["n_full"] >= 0
