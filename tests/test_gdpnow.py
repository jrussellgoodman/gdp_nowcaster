"""
test_gdpnow.py — checks on the GDPNow comparison data (data/gdpnow_eoq.csv).

These tests validate the OUTPUT FILE, not the ALFRED API, so they run offline
and never spend API calls. If the CSV is missing, the whole module is skipped
with a pointer to the script that creates it.

What we check:
  1. Structure  — one row per backtest quarter, aligned with the backtest CSV.
  2. Vintage discipline — NaN exactly where ALFRED has no GDPNow archive
     (first vintage is 2016-05-17), values everywhere else.
  3. Plausibility — all values inside the historically observed GDPNow range.
  4. Benchmark spot-checks — two quarters compared against publicly documented
     Atlanta Fed values (the COVID extremes, which are easy to verify).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GDPNOW_CSV   = Path(__file__).resolve().parents[1] / "data" / "gdpnow_eoq.csv"
BACKTEST_CSV = Path(__file__).resolve().parents[1] / "data" / "backtest_results_2factor.csv"

pytestmark = pytest.mark.skipif(
    not GDPNOW_CSV.exists(),
    reason="data/gdpnow_eoq.csv not found — run scripts/fetch_gdpnow.py first",
)

# First GDPNow vintage archived in ALFRED (see scripts/fetch_gdpnow.py).
ALFRED_GDPNOW_START = pd.Timestamp("2016-05-17")


@pytest.fixture(scope="module")
def gdpnow() -> pd.DataFrame:
    return pd.read_csv(GDPNOW_CSV, parse_dates=["target_quarter", "evaluation_date"])


def test_aligned_with_backtest(gdpnow):
    """Every backtest quarter must appear exactly once, in the same order."""
    bt = pd.read_csv(BACKTEST_CSV)
    assert len(gdpnow) == len(bt)
    assert list(gdpnow["quarter_label"]) == list(bt["quarter_label"])


def test_nan_exactly_for_pre_alfred_vintages(gdpnow):
    """
    NaN iff the evaluation date predates ALFRED's first GDPNow vintage.
    A NaN after that date would mean the fetch silently failed; a value
    before it would mean look-ahead (impossible data).
    """
    pre = gdpnow["evaluation_date"] < ALFRED_GDPNOW_START
    assert gdpnow.loc[pre, "gdpnow_ann"].isna().all(), "value before first vintage = look-ahead"
    assert gdpnow.loc[~pre, "gdpnow_ann"].notna().all(), "missing value in covered period"


def test_values_plausible(gdpnow):
    """
    GDPNow's most extreme readings ever were the 2020 COVID quarters
    (≈ −54% at the trough of Q2 2020 tracking, ≈ +35% during the Q3 rebound).
    Anything outside ±60 would indicate a units bug (e.g., quarterly vs
    annualized, or fraction vs percent).
    """
    vals = gdpnow["gdpnow_ann"].dropna()
    assert (vals.abs() < 60).all()
    # Non-COVID quarters live in a much tighter band.
    non_covid = gdpnow.loc[~gdpnow["quarter_label"].str.startswith("2020"), "gdpnow_ann"].dropna()
    assert (non_covid.abs() < 15).all()


def test_covid_benchmark_values(gdpnow):
    """
    Spot-check against publicly documented Atlanta Fed history:
      * End of Q2 2020 (June 30 vintage): GDPNow tracked roughly −39% SAAR.
      * End of Q3 2020 (Sep 30 vintage): GDPNow tracked roughly +32% SAAR.
    We allow ±3 pp — the exact value depends on the precise vintage day,
    but sign and magnitude must match the published record.
    """
    by_label = gdpnow.set_index("quarter_label")["gdpnow_ann"]
    assert by_label["2020 Q2"] == pytest.approx(-39.5, abs=3.0)
    assert by_label["2020 Q3"] == pytest.approx(32.0, abs=3.0)


def test_signs_match_known_quarters(gdpnow):
    """Directional sanity: 2020 Q2 negative (lockdown crash), 2020 Q3 positive
    (reopening rebound), 2025 Q1 negative (import-surge distortion)."""
    by_label = gdpnow.set_index("quarter_label")["gdpnow_ann"]
    assert by_label["2020 Q2"] < 0
    assert by_label["2020 Q3"] > 0
    assert by_label["2025 Q1"] < 0
