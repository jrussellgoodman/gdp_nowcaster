"""
baseline.py — AR(1) benchmark model for quarterly GDP growth.

What this is:
  The simplest possible GDP forecasting model: predict next quarter's real GDP
  growth using only last quarter's growth. It ignores every other economic
  indicator (jobs, production, spending, etc.).

  Why build such a simple model at all?
  This is our "benchmark." If the fancy DFM model (which uses 12+ indicators)
  can't beat a model that only looks at past GDP, then the extra complexity
  isn't adding value. Good forecasting papers always report how much better
  the fancy model is compared to a simple baseline.

  The AR(1) model says:
      gdp_t = α + β × gdp_{t-1} + ε_t

  where:
    gdp_t   = quarterly log-difference of real GDP (≈ quarterly % growth)
    α       = a constant (absorbs the long-run average growth rate)
    β       = the AR coefficient (how much momentum GDP has)
    ε_t     = noise (things the model can't explain)

  A β close to 0 means this quarter's growth gives little information about
  next quarter's. A β close to 1 would mean a near-random-walk.
  Empirically, quarterly US GDP growth has β around 0.2–0.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_ar1(gdp_growth: pd.Series) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit an AR(1) model to a series of quarterly GDP growth rates via OLS.

    Why OLS and not a more exotic method?
      For a simple AR(1), OLS and maximum likelihood give essentially the same
      answer. OLS is transparent, well-understood, and easy to sanity-check.

    Parameters
    ----------
    gdp_growth : pd.Series
        Quarterly log-differences of real GDP (output of transform_series with
        "log_diff"). Units: percent (e.g., 0.5 = 0.5% quarterly growth).
        NaN in the first element is dropped automatically.

    Returns
    -------
    statsmodels OLS results object.
      Key attributes:
        results.params['const']    → α (intercept)
        results.params['gdp_lag1'] → β (AR coefficient)
        results.rsquared           → R² (fraction of variance explained)
    """
    clean = gdp_growth.dropna()

    # Build y (current quarter) and X (lagged quarter).
    # We reset the index so statsmodels doesn't raise "indices not aligned"
    # — the two slices have different DatetimeIndex timestamps by construction
    # (y starts one period after X), so we strip timestamps and work with
    # positional integers instead.
    y     = clean.iloc[1:].reset_index(drop=True)
    X_lag = clean.iloc[:-1].reset_index(drop=True)
    X_lag.name = "gdp_lag1"

    # Add a constant (intercept) for the α term
    X = sm.add_constant(X_lag)

    results = sm.OLS(y, X).fit()
    return results


def ar1_nowcast(results, last_gdp_growth: float) -> dict:
    """
    Produce a one-step-ahead nowcast from a fitted AR(1) model.

    In plain English:
      "Given that last quarter's GDP growth was X%, our best guess for this
      quarter's GDP growth is α + β × X%."

    Parameters
    ----------
    results : OLS results
        Fitted AR(1) model from fit_ar1().
    last_gdp_growth : float
        The most recently observed quarterly GDP growth rate (in %).
        This is the "gdp_{t-1}" in the AR(1) formula.

    Returns
    -------
    dict with keys:
        'nowcast_qtr_pct'  : predicted quarterly % growth (not annualized)
        'nowcast_ann_pct'  : same × 4  (annualized, for comparison to news/GDPNow)
        'intercept'        : α
        'ar_coef'          : β
    """
    alpha = results.params["const"]
    beta  = results.params["gdp_lag1"]
    nowcast_qtr = alpha + beta * last_gdp_growth

    return {
        "nowcast_qtr_pct": nowcast_qtr,
        "nowcast_ann_pct": nowcast_qtr * 4,
        "intercept":       alpha,
        "ar_coef":         beta,
    }


def print_ar1_summary(results, nowcast: dict | None = None) -> None:
    """Print a plain-English summary of the fitted AR(1) model."""
    alpha = results.params["const"]
    beta  = results.params["gdp_lag1"]
    r2    = results.rsquared
    nobs  = int(results.nobs)

    print("=" * 55)
    print("AR(1) Baseline — Quarterly Real GDP Growth")
    print("=" * 55)
    print(f"  Observations  : {nobs} quarters")
    print(f"  Intercept (α) : {alpha:.4f}%")
    print(f"  AR coef  (β)  : {beta:.4f}")
    print(f"  R²            : {r2:.4f}  (model explains {r2*100:.1f}% of variance)")
    print(f"  Long-run mean : {alpha / (1 - beta):.4f}%/qtr  "
          f"= {alpha / (1 - beta) * 4:.2f}%/yr annualized")
    if nowcast:
        print()
        print(f"  Nowcast (this quarter):")
        print(f"    Quarterly  : {nowcast['nowcast_qtr_pct']:>+.4f}%")
        print(f"    Annualized : {nowcast['nowcast_ann_pct']:>+.2f}%")
    print("=" * 55)
