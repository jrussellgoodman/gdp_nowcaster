"""
streamlit_app.py — US GDP Nowcasting Dashboard (Phase 6, visual redesign v2)

Four tabs:
  1. Live Nowcast        — current-quarter DFM prediction as the focal point,
                           with the 2015–2025 DFM-vs-GDPNow track record below
  2. The Factors         — smoothed global + real-activity factors with NBER
                           recession shading, plus factor loadings
  3. News Decomposition  — which data releases moved the nowcast (precomputed
                           Apr 30 → Jun 10, 2026 example loads instantly;
                           custom vintages compute live)
  4. Backtest            — 2015–2025 ALFRED vintage accuracy results

Visual identity (Phase 6):
  Dark navy page-header card · pill/segmented-control tab navigation ·
  dark hero-card for the live nowcast · kpi-card tiles with top accent band ·
  chart card containers (border + shadow on every Plotly chart) ·
  navy left-bar section headers.  All design tokens live in app/styles.css as
  CSS custom properties; Python colors in app/theme.py remain the single
  source of truth for chart palettes.

Heavy computation (EM fitting) runs once per session and is cached in memory
via st.cache_resource, so clicking between tabs never re-fits the model.

Model: 2-factor blocked DFM (global + real-activity factor).
Data:  FRED monthly indicators + quarterly GDPC1, disk-cached in data/.
       Backtest + GDPNow comparison + news example are pre-computed CSVs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import app.theme as th
from src.backtest.runner import compute_rmse_stats
from src.model.baseline import ar1_nowcast, fit_ar1
from src.model.dfm import (
    compute_news,
    extract_nowcast,
    fit_dfm,
    make_block_structure,
    prepare_dfm_data,
)

# ── Debug timing instrumentation (Phase 5.5 performance pass) ─────────────────
# Set NOWCAST_DEBUG_TIMING=1 to print per-section wall times to the terminal on
# every script rerun. Zero overhead when the flag is off. Used to measure the
# cost of widget-triggered reruns (every interaction re-executes this whole
# script and rebuilds all four tabs — tab CLICKS alone don't rerun anything,
# they are pure frontend).

DEBUG_TIMING = os.getenv("NOWCAST_DEBUG_TIMING", "0") == "1"
_timings: list[tuple[str, float]] = []


@contextmanager
def _timed(label: str):
    """Time a code section and record it when NOWCAST_DEBUG_TIMING=1."""
    if not DEBUG_TIMING:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _timings.append((label, (time.perf_counter() - t0) * 1000))


def _print_timings() -> None:
    """Dump the per-section timing table to stdout (terminal, not the page)."""
    if not DEBUG_TIMING or not _timings:
        return
    total = sum(ms for _, ms in _timings)
    print(f"\n[timing] ── rerun breakdown ({total:,.0f} ms total) " + "─" * 20)
    for label, ms in sorted(_timings, key=lambda kv: -kv[1]):
        print(f"[timing] {ms:>10,.1f} ms  {label}")


# ── Data file paths ────────────────────────────────────────────────────────────

DATA_DIR          = _PROJECT_ROOT / "data"
BACKTEST_2F_CSV   = DATA_DIR / "backtest_results_2factor.csv"
BACKTEST_1F_CSV   = DATA_DIR / "backtest_results.csv"
GDPNOW_EOQ_CSV    = DATA_DIR / "gdpnow_eoq.csv"
NEWS_EXAMPLE_CSV  = DATA_DIR / "news_example.csv"
NEWS_EXAMPLE_META = DATA_DIR / "news_example_meta.json"


# ── Caching ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_model():
    """
    Fit the 2-factor blocked DFM and return the results plus the data panels.

    st.cache_resource keeps the fitted statsmodels object in memory for the
    lifetime of the server process, so the ~30 s EM estimation runs once —
    not on every rerender or tab click.
    """
    monthly_df, quarterly_df = prepare_dfm_data(start="2000-01-01", use_cache=True)
    factor_blocks = make_block_structure(monthly_df, quarterly_df)
    results = fit_dfm(monthly_df, quarterly_df, factor_blocks=factor_blocks)
    return results, monthly_df, quarterly_df


@st.cache_data(show_spinner=False)
def _load_backtest(path: Path) -> pd.DataFrame:
    """Read a pre-computed backtest CSV (instant — no API calls)."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["target_quarter"] = pd.to_datetime(df["target_quarter"])
    return df.sort_values("target_quarter").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_gdpnow_eoq() -> pd.DataFrame:
    """End-of-quarter GDPNow vintages (pre-fetched by scripts/fetch_gdpnow.py)."""
    if not GDPNOW_EOQ_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(GDPNOW_EOQ_CSV)
    df["target_quarter"] = pd.to_datetime(df["target_quarter"])
    return df


@st.cache_data(show_spinner=False)
def _load_news_example():
    """The precomputed Apr 30 → Jun 10 news example (CSV + JSON sidecar)."""
    if not (NEWS_EXAMPLE_CSV.exists() and NEWS_EXAMPLE_META.exists()):
        return None, None
    tidy = pd.read_csv(NEWS_EXAMPLE_CSV, index_col="series")
    meta = json.loads(NEWS_EXAMPLE_META.read_text())
    return tidy, meta


@st.cache_data(ttl=3600, show_spinner=False)
def _latest_gdpnow():
    """
    Latest published GDPNow value, straight from FRED (display only — this
    number never enters our model). Cached for an hour; returns None if the
    API key is missing or the request fails, and the UI degrades gracefully.
    """
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        return None
    try:
        from fredapi import Fred
        s = Fred(api_key=api_key).get_series("GDPNOW").dropna()
        return {"value": float(s.iloc[-1]), "quarter": s.index[-1]}
    except Exception:
        return None


# ── Shared chart builders ──────────────────────────────────────────────────────

def _quarter_hover(name: str, labels) -> dict:
    """Per-quarter hover: '2020 Q3 · +30.1%' instead of a raw date."""
    return dict(
        customdata=list(labels),
        hovertemplate=f"{name}: %{{y:+.2f}}%<extra>%{{customdata}}</extra>",
    )


@st.cache_data(show_spinner=False)
def _impact_bar_chart(tidy: pd.DataFrame, target_quarter: str, height: int = 550) -> go.Figure:
    """
    The news-decomposition hero chart: one horizontal bar per data series,
    navy = the release pushed the GDP nowcast up, brick = pulled it down.
    `tidy` is indexed by FRED series ID with an 'impact_ann' column.
    Cached: the precomputed example renders on every rerun (Phase 5.5).
    """
    vals   = tidy["impact_ann"]
    labels = [th.label_for(s) for s in tidy.index]
    fig = go.Figure(go.Bar(
        x=vals, y=labels, orientation="h",
        marker=dict(color=th.diverging_colors(vals), line_width=0),
        text=[f"{v:+.3f}" for v in vals],
        textposition="outside",
        textfont=dict(size=12),
        cliponaxis=False,
        hovertemplate="%{y}: %{x:+.4f} pp<extra></extra>",
    ))
    th.apply_layout(
        fig,
        height=max(height, 60 * len(tidy) + 120),
        x_title=f"Impact on {target_quarter} GDP nowcast (annualized pp)",
        legend_top=False,
    )
    # Unified-x hover doesn't make sense for horizontal bars; headroom keeps
    # the outside text labels from being clipped at the axis edges.
    span = float(vals.abs().max()) if len(vals) else 1.0
    fig.update_layout(hovermode="closest", showlegend=False,
                      xaxis_range=[-span * 1.35, span * 1.35])
    fig.add_vline(x=0, line_color=th.INK, line_width=1)
    return fig


# ── Cached figure builders (Phase 5.5) ─────────────────────────────────────────
# Every widget interaction reruns this entire script, which used to rebuild all
# ~6 figures from scratch (~50 ms/rerun). st.cache_data returns the pickled
# figure instantly when the inputs haven't changed — and the inputs only change
# when the model refits or a CSV is regenerated. Functions are keyed on the
# actual DataFrames; the unhashable statsmodels results object is excluded from
# hashing (leading underscore) and stood in for by a (llf, nobs) fingerprint.


def _results_key(results) -> tuple:
    """Tiny hashable fingerprint of a fitted model, for cache keying.
    Changes whenever the model is re-estimated on different data."""
    return (float(results.llf), int(results.nobs))


@st.cache_data(show_spinner=False)
def _get_factor_df(_results, cache_key: tuple) -> pd.DataFrame:
    """Smoothed factors as a plain-Timestamp DataFrame (Plotly can't serialize
    the PeriodIndex statsmodels returns). Cached so the MultiIndex unwrap and
    index conversion don't run on every rerun."""
    fac_smooth = _results.factors.smoothed
    if isinstance(fac_smooth.columns, pd.MultiIndex):
        try:
            factor_df = fac_smooth.xs(0, level=1, axis=1)
        except KeyError:
            factor_df = fac_smooth.droplevel(1, axis=1)
    else:
        factor_df = fac_smooth
    if isinstance(factor_df.index, pd.PeriodIndex):
        factor_df = factor_df.copy()
        factor_df.index = factor_df.index.to_timestamp()
    return factor_df


@st.cache_data(show_spinner=False)
def _get_loadings(_results, cache_key: tuple) -> pd.DataFrame:
    """Loading coefficients as a DataFrame with 'global' and 'real' columns
    (NaN where a series doesn't load on the real factor). statsmodels names
    the params 'loading.{factor}->{series}'."""
    out: dict[str, dict[str, float]] = {}
    for factor in ("global", "real"):
        prefix = f"loading.{factor}->"
        out[factor] = {
            k[len(prefix):]: v
            for k, v in _results.params.items()
            if k.startswith(prefix)
        }
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def _build_track_record_fig(bt2: pd.DataFrame, gn: pd.DataFrame) -> go.Figure:
    """Tab 1 hero chart: DFM vs GDPNow vs actual, 2015–2025."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bt2["target_quarter"], y=bt2["actual_ann"],
        name="Actual GDP", mode="lines",
        line=dict(color=th.COLOR_ACTUAL, width=2.4),
        **_quarter_hover("Actual GDP", bt2["quarter_label"]),
    ))
    fig.add_trace(go.Scatter(
        x=bt2["target_quarter"], y=bt2["dfm_nowcast_ann"],
        name="DFM nowcast (this model)", mode="lines+markers",
        line=dict(color=th.COLOR_DFM, width=2.6),
        marker=dict(size=5),
        **_quarter_hover("DFM nowcast", bt2["quarter_label"]),
    ))
    if not gn.empty:
        fig.add_trace(go.Scatter(
            x=gn["target_quarter"], y=gn["gdpnow_ann"],
            name="Atlanta Fed GDPNow", mode="lines+markers",
            line=dict(color=th.COLOR_GDPNOW, width=2.2),
            marker=dict(size=5),
            connectgaps=False,
            **_quarter_hover("GDPNow", gn["quarter_label"]),
        ))
    fig.add_trace(go.Scatter(
        x=bt2["target_quarter"], y=bt2["ar1_nowcast_ann"],
        name="AR(1) baseline", mode="lines",
        line=dict(color=th.COLOR_AR1, width=1.4, dash="dot"),
        visible="legendonly",   # available, but off by default — reduces clutter
        **_quarter_hover("AR(1)", bt2["quarter_label"]),
    ))
    th.add_recession_bands(
        fig, periods=[("2020-01-01", "2021-01-01")], label="COVID 2020",
    )
    th.apply_layout(fig, height=600, y_title="Annualized quarterly growth (%)")
    return fig


@st.cache_data(show_spinner=False)
def _build_factor_fig(factor_df: pd.DataFrame) -> go.Figure:
    """Tab 2 time-series chart of the two smoothed factors, with the
    Yahoo-Finance-style navigation (pan / scroll-zoom / range slider)."""
    col_names     = list(factor_df.columns)[:2]
    factor_colors = [th.COLOR_FACTOR_GLOBAL, th.COLOR_FACTOR_REAL]
    factor_labels = ["Global factor", "Real-activity factor"]

    fig2 = go.Figure()
    for i, col in enumerate(col_names):
        fig2.add_trace(go.Scatter(
            x=factor_df.index, y=factor_df[col],
            name=factor_labels[i], mode="lines",
            line=dict(color=factor_colors[i], width=2.2 if i == 0 else 1.8),
            hovertemplate=f"{factor_labels[i]}: %{{y:.2f}}<extra>%{{x|%b %Y}}</extra>",
        ))
    th.add_recession_bands(fig2, label="NBER recession")
    th.apply_layout(fig2, height=600, y_title="Standardized factor value")
    th.make_time_explorable(fig2)

    # Annotate the most dramatic moment in the sample — the COVID collapse.
    g = factor_df[col_names[0]]
    trough = g.idxmin()
    if trough.year == 2020:
        fig2.add_annotation(
            x=trough, y=float(g.min()),
            text="<b>COVID collapse</b><br>sharpest drop in the sample —<br>"
                 "and an instant rebound, unlike 2008",
            showarrow=True, arrowhead=2, arrowcolor=th.TEXT_MUTED,
            ax=110, ay=10, align="left",
            font=dict(size=12, color=th.TEXT),
            bgcolor="rgba(255,255,255,0.85)", bordercolor=th.CARD_BORDER,
        )
    return fig2


@st.cache_data(show_spinner=False)
def _build_loadings_fig(loadings_df: pd.DataFrame) -> go.Figure:
    """Tab 2 grouped diverging bar chart of factor loadings."""
    global_loadings = loadings_df["global"].dropna()
    real_loadings   = loadings_df["real"].dropna()

    # Sort by global loading value (most negative → most positive) so the
    # "weakness" vs "strength" orientation reads clearly left-to-right.
    order  = global_loadings.sort_values().index
    labels = [th.label_for(s) for s in order]
    g_vals = [float(global_loadings[s]) for s in order]
    # Series that don't load on 'real' (surveys, CPI) get None so Plotly
    # skips those bars rather than drawing a zero bar.
    r_vals = [
        float(real_loadings[s]) if s in real_loadings.index else None
        for s in order
    ]

    max_abs = max(
        global_loadings.abs().max() if not global_loadings.empty else 1.0,
        real_loadings.abs().max()   if not real_loadings.empty   else 0.0,
    ) * 1.2

    fig_load = go.Figure()
    fig_load.add_trace(go.Bar(
        x=g_vals, y=labels, orientation="h",
        name="Global factor",
        marker=dict(color=th.COLOR_FACTOR_GLOBAL, opacity=0.88, line_width=0),
        hovertemplate="%{y}<br>Global loading: %{x:+.3f}<extra></extra>",
    ))
    fig_load.add_trace(go.Bar(
        x=r_vals, y=labels, orientation="h",
        name="Real-activity factor",
        marker=dict(color=th.COLOR_FACTOR_REAL, opacity=0.88, line_width=0),
        hovertemplate="%{y}<br>Real-activity loading: %{x:+.3f}<extra></extra>",
    ))
    th.apply_layout(fig_load, height=520, x_title="Loading coefficient",
                    legend_top=True)
    fig_load.update_layout(
        barmode="group",
        hovermode="closest",
        showlegend=True,
        xaxis_range=[-max_abs, max_abs],
    )
    fig_load.add_vline(x=0, line_color=th.INK, line_width=1)
    return fig_load


@st.cache_data(show_spinner=False)
def _build_backtest_fig(bt2: pd.DataFrame) -> go.Figure:
    """Tab 4 main chart: nowcast vs actual through time, worst miss flagged."""
    figb = go.Figure()
    figb.add_trace(go.Scatter(
        x=bt2["target_quarter"], y=bt2["actual_ann"],
        name="Actual GDP (current vintage)", mode="lines",
        line=dict(color=th.COLOR_ACTUAL, width=2.4),
        **_quarter_hover("Actual GDP", bt2["quarter_label"]),
    ))
    figb.add_trace(go.Scatter(
        x=bt2["target_quarter"], y=bt2["dfm_nowcast_ann"],
        name="DFM nowcast (2-factor)", mode="lines+markers",
        line=dict(color=th.COLOR_DFM, width=2.4, dash="dash"),
        marker=dict(size=5),
        **_quarter_hover("DFM nowcast", bt2["quarter_label"]),
    ))
    figb.add_trace(go.Scatter(
        x=bt2["target_quarter"], y=bt2["ar1_nowcast_ann"],
        name="AR(1) baseline", mode="lines",
        line=dict(color=th.COLOR_AR1, width=1.4, dash="dot"),
        **_quarter_hover("AR(1)", bt2["quarter_label"]),
    ))
    th.add_recession_bands(figb, periods=[("2020-01-01", "2021-01-01")],
                           label="COVID 2020")

    # Flag the single worst miss in the sample, per the "don't hide 2020"
    # design decision.
    q3_2020 = bt2[bt2["quarter_label"] == "2020 Q3"]
    if not q3_2020.empty:
        figb.add_annotation(
            x=q3_2020["target_quarter"].iloc[0],
            y=float(q3_2020["actual_ann"].iloc[0]),
            text="<b>Q3 2020: the model's worst miss (~43 pp)</b><br>"
                 "actual GDP rebounded +30% as the economy reopened;<br>"
                 "the DFM, trained on business cycles, predicted more collapse",
            showarrow=True, arrowhead=2, arrowcolor=th.TEXT_MUTED,
            ax=150, ay=-10, align="left",
            font=dict(size=12, color=th.TEXT),
            bgcolor="rgba(255,255,255,0.85)", bordercolor=th.CARD_BORDER,
        )
    th.apply_layout(figb, height=600, y_title="Annualized quarterly growth (%)")
    return figb


@st.cache_data(show_spinner=False)
def _build_scatter_fig(bt2: pd.DataFrame) -> go.Figure:
    """Tab 4 nowcast-vs-actual scatter, COVID quarters flagged not hidden."""
    ex20  = bt2[~bt2["is_2020"]]
    cov20 = bt2[bt2["is_2020"]]

    figs = go.Figure()
    figs.add_trace(go.Scatter(
        x=ex20["dfm_nowcast_ann"], y=ex20["actual_ann"],
        name="Normal quarters (40)", mode="markers",
        marker=dict(color=th.COLOR_DFM, size=9, opacity=0.75,
                    line=dict(color="white", width=1)),
        customdata=list(ex20["quarter_label"]),
        hovertemplate="%{customdata}<br>nowcast %{x:+.2f}% · actual %{y:+.2f}%<extra></extra>",
    ))
    figs.add_trace(go.Scatter(
        x=cov20["dfm_nowcast_ann"], y=cov20["actual_ann"],
        name="COVID 2020 quarters (4)", mode="markers+text",
        marker=dict(color="rgba(0,0,0,0)", size=11,
                    line=dict(color=th.BRICK, width=2)),
        text=cov20["quarter_label"].str.replace("2020 ", ""),
        textposition="top center",
        textfont=dict(size=11, color=th.BRICK),
        customdata=list(cov20["quarter_label"]),
        hovertemplate="%{customdata}<br>nowcast %{x:+.2f}% · actual %{y:+.2f}%<extra></extra>",
    ))
    lim = float(np.nanmax(np.abs(bt2[["dfm_nowcast_ann", "actual_ann"]].values))) * 1.12
    figs.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim], mode="lines", name="Perfect nowcast",
        line=dict(color=th.SLATE, width=1, dash="dash"), hoverinfo="skip",
    ))
    th.apply_layout(figs, height=560, x_title="DFM nowcast (annualized %)",
                    y_title="Actual GDP (annualized %)")
    figs.update_layout(hovermode="closest")
    figs.update_yaxes(scaleanchor="x", scaleratio=1)
    figs.update_xaxes(range=[-lim, lim])
    return figs


@st.cache_data(show_spinner=False)
def _build_comparison_table(bt2: pd.DataFrame, bt1: pd.DataFrame) -> pd.DataFrame:
    """Tab 4 model-comparison table (2f / 1f / AR(1) RMSE rows), as a plain
    DataFrame — the Styler is applied outside the cache (Stylers don't pickle)."""
    def _rmse(errors: pd.Series) -> float:
        e = errors.dropna()
        return float(np.sqrt((e ** 2).mean()))

    def _col(df: pd.DataFrame, err_col: str) -> list[str]:
        """One table column: RMSE ex-2020 / full / 2020-only + bias, formatted."""
        ex20  = df.loc[~df["is_2020"], err_col]
        covid = df.loc[df["is_2020"], err_col]
        return [
            f"{_rmse(ex20):.2f} pp",
            f"{_rmse(df[err_col]):.2f} pp",
            f"{_rmse(covid):.2f} pp",
            f"{df[err_col].mean():+.2f} pp",
        ]

    row_labels = ["RMSE — ex-2020 (primary)", "RMSE — full sample",
                  "RMSE — 2020 only", "Mean error (bias)"]
    table = pd.DataFrame(index=row_labels)
    table["DFM 2-factor (this app)"] = _col(bt2, "dfm_error_ann")
    if not bt1.empty:
        table["DFM 1-factor"] = _col(bt1, "dfm_error_ann")
    table["AR(1) baseline"] = _col(bt2, "ar1_error_ann")
    return table


def _render_news_result(tidy: pd.DataFrame, meta: dict, subtitle: str) -> None:
    """Before/after revision strip + hero impact chart + detail table."""
    rev = meta["revision_ann"]
    c1, c2, c3 = st.columns(3)
    c1.markdown(
        th.stat_card(f"Nowcast as of {meta['before_end']}",
                     f"{meta['before_nowcast_ann']:+.2f}%",
                     "annualized, Kalman filter", accent=th.SLATE),
        unsafe_allow_html=True,
    )
    c2.markdown(
        th.stat_card("Revision from new data", f"{rev:+.3f} pp",
                     "sum of all per-series impacts",
                     accent=th.COLOR_POS if rev >= 0 else th.COLOR_NEG),
        unsafe_allow_html=True,
    )
    c3.markdown(
        th.stat_card(f"Nowcast as of {meta['computed_on']}",
                     f"{meta['after_nowcast_ann']:+.2f}%",
                     "annualized, Kalman filter", accent=th.NAVY),
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="section-desc" style="margin-top:0.8rem;">{subtitle}</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _impact_bar_chart(tidy, meta["target_quarter"]),
        width="stretch", config=th.PLOTLY_BASE_CONFIG,
    )

    with st.expander("Detail table — surprise and impact per series"):
        detail = tidy.copy()
        detail.index = [th.label_for(s) for s in detail.index]
        detail = detail[["news", "impact_ann"]].rename(columns={
            "news":       "Surprise (sum of news)",
            "impact_ann": "Impact on nowcast (annualized pp)",
        })
        st.dataframe(detail.style.format("{:+.4f}"), width="stretch")


# ── Page setup ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="US GDP Nowcast — Dynamic Factor Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)
th.inject_css(st)

# ── Header / landing area ──────────────────────────────────────────────────────

st.markdown(
    """
    <div class="page-header">
      <div class="page-header-meta">Dynamic Factor Model &nbsp;·&nbsp; FRED &amp; ALFRED &nbsp;·&nbsp; 2015–2025</div>
      <div class="page-header-title">US GDP <span class="accent">Nowcast</span></div>
      <div class="page-header-sub">
        Official GDP figures arrive nearly a month after a quarter ends.
        A nowcast uses high-frequency monthly releases — jobs, factory output,
        retail sales — to estimate current-quarter growth <em>right now</em>, using a
        two-factor blocked dynamic factor model with Kalman filter and EM estimation,
        following the methodology of Bok&nbsp;et&nbsp;al.&nbsp;(2018), NY Fed Staff Nowcast.
      </div>
      <div class="page-header-tags">
        <span class="header-tag">2-Factor Blocked DFM</span>
        <span class="header-tag">Kalman Filter + EM</span>
        <span class="header-tag">11 Monthly Indicators</span>
        <span class="header-tag">44 Quarters Backtested</span>
        <span class="header-tag live">&#9679; Live FRED Data</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Fit the model once — every tab shares these objects.
with st.spinner("Downloading FRED data and fitting the 2-factor DFM — ~30 seconds on first load…"):
    try:
        with _timed("model_load (cached after first run)"):
            results, monthly_df, quarterly_df = _load_model()
        model_ok = True
    except Exception as _model_err:
        st.error(f"Model failed to load: {_model_err}")
        model_ok = False

with _timed("backtest_csv + rmse_stats"):
    bt2   = _load_backtest(BACKTEST_2F_CSV)
    stats = compute_rmse_stats(bt2) if not bt2.empty else None

# Current nowcast + AR(1) baseline (cheap once the model is fitted).
with _timed("nowcast_extract + ar1_fit"):
    nowcast = extract_nowcast(results, quarterly_df) if model_ok else None
    ar1_ann = None
    if model_ok:
        _gdp_released = quarterly_df["GDPC1"].dropna()
        _ar1_res = fit_ar1(_gdp_released)
        ar1_ann  = ar1_nowcast(_ar1_res, float(_gdp_released.iloc[-1]))["nowcast_ann_pct"]

# ── Headline stat cards ────────────────────────────────────────────────────────

h1, h2, h3, h4 = st.columns(4)
h1.markdown(
    th.stat_card(
        f"DFM nowcast — {nowcast['quarter']}" if nowcast else "DFM nowcast",
        f"{nowcast['nowcast_ann_pct']:+.2f}%" if nowcast else "—",
        "annualized real GDP growth", accent=th.NAVY,
    ),
    unsafe_allow_html=True,
)
h2.markdown(
    th.stat_card(
        "Backtest RMSE (ex-2020)",
        f"{stats['dfm_ex2020_rmse']:.2f} pp" if stats else "—",
        "vintage-faithful, 40 quarters", accent=th.AMBER,
    ),
    unsafe_allow_html=True,
)
h3.markdown(
    th.stat_card("Monthly indicators", "11 + GDP",
                 "employment · output · spending · surveys", accent=th.SLATE),
    unsafe_allow_html=True,
)
h4.markdown(
    th.stat_card("Quarters backtested", f"{len(bt2)}" if not bt2.empty else "—",
                 "2015 Q1 – 2025 Q4, ALFRED vintages", accent=th.INK),
    unsafe_allow_html=True,
)

st.markdown("<div style='height:0.8rem'></div>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📈  Live Nowcast",
    "📊  The Factors",
    "🗞  News",
    "📋  Backtest",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE NOWCAST
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not model_ok:
        st.error("Model unavailable — see error above.")
    elif nowcast is None:
        st.warning("No unreleased GDP quarter in the panel. Nothing to nowcast.")
    else:
        dfm_ann = nowcast["nowcast_ann_pct"]

        gdp_released = quarterly_df["GDPC1"].dropna()
        last_q   = gdp_released.index[-1]
        last_lbl = f"Q{(last_q.month - 1) // 3 + 1} {last_q.year}"
        last_ann = float(gdp_released.iloc[-1]) * 4

        gdpnow_live = _latest_gdpnow()

        # ── Hero: the nowcast itself, flanked by reference points ────────────
        hero_col, ref1, ref2, ref3 = st.columns([1.9, 1, 1, 1])
        hero_col.markdown(
            th.hero_number(
                f"DFM nowcast · {nowcast['quarter']} real GDP growth",
                f"{dfm_ann:+.2f}%",
                f"annualized · {nowcast['nowcast_qtr_pct']:+.3f}% quarterly · "
                "updates with each data release",
            ),
            unsafe_allow_html=True,
        )
        ref1.markdown(
            th.stat_card(
                "Atlanta Fed GDPNow",
                f"{gdpnow_live['value']:+.2f}%" if gdpnow_live else "n/a",
                "latest published estimate", accent=th.COLOR_GDPNOW,
            ),
            unsafe_allow_html=True,
        )
        ref2.markdown(
            th.stat_card("AR(1) baseline", f"{ar1_ann:+.2f}%",
                         "ignores all monthly data", accent=th.COLOR_AR1),
            unsafe_allow_html=True,
        )
        ref3.markdown(
            th.stat_card(f"Last actual — {last_lbl}", f"{last_ann:+.2f}%",
                         "official GDP, current vintage", accent=th.COLOR_ACTUAL),
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

        # ── Main chart: 10-year track record, DFM vs GDPNow vs actual ────────
        st.markdown('<div class="section-header">Track record — this model vs GDPNow, 2015–2025</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="section-desc">Every point uses only the data that existed '
            "on the last day of that quarter (ALFRED archival vintages) — both for our "
            "model and for GDPNow, so the comparison has no hindsight advantage. "
            "Click legend entries to show or hide lines; drag to zoom.</div>",
            unsafe_allow_html=True,
        )

        with _timed("tab1_track_record_fig_build"):
            gn  = _load_gdpnow_eoq()
            fig = _build_track_record_fig(bt2, gn)
        with _timed("tab1_track_record_fig_render"):
            st.plotly_chart(fig, width="stretch", config=th.PLOTLY_BASE_CONFIG)

        st.caption(
            "GDPNow values are unavailable before 2016 Q2 — FRED's archive (ALFRED) "
            "only began storing GDPNow vintages on 2016-05-17. "
            "“Actual GDP” is the current-vintage GDPC1 series, which includes later BEA "
            "revisions; the Phase 4 backtest found this inflicts a systematic ~2 pp "
            "under-prediction bias on the model (details in the Backtest tab)."
        )

        with st.expander("Latest monthly indicator data (model inputs, last 6 months)"):
            recent = monthly_df.tail(6).copy()
            recent.index = recent.index.strftime("%Y-%m")
            recent.columns = [th.label_for(c) for c in recent.columns]
            st.dataframe(recent.style.format("{:.3f}"), width="stretch")
            st.caption(
                "Values are stationarity-transformed (mostly month-over-month log "
                "differences; survey indices in levels). See src/data/fred_loader.py "
                "for the exact transformation per series."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — THE FACTORS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    if not model_ok:
        st.error("Model unavailable — see error above.")
    else:
        st.markdown('<div class="section-header">The two hidden factors behind the nowcast</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="section-desc">The model compresses all 11 monthly indicators '
            "into two latent factors, estimated by the Kalman smoother. The "
            "<b>global factor</b> (navy) loads on every series and acts as a "
            "model-derived business-cycle index — it falls in every recession. The "
            "<b>real-activity factor</b> (amber) loads only on “real-side” series "
            "(jobs, output, spending) and captures what they do beyond the common cycle. "
            "Gray bands are NBER-dated recessions.</div>",
            unsafe_allow_html=True,
        )

        # ── Smoothed factor chart (cached builders, Phase 5.5) ────────────────
        with _timed("tab2_factor_extract (PeriodIndex→Timestamp)"):
            factor_df = _get_factor_df(results, _results_key(results))
        with _timed("tab2_factor_fig_build"):
            fig2 = _build_factor_fig(factor_df)
        with _timed("tab2_factor_fig_render"):
            st.plotly_chart(fig2, width="stretch", config=th.PLOTLY_EXPLORE_CONFIG)
        st.caption(
            "Drag to pan · scroll to zoom · drag the mini-chart below the axis "
            "to jump around · 1Y/5Y/10Y/All buttons for quick ranges · "
            "double-click to reset."
        )

        st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

        # ── Factor loadings — combined diverging bar chart ───────────────────────
        st.markdown(
            '<div class="section-header">Factor loadings — how each indicator connects to the factors</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="section-desc">Each pair of bars shows how strongly an '
            "indicator co-moves with each factor after the EM algorithm converges. "
            "<b style='color:#14365D'>Navy = global factor</b> (overall business-cycle "
            "index that all 11 series load on); "
            "<b style='color:#E8A33D'>amber = real-activity factor</b> (labour, output "
            "and spending sub-index). Survey indices and CPI load only on the global "
            "factor, so they show a single navy bar.</div>",
            unsafe_allow_html=True,
        )

        with _timed("tab2_loadings_extract"):
            loadings_df = _get_loadings(results, _results_key(results))
        with _timed("tab2_loadings_fig_build"):
            fig_load = _build_loadings_fig(loadings_df)
        st.plotly_chart(fig_load, width="stretch", config=th.PLOTLY_BASE_CONFIG)

        st.caption(
            "**Sign conventions differ between the two factors.** "
            "The global factor uses an *economic-weakness* convention: when the "
            "global factor rises, conditions are deteriorating — so pro-growth "
            "series like Nonfarm Payrolls load **negatively** on it (more jobs = "
            "less weakness). The real-activity factor uses the opposite convention "
            "(*strong economy = positive*): Nonfarm Payrolls loads **positively** "
            "there. This sign flip is an EM identification artifact — factors are "
            "estimated only up to a sign reversal. What matters for the nowcast is "
            "the magnitude, not the direction."
        )

        with st.expander("Why does Disposable Income load negatively on the global factor?"):
            st.markdown(
                "The CARES Act and later stimulus packages sent disposable income "
                "sharply **up** in 2020, *during* the worst recession in decades. The "
                "model learned this counter-cyclical pattern from the data. The "
                "magnitude is small (|loading| ≈ 0.13), so it has little practical "
                "effect on the nowcast."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NEWS DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">What moved the nowcast?</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Every data release contains a <b>surprise</b> — '
        "the gap between what came in and what the model expected. Multiplying each "
        "surprise by that series' Kalman-gain weight gives its <b>impact</b> on the "
        "GDP nowcast, and the impacts provably sum to the total revision. This is the "
        "same decomposition the New York Fed publishes for its Staff Nowcast: "
        "it turns “the nowcast moved” into “<i>here is which release moved it, and by "
        "how much</i>.”</div>",
        unsafe_allow_html=True,
    )

    with _timed("tab3_example_load (cached CSV)"):
        example_tidy, example_meta = _load_news_example()

    if example_tidy is not None:
        with _timed("tab3_example_render"):
            _render_news_result(
                example_tidy, example_meta,
                subtitle=(
                    f"<b>Worked example — {example_meta['before_end']} → "
                    f"{example_meta['computed_on']}:</b> a quiet vintage window. The weak "
                    "May Philly Fed survey dragged the nowcast down, almost fully offset "
                    "by a strong NY Fed Empire State reading — a net revision of just "
                    f"{example_meta['revision_ann']:+.3f} pp."
                ),
            )
    else:
        st.info(
            "Precomputed example not found — run "
            "`python scripts/precompute_news_example.py` to generate it. "
            "You can still compute a custom window below."
        )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.divider()

    # ── Custom vintage comparison (live computation) ──────────────────────────
    st.markdown('<div class="section-header">Run your own vintage comparison</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">Pick a cutoff date: the model compares the dataset '
        "as it stood on that date against all data available today. Fitting the DFM "
        "twice takes ~60 seconds; the result is kept for the rest of the session.</div>",
        unsafe_allow_html=True,
    )

    if not model_ok:
        st.error("Model unavailable — news decomposition cannot run.")
    else:
        today = pd.Timestamp.today().normalize()
        default_before = today - pd.offsets.MonthBegin(2)

        before_input = st.date_input(
            "Old vintage cutoff ('before' date)",
            value=default_before,
            min_value=pd.Timestamp("2010-01-01"),
            max_value=today - pd.Timedelta(days=1),
        )
        before_str = str(before_input)

        if st.button("Compute news decomposition", type="primary"):
            with st.spinner("Fitting DFM on both vintages — ~60 seconds…"):
                try:
                    factor_blocks_news = make_block_structure(monthly_df, quarterly_df)
                    news_result = compute_news(
                        before_end=before_str,
                        factor_blocks=factor_blocks_news,
                        start="2000-01-01",
                        use_cache=True,
                    )
                    st.session_state["news_result"] = news_result
                    st.session_state["news_before"] = before_str
                except Exception as exc:
                    st.error(f"News computation failed: {exc}")
                    st.session_state["news_result"] = None

        news = st.session_state.get("news_result")
        if news is not None:
            impacts = news["impacts"]
            if impacts.empty:
                st.info("No new data releases in this vintage window.")
            else:
                # statsmodels names this index level "updated variable"
                # (space, not underscore).
                tidy_live = (
                    impacts.reset_index()
                    .groupby("updated variable")[["news", "impact"]]
                    .sum()
                    .sort_values("impact")
                    .rename(columns={"impact": "impact_qtr"})
                )
                tidy_live["impact_ann"] = tidy_live["impact_qtr"] * 4
                meta_live = {
                    "before_end":         st.session_state.get("news_before", before_str),
                    "computed_on":        today.strftime("%Y-%m-%d"),
                    "target_quarter":     news["target_quarter"],
                    "before_nowcast_ann": news["before_nowcast_ann"],
                    "after_nowcast_ann":  news["after_nowcast_ann"],
                    "revision_ann":       news["revision_ann"],
                }
                _render_news_result(
                    tidy_live, meta_live,
                    subtitle=f"<b>Custom window — {meta_live['before_end']} → today.</b>",
                )

    st.caption(
        "Technical note: news impacts use the one-sided Kalman filter (conditioning "
        "only on past data), while the Live Nowcast tab uses the smoother (all data). "
        "The two differ numerically — both are correct; they answer different questions."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BACKTEST
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Would it have worked? 2015–2025, with no hindsight</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="section-desc">For each of the 44 quarters, the model was re-fit '
        "using <b>only the data that existed on the last calendar day of that "
        "quarter</b>, fetched from ALFRED (the St. Louis Fed's archive of every "
        "historical data vintage). No look-ahead: the model never sees future "
        "observations or future revisions. Results are reported with <i>and</i> "
        "without the four 2020 COVID quarters — leaving the bad quarters in is part "
        "of the point.</div>",
        unsafe_allow_html=True,
    )

    if bt2.empty:
        st.error(f"Backtest file not found: {BACKTEST_2F_CSV}. "
                 "Run scripts/run_phase4_backtest_2factor.py first.")
        st.stop()

    # ── Headline metric cards ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        th.stat_card("RMSE — ex-2020 (primary)", f"{stats['dfm_ex2020_rmse']:.2f} pp",
                     "40 quarters, COVID excluded", accent=th.NAVY),
        unsafe_allow_html=True)
    c2.markdown(
        th.stat_card("RMSE — full sample", f"{stats['dfm_full_rmse']:.2f} pp",
                     "all 44 quarters incl. COVID", accent=th.SLATE),
        unsafe_allow_html=True)
    c3.markdown(
        th.stat_card("Bias (mean error)", f"{stats['dfm_mean_error']:+.2f} pp",
                     "vintage mismatch — see note", accent=th.BRICK),
        unsafe_allow_html=True)
    c4.markdown(
        th.stat_card("Skill vs AR(1), ex-2020", f"{stats['dfm_skill_ex2020']:+.2f} pp",
                     "negative = AR(1) wins", accent=th.AMBER),
        unsafe_allow_html=True)

    st.markdown("<div style='height:1.0rem'></div>", unsafe_allow_html=True)

    # ── Main chart: nowcast vs actual through time ─────────────────────────────
    with _timed("tab4_backtest_fig_build"):
        figb = _build_backtest_fig(bt2)
    with _timed("tab4_backtest_fig_render"):
        st.plotly_chart(figb, width="stretch", config=th.PLOTLY_BASE_CONFIG)

    # ── Styled comparison table ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model comparison</div>',
                unsafe_allow_html=True)

    bt1 = _load_backtest(BACKTEST_1F_CSV)

    with _timed("tab4_comparison_table"):
        table = _build_comparison_table(bt2, bt1)
        styled = (
            table.style
            .set_properties(**{"text-align": "right", "font-variant-numeric": "tabular-nums"})
            .set_properties(subset=pd.IndexSlice[["RMSE — ex-2020 (primary)"], :],
                            **{"font-weight": "700"})
            .set_properties(subset=pd.IndexSlice[["RMSE — ex-2020 (primary)"],
                                                 ["DFM 2-factor (this app)"]],
                            **{"background-color": th.NAVY, "color": "white"})
        )
        st.table(styled)

    st.markdown(
        '<div class="section-desc"><b>Reading this honestly:</b> the AR(1) baseline '
        "actually beats both DFMs ex-2020 — predicting “roughly average growth” is "
        "hard to beat in calm times with only 11 indicators (the NY Fed's production "
        "model uses 127 series). The DFM's value shows in turbulent periods (lower "
        "full-sample and 2020 RMSE) and in its interpretability: factors and news "
        "decompositions, which no AR(1) can provide. The consistent ~−2 pp bias is a "
        "data-vintage artifact: nowcasts were made from pre-revision ALFRED data but "
        "scored against today's GDP figures, which the 2023 BEA comprehensive "
        "revision lifted. Both models face it equally.</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Scatter: nowcast vs actual, 2020 flagged not hidden ───────────────────
    st.markdown('<div class="section-header">Nowcast vs actual, quarter by quarter</div>',
                unsafe_allow_html=True)

    with _timed("tab4_scatter_fig_build"):
        figs = _build_scatter_fig(bt2)
    with _timed("tab4_scatter_fig_render"):
        st.plotly_chart(figs, width="stretch", config=th.PLOTLY_BASE_CONFIG)
    st.caption(
        "Points above the dashed 45° line = the model under-predicted. The 2020 "
        "quarters (hollow brick markers) dwarf everything else — drag-select the "
        "central cluster to zoom into the normal-times fit. Their inclusion is "
        "deliberate: reporting accuracy only on easy quarters would be cherry-picking."
    )

    with st.expander("Full results table — all 44 quarters"):
        display = bt2[[
            "quarter_label", "dfm_nowcast_ann", "ar1_nowcast_ann",
            "actual_ann", "dfm_error_ann", "is_2020", "converged",
        ]].rename(columns={
            "quarter_label":   "Quarter",
            "dfm_nowcast_ann": "DFM (%)",
            "ar1_nowcast_ann": "AR(1) (%)",
            "actual_ann":      "Actual (%)",
            "dfm_error_ann":   "Error (pp)",
            "is_2020":         "COVID quarter",
            "converged":       "EM converged",
        })
        st.dataframe(
            display.style.format({
                "DFM (%)":    "{:+.2f}",
                "AR(1) (%)":  "{:+.2f}",
                "Actual (%)": "{:+.2f}",
                "Error (pp)": "{:+.2f}",
            }),
            width="stretch", hide_index=True,
        )


# ── Footer ─────────────────────────────────────────────────────────────────────
th.footer(st)

_print_timings()
