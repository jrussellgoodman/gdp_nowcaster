"""
theme.py — single source of truth for the dashboard's visual identity.

Why this file exists:
    A professional research dashboard uses the SAME color for the same concept
    in every chart (our DFM nowcast is always navy, GDPNow is always amber,
    recessions are always light-gray bands). Defining the palette, the Plotly
    layout defaults, and the CSS in one module means no chart can drift
    out of sync with the others.

Contents:
    * Palette constants            — hex codes, one per concept
    * SERIES_LABELS                — FRED codes → short human-readable names
    * NBER_RECESSIONS              — fixed recession dates for shading
    * apply_layout()               — shared Plotly look (fonts, grid, margins)
    * add_recession_bands()        — gray NBER bands on any Plotly figure
    * diverging_colors()           — navy/brick colors for +/− bar charts
    * stat_card() / hero_number()  — styled HTML metric components
    * inject_css()                 — page-level CSS (header, tabs, cards)
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

# ── Palette ────────────────────────────────────────────────────────────────────
# "Economic research" navy / slate / amber. Same hex = same meaning everywhere.

NAVY      = "#14365D"   # our DFM nowcast, global factor, positive impacts
AMBER     = "#E8A33D"   # Atlanta Fed GDPNow, real-activity factor
SLATE     = "#8A97A6"   # AR(1) baseline — deliberately de-emphasized
INK       = "#2B2B2B"   # actual GDP — near-black, "the truth"
BRICK     = "#B0413E"   # negative impacts/loadings, flagged outliers
RECESSION = "#94A3B8"   # NBER recession bands (low opacity)

BG        = "#FAFBFC"   # page background (matches .streamlit/config.toml)
TEXT      = "#1A2332"   # body text
TEXT_MUTED = "#5B6B7C"  # captions, secondary labels
CARD_BG   = "#FFFFFF"
CARD_BORDER = "#DDE3EA"
GRID      = "#E8ECF1"   # chart gridlines — barely-there

# Semantic aliases used by the app — prefer these in chart code so the intent
# is readable ("COLOR_DFM" not "NAVY").
COLOR_DFM    = NAVY
COLOR_GDPNOW = AMBER
COLOR_AR1    = SLATE
COLOR_ACTUAL = INK
COLOR_FACTOR_GLOBAL = NAVY
COLOR_FACTOR_REAL   = AMBER
COLOR_POS    = NAVY
COLOR_NEG    = BRICK

FONT_FAMILY = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Helvetica Neue', "
    "Arial, sans-serif"
)

# ── Human-readable series names ────────────────────────────────────────────────
# Short versions for chart axes (the full official names from
# src/data/fred_loader.py SERIES_CATALOG are too long for bar-chart labels).

SERIES_LABELS: dict[str, str] = {
    "GDPC1":              "Real GDP",
    "INDPRO":             "Industrial Production",
    "PAYEMS":             "Nonfarm Payrolls",
    "UNRATE":             "Unemployment Rate",
    "PCEC96":             "Real Consumption (PCE)",
    "RSAFS":              "Retail Sales",
    "HOUST":              "Housing Starts",
    "DGORDER":            "Durable Goods Orders",
    "DSPIC96":            "Disposable Income",
    "CPIAUCSL":           "CPI Inflation",
    "GACDISA066MSFRBNY":  "NY Fed Empire Survey",
    "GACDFSA066MSFRBPHI": "Philly Fed Survey",
}


def label_for(series_id: str) -> str:
    """Short human name for a FRED series ID (falls back to the raw ID)."""
    return SERIES_LABELS.get(series_id, series_id)


# ── NBER recessions ────────────────────────────────────────────────────────────
# Fixed historical dates — no need to fetch USREC from FRED.

NBER_RECESSIONS = [
    ("2001-03-01", "2001-11-01"),  # dot-com bust
    ("2007-12-01", "2009-06-01"),  # Global Financial Crisis
    ("2020-02-01", "2020-04-01"),  # COVID-19
]


# ── Plotly helpers ─────────────────────────────────────────────────────────────

def apply_layout(
    fig: go.Figure,
    *,
    height: int = 600,
    title: str | None = None,
    y_title: str | None = None,
    x_title: str | None = None,
    legend_top: bool = True,
) -> go.Figure:
    """
    Apply the shared dashboard look to a Plotly figure.

    Every chart in the app goes through this function so fonts, gridlines,
    margins and hover styling are identical everywhere. Returns the same
    figure for chaining.
    """
    # Only set a title when one is given — passing title=None explicitly makes
    # plotly.js render the string "undefined" in the corner of the chart.
    if title:
        fig.update_layout(
            title=dict(text=title, font=dict(size=17, color=TEXT, family=FONT_FAMILY),
                       x=0, xanchor="left")
        )
    fig.update_layout(
        height=height,
        font=dict(family=FONT_FAMILY, size=13, color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)",   # let the page background show through
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60 if title else 30, b=10),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white", bordercolor=CARD_BORDER,
            font=dict(family=FONT_FAMILY, size=12, color=TEXT),
        ),
        legend=(
            dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                 font=dict(size=12), bgcolor="rgba(0,0,0,0)")
            if legend_top else dict(font=dict(size=12))
        ),
    )
    fig.update_xaxes(
        title=x_title, showgrid=False, zeroline=False,
        showline=True, linecolor=CARD_BORDER, ticks="outside", tickcolor=CARD_BORDER,
    )
    fig.update_yaxes(
        title=y_title, gridcolor=GRID, zerolinecolor=SLATE, zerolinewidth=1,
    )
    return fig


def add_recession_bands(
    fig: go.Figure,
    *,
    periods: list[tuple[str, str]] | None = None,
    label: str | None = None,
) -> go.Figure:
    """
    Shade recession (or COVID) periods as light-gray vertical bands.

    Plotly vrects don't appear in the legend, so when `label` is given we add
    an invisible bar trace purely to create a legend entry.
    """
    for start, end in (periods or NBER_RECESSIONS):
        fig.add_vrect(
            x0=pd.Timestamp(start), x1=pd.Timestamp(end),
            fillcolor=RECESSION, opacity=0.18, layer="below", line_width=0,
        )
    if label:
        # Invisible point whose square marker stands in for the band color.
        # (A Bar trace here would corrupt the date axis with a category.)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], name=label, mode="markers",
            marker=dict(symbol="square", size=12, color=RECESSION, opacity=0.4),
            showlegend=True, hoverinfo="skip",
        ))
    return fig


# Default config for ordinary (non-explorable) charts: keep zoom/pan/reset and
# the PNG-download camera, drop the box/lasso-select and stepwise-zoom buttons
# nothing in this app uses. Smaller modebar = less hover clutter; perf impact
# is negligible and that's fine — this is a UX trim, not a speed fix.
PLOTLY_BASE_CONFIG: dict = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d",
    ],
}

# Pass this as `config=` to st.plotly_chart alongside make_time_explorable().
# scrollZoom     — mouse wheel / trackpad pinch zooms the time axis
# doubleClick    — double-click snaps back to the full date range
# displayModeBar — off: the hover toolbar floats over the top-right corner,
#                  exactly where the 1Y/5Y/10Y/All buttons live, and blocks
#                  clicks on them. Pan/zoom/reset are all native now anyway.
PLOTLY_EXPLORE_CONFIG: dict = {
    "scrollZoom": True,
    "doubleClick": "reset",
    "displayModeBar": False,
    "displaylogo": False,
}


def make_time_explorable(fig: go.Figure) -> go.Figure:
    """
    Yahoo-Finance-style navigation for a time-series chart:

      * click-and-drag pans (instead of Plotly's default box-zoom)
      * a mini range-slider below the x-axis for jumping anywhere in the sample
      * 1Y / 5Y / 10Y / All quick-range buttons (top-right, clear of the legend)

    Scroll-to-zoom and double-click-to-reset live in the browser-side config,
    not the figure — pass PLOTLY_EXPLORE_CONFIG to st.plotly_chart as well:

        th.make_time_explorable(fig)
        st.plotly_chart(fig, config=th.PLOTLY_EXPLORE_CONFIG)
    """
    fig.update_layout(dragmode="pan")
    # Plotly silently locks the y-axis whenever a rangeslider is shown.
    # Unlock it: otherwise scroll-zoom/pan act on the time axis only and the
    # 2020 COVID spike forever dictates the y-scale — the one thing zooming
    # is supposed to escape.
    fig.update_yaxes(fixedrange=False)
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.07,
                         bgcolor=BG, bordercolor=CARD_BORDER, borderwidth=1),
        rangeselector=dict(
            buttons=[
                dict(count=1,  label="1Y",  step="year", stepmode="backward"),
                dict(count=5,  label="5Y",  step="year", stepmode="backward"),
                dict(count=10, label="10Y", step="year", stepmode="backward"),
                dict(step="all", label="All"),
            ],
            # Top-right so the buttons don't collide with the top-left legend.
            x=1.0, xanchor="right", y=1.02, yanchor="bottom",
            bgcolor=CARD_BG, activecolor=GRID,
            bordercolor=CARD_BORDER, borderwidth=1,
            font=dict(size=12, color=TEXT),
        ),
    )
    return fig


def diverging_colors(values) -> list[str]:
    """Navy for non-negative values, brick red for negative — used by every
    +/− horizontal bar chart (loadings, news impacts) for consistency."""
    return [COLOR_POS if v >= 0 else COLOR_NEG for v in values]


# ── HTML components ────────────────────────────────────────────────────────────
# st.metric can't be resized or restyled much, so headline numbers are
# rendered as small HTML cards via st.markdown(unsafe_allow_html=True).

def stat_card(label: str, value: str, sub: str = "", accent: str = NAVY) -> str:
    """A small 'stat card': muted label on top, big value, optional footnote."""
    sub_html = f'<div class="stat-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="stat-card" style="border-top: 3px solid {accent};">
      <div class="stat-label">{label}</div>
      <div class="stat-value" style="color:{accent};">{value}</div>
      {sub_html}
    </div>
    """


def hero_number(label: str, value: str, sub: str = "", color: str = NAVY) -> str:
    """The Live Nowcast focal point — a very large number with context lines."""
    sub_html = f'<div class="hero-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="hero-card">
      <div class="hero-label">{label}</div>
      <div class="hero-value" style="color:{color};">{value}</div>
      {sub_html}
    </div>
    """


# ── Page CSS ───────────────────────────────────────────────────────────────────

_CSS = f"""
<style>
/* Tighter top padding; comfortable max width for reading */
.block-container {{
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}}

/* Hide Streamlit chrome — this is a research product, not a dev tool */
#MainMenu, footer, .stDeployButton {{ visibility: hidden; }}

/* ── Header ─────────────────────────────────────────────── */
.app-title {{
    font-size: 2.3rem;
    font-weight: 750;
    letter-spacing: -0.02em;
    color: {TEXT};
    margin-bottom: 0.1rem;
}}
.app-title .accent {{ color: {NAVY}; }}
.app-byline {{
    font-size: 0.86rem;
    color: {TEXT_MUTED};
    margin-bottom: 0.9rem;
}}
.app-intro {{
    font-size: 1.0rem;
    line-height: 1.55;
    color: {TEXT};
    max-width: 60rem;
    margin-bottom: 0.4rem;
}}

/* ── Stat cards ─────────────────────────────────────────── */
.stat-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 8px;
    padding: 0.9rem 1.1rem 0.8rem 1.1rem;
    box-shadow: 0 1px 2px rgba(20, 54, 93, 0.06);
    height: 100%;
}}
.stat-label {{
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {TEXT_MUTED};
    margin-bottom: 0.15rem;
}}
.stat-value {{
    font-size: 1.75rem;
    font-weight: 720;
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
}}
.stat-sub {{
    font-size: 0.76rem;
    color: {TEXT_MUTED};
    margin-top: 0.2rem;
}}

/* ── Hero number (Live Nowcast) ─────────────────────────── */
.hero-card {{
    background: {CARD_BG};
    border: 1px solid {CARD_BORDER};
    border-radius: 10px;
    padding: 1.4rem 1.6rem 1.2rem 1.6rem;
    box-shadow: 0 1px 3px rgba(20, 54, 93, 0.08);
    text-align: center;
    height: 100%;
}}
.hero-label {{
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: {TEXT_MUTED};
}}
.hero-value {{
    font-size: 4.2rem;
    font-weight: 760;
    line-height: 1.05;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}}
.hero-sub {{
    font-size: 0.84rem;
    color: {TEXT_MUTED};
    margin-top: 0.3rem;
}}

/* ── Tabs: larger, navy active underline ────────────────── */
.stTabs [data-baseweb="tab-list"] {{
    gap: 0.4rem;
    border-bottom: 1px solid {CARD_BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 1.0rem;
    font-weight: 600;
    padding: 0.55rem 1.0rem;
    color: {TEXT_MUTED};
}}
.stTabs [aria-selected="true"] {{
    color: {NAVY};
}}

/* Section headers inside tabs */
.section-header {{
    font-size: 1.28rem;
    font-weight: 700;
    color: {TEXT};
    margin: 0.4rem 0 0.2rem 0;
}}
.section-desc {{
    font-size: 0.94rem;
    line-height: 1.5;
    color: {TEXT_MUTED};
    max-width: 56rem;
    margin-bottom: 0.6rem;
}}

/* ── Footer ─────────────────────────────────────────────── */
.app-footer {{
    margin-top: 2.5rem;
    padding-top: 0.9rem;
    border-top: 1px solid {CARD_BORDER};
    font-size: 0.78rem;
    line-height: 1.6;
    color: {TEXT_MUTED};
}}
.app-footer a {{ color: {NAVY}; }}
</style>
"""


def inject_css(st_module) -> None:
    """Inject the page CSS once. Call right after st.set_page_config()."""
    st_module.markdown(_CSS, unsafe_allow_html=True)


def footer(st_module) -> None:
    """Standard footer: data sources + methodology citation."""
    st_module.markdown(
        """
        <div class="app-footer">
        <b>Data:</b> FRED &amp; ALFRED (Federal Reserve Bank of St.&nbsp;Louis) ·
        Atlanta Fed GDPNow ·
        <b>Methodology:</b> Bok, Caratelli, Giannone, Sbordone &amp; Tambalotti (2018),
        <i>"Macroeconomic Nowcasting and Forecasting with Big Data"</i>, FRBNY Staff Report 830 ·
        Estimated with statsmodels <code>DynamicFactorMQ</code> (Kalman filter + EM).<br/>
        Nowcasts are a student research project, not official forecasts.
        </div>
        """,
        unsafe_allow_html=True,
    )
