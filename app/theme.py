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
    * inject_css()                 — loads app/styles.css once per rerun;
                                     injects Google Fonts + CSS design tokens
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# ── Palette ────────────────────────────────────────────────────────────────────
# "Economic research" navy / slate / amber. Same hex = same meaning everywhere.
# These are also exported as CSS custom properties by inject_css().

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
CARD_BORDER = "#CDD5DF"  # slightly more visible than previous #DDE3EA
GRID      = "#E2E8F0"   # chart gridlines — barely-there

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

# Inter first — loaded as a webfont by inject_css(); system fonts are fallbacks.
FONT_FAMILY = (
    "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', "
    "'Helvetica Neue', Arial, sans-serif"
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
    if title:
        fig.update_layout(
            title=dict(text=title, font=dict(size=17, color=TEXT, family=FONT_FAMILY),
                       x=0, xanchor="left")
        )
    fig.update_layout(
        height=height,
        font=dict(family=FONT_FAMILY, size=12.5, color=TEXT),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=60 if title else 34, b=8),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="white", bordercolor=CARD_BORDER,
            font=dict(family=FONT_FAMILY, size=12, color=TEXT),
        ),
        legend=(
            dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                 font=dict(size=11.5), bgcolor="rgba(0,0,0,0)")
            if legend_top else dict(font=dict(size=11.5))
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
        fig.add_trace(go.Scatter(
            x=[None], y=[None], name=label, mode="markers",
            marker=dict(symbol="square", size=12, color=RECESSION, opacity=0.4),
            showlegend=True, hoverinfo="skip",
        ))
    return fig


# Default config for ordinary (non-explorable) charts: keep zoom/pan/reset and
# the PNG-download camera, drop the box/lasso-select and stepwise-zoom buttons.
PLOTLY_BASE_CONFIG: dict = {
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "zoomIn2d", "zoomOut2d", "autoScale2d",
    ],
}

# Pass alongside make_time_explorable() on the Factors chart.
# displayModeBar off: the hover toolbar sits exactly over the range buttons.
PLOTLY_EXPLORE_CONFIG: dict = {
    "scrollZoom": True,
    "doubleClick": "reset",
    "displayModeBar": False,
    "displaylogo": False,
}


def make_time_explorable(fig: go.Figure) -> go.Figure:
    """
    Yahoo-Finance-style navigation for a time-series chart:
      * click-and-drag pans instead of Plotly's default box-zoom
      * mini range-slider below the x-axis for jumping anywhere in the sample
      * 1Y / 5Y / 10Y / All quick-range buttons (top-right)

    Pass PLOTLY_EXPLORE_CONFIG to st.plotly_chart() as well.
    """
    fig.update_layout(dragmode="pan")
    # Plotly silently locks the y-axis whenever a rangeslider is shown.
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
            x=0.99, xanchor="right", y=1.02, yanchor="bottom",
            bgcolor=CARD_BG, activecolor=GRID,
            bordercolor=CARD_BORDER, borderwidth=1,
            font=dict(size=12, color=TEXT),
        ),
    )
    return fig


def diverging_colors(values) -> list[str]:
    """Navy for non-negative values, brick red for negative."""
    return [COLOR_POS if v >= 0 else COLOR_NEG for v in values]


def focused_yaxis(
    fig: go.Figure,
    series_list: list[pd.Series],
    *,
    exclude: tuple[str, str] = ("2020-01-01", "2021-01-01"),
    pad: float = 0.16,
    note: str | None = None,
    note_position: str = "bottom",
    toggle_y: float = 1.10,
    margin_top: int = 80,
) -> go.Figure:
    """
    Default the y-axis to the data's NORMAL range, with a Focused/Full toggle.

    The post-2020 charting problem: COVID observations are 5–10× larger than
    anything else in a macro sample. Autoscaling the y-axis to include them
    compresses all other variation into a flat stripe — the chart becomes
    unreadable for the 95% of history the reader actually wants to compare.

    Standard editorial practice (FT, The Economist, Fed research notes) is to
    clip the axis to the normal range, print the off-scale extremes as text,
    and let the reader opt into the full range. This helper implements that:

      * y-axis defaults to the min/max of the data OUTSIDE the `exclude`
        window (COVID 2020 by default), padded by `pad`;
      * a "Focused / Full range" button pair (top-right) switches scales;
      * `note` (if given) is pinned near the excluded window, so the clipped
        extremes are stated, not hidden.

    If the excluded window doesn't actually extend beyond the normal range
    (nothing would be clipped), the toggle is skipped entirely.
    """
    included, excluded_vals = [], []
    x0, x1 = pd.Timestamp(exclude[0]), pd.Timestamp(exclude[1])
    for s in series_list:
        s = s.dropna()
        mask = (s.index >= x0) & (s.index < x1)
        included.append(s[~mask])
        excluded_vals.append(s[mask])

    inc = pd.concat(included)
    exc = pd.concat(excluded_vals) if excluded_vals else pd.Series(dtype=float)
    if inc.empty:
        return fig

    span = float(inc.max() - inc.min()) or 1.0
    lo   = float(inc.min()) - span * pad
    hi   = float(inc.max()) + span * pad

    # Nothing outside the focused range → plain autoscale, no toggle needed.
    if exc.empty or (exc.min() >= lo and exc.max() <= hi):
        return fig

    fig.update_layout(
        yaxis_range=[lo, hi],
        # extra headroom so the toggle row sits above the legend, not on it
        margin=dict(t=margin_top),
        updatemenus=[dict(
            type="buttons", direction="right",
            x=0.99, xanchor="right", y=toggle_y, yanchor="bottom",
            pad=dict(r=0, t=0, b=0, l=0),
            bgcolor=CARD_BG, bordercolor=CARD_BORDER, borderwidth=1,
            font=dict(size=11.5, color=TEXT, family=FONT_FAMILY),
            buttons=[
                dict(label="Focused scale", method="relayout",
                     args=[{"yaxis.range": [lo, hi]}]),
                dict(label="Full range", method="relayout",
                     args=[{"yaxis.autorange": True}]),
            ],
        )],
    )

    if note:
        y_note = lo + span * 0.06 if note_position == "bottom" else hi - span * 0.06
        fig.add_annotation(
            x=x0 + (x1 - x0) / 2, y=y_note,
            text=note, showarrow=False,
            font=dict(size=11, color=TEXT_MUTED, family=FONT_FAMILY),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor=CARD_BORDER, borderwidth=1, borderpad=4,
        )
    return fig


# ── HTML components ────────────────────────────────────────────────────────────

def stat_card(label: str, value: str, sub: str = "", accent: str = NAVY) -> str:
    """
    A KPI metric tile: tiny uppercase label, large number, optional footnote.

    --kpi-accent drives the 3px top band color and the value color.
    CSS class: .kpi-card (see app/styles.css).
    """
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="kpi-card" style="--kpi-accent: {accent};">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {sub_html}
    </div>
    """


def hero_number(label: str, value: str, sub: str = "", color: str = NAVY) -> str:
    """
    The Live Nowcast focal point — dark navy card with a very large white number.

    `color` is kept for backward-compatibility but is not used: the dark
    background means the value is always white (CSS `.hero-value { color: white }`).
    CSS class: .hero-card (see app/styles.css).
    """
    sub_html = f'<div class="hero-sub">{sub}</div>' if sub else ""
    return f"""
    <div class="hero-card">
      <div class="hero-label">{label}</div>
      <div class="hero-value">{value}</div>
      {sub_html}
    </div>
    """


# ── CSS injection ──────────────────────────────────────────────────────────────

def inject_css(st_module) -> None:
    """
    Inject the full page stylesheet once, right after st.set_page_config().

    Structure of the injected <style> block:
      1. @import for Inter (Google Fonts) — must be first in the block
      2. :root {} CSS custom properties derived from the Python color constants
         (so Python is the single source of truth for all colors)
      3. The contents of app/styles.css — pure portable CSS using var()
    """
    styles_content = (Path(__file__).parent / "styles.css").read_text()

    # CSS custom properties — every var() in styles.css resolves to these.
    # Shadows use rgba values matched to NAVY so they tint navy rather than gray.
    css_vars = (
        f":root {{\n"
        f"  --navy:        {NAVY};\n"
        f"  --amber:       {AMBER};\n"
        f"  --slate:       {SLATE};\n"
        f"  --ink:         {INK};\n"
        f"  --brick:       {BRICK};\n"
        f"  --bg:          {BG};\n"
        f"  --text:        {TEXT};\n"
        f"  --text-muted:  {TEXT_MUTED};\n"
        f"  --card-bg:     {CARD_BG};\n"
        f"  --card-border: {CARD_BORDER};\n"
        f"  --grid:        {GRID};\n"
        f"  --radius:      12px;\n"
        f"  --radius-sm:   8px;\n"
        f"  --radius-lg:   16px;\n"
        f"  --shadow-sm:   0 1px 3px rgba(20,54,93,0.07), 0 1px 2px rgba(0,0,0,0.04);\n"
        f"  --shadow-md:   0 4px 12px rgba(20,54,93,0.11), 0 1px 3px rgba(0,0,0,0.05);\n"
        f"  --shadow-lg:   0 8px 24px rgba(20,54,93,0.15), 0 2px 6px rgba(0,0,0,0.06);\n"
        f"}}\n"
    )

    # @import must be the very first rule in its <style> block.
    font_import = (
        "@import url('https://fonts.googleapis.com/css2?family=Inter:"
        "ital,wght@0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap');\n"
    )

    full_css = font_import + css_vars + styles_content
    st_module.markdown(f"<style>{full_css}</style>", unsafe_allow_html=True)


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
