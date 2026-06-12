# GDP Nowcast Dashboard — Design System

**Version:** Phase 6 (2026-06-11)  
**Source of truth:** `app/styles.css` (pure CSS) + `app/theme.py` (Python color constants)

---

## Color palette

All colors live as Python constants in `app/theme.py` and are exported as CSS
custom properties by `inject_css()`. Chart code uses the Python constants;
CSS uses the `var()` aliases.

| Constant | Hex | Semantic role |
|---|---|---|
| `NAVY` | `#14365D` | DFM nowcast, global factor, positive impacts |
| `AMBER` | `#E8A33D` | GDPNow, real-activity factor, key accent |
| `SLATE` | `#8A97A6` | AR(1) baseline (deliberately de-emphasized) |
| `INK` | `#2B2B2B` | Actual GDP — "the truth" |
| `BRICK` | `#B0413E` | Negative impacts, flagged outliers |
| `RECESSION` | `#94A3B8` | NBER recession bands (low opacity) |
| `BG` | `#FAFBFC` | Page background |
| `TEXT` | `#1A2332` | Body text |
| `TEXT_MUTED` | `#5B6B7C` | Captions, labels, secondary text |
| `CARD_BG` | `#FFFFFF` | Card backgrounds |
| `CARD_BORDER` | `#CDD5DF` | Card borders, dividers |
| `GRID` | `#E2E8F0` | Chart gridlines |

**Rule:** One hex per concept. Never use a color for a purpose other than its
semantic role — a reader should be able to identify the DFM line in any chart
without a legend.

---

## Spacing & radius tokens

Defined in the `:root {}` block injected by `inject_css()`:

```css
--radius:    12px   /* cards, chart containers, buttons */
--radius-sm: 8px    /* small buttons, inner elements */
--radius-lg: 16px   /* page header, hero card */
```

---

## Shadow system

Three-level depth scale; all shadows tint with navy so they harmonize with
the color palette rather than reading as generic gray:

```css
--shadow-sm: 0 1px 3px rgba(20,54,93,0.07), 0 1px 2px rgba(0,0,0,0.04)
--shadow-md: 0 4px 12px rgba(20,54,93,0.11), 0 1px 3px rgba(0,0,0,0.05)
--shadow-lg: 0 8px 24px rgba(20,54,93,0.15), 0 2px 6px rgba(0,0,0,0.06)
```

---

## Typography

**Font:** Inter (400–900 weight range), loaded via Google Fonts CDN with
system-font fallback stack. Applied globally via `font-family: 'Inter', ...
!important` on all Streamlit elements.

| Element | Size | Weight | Notes |
|---|---|---|---|
| Page header title | 3.8rem | 900 | letter-spacing −0.05em |
| Hero nowcast value | 5.4rem | 900 | letter-spacing −0.055em, tabular-nums |
| KPI card value | 2.25rem | 900 | letter-spacing −0.03em |
| Section header | 1.18rem | 800 | letter-spacing −0.02em |
| Body / section desc | 0.90rem | 400 | line-height 1.68 |
| KPI label (caps) | 0.65rem | 700 | uppercase, letter-spacing 0.11em |
| Caption / footnote | 0.77rem | 400 | via `.stCaption p` |

---

## Components

### Page header (`.page-header`)

Dark navy gradient card at the very top of the page. Contains:
- **Eyebrow** (`.page-header-meta`): tiny all-caps, 0.67rem, muted white
- **Title** (`.page-header-title`): 3.8rem/900, white; `<span class="accent">` applies amber to "Nowcast"
- **Description** (`.page-header-sub`): 0.93rem, rgba(white, 0.58)
- **Tags** (`.header-tag`): pill-shaped, glass-morphism style; `.live` variant is green-tinted

Decorative elements: amber radial glow (top-right `::before`), subtle orb (bottom-left `::after`).

### KPI stat card (`.kpi-card`)

White card with a 3px colored top band. Usage:
```python
th.stat_card("LABEL", "VALUE", "subtext", accent=th.NAVY)
```
The `accent` color drives `--kpi-accent`, which colors both the top band (`::before`)
and the value text. Hover: `translateY(-2px)` + shadow upgrade.

### Hero nowcast card (`.hero-card`)

Dark navy gradient card for the Live Nowcast focal point. Value always renders
white (the dark background provides the contrast — the `color` param in
`hero_number()` is no longer applied inline). Amber decorative glow in the
top-right corner via `::before`.

### Navigation (`.stTabs`)

Restyled as a pill / segmented control. The tab list has a gray pill container
(`background: #EEF2F7`, `border-radius: 100px`); the active tab gets a white
background with a navy shadow; the animated indicator bar is hidden. This
replaces the default Streamlit underline-tab pattern entirely.

### Section header (`.section-header`)

`display: flex` with a navy 4px vertical bar from `::before`. Makes section
boundaries clearly visible without using extra whitespace.

### Chart cards

All Plotly chart containers (`[data-testid="stPlotlyChart"]`) receive:
`border: 1px solid var(--card-border)`, `border-radius: var(--radius)`,
`box-shadow: var(--shadow-sm)`. Charts look embedded in data panels, not
floating on the page.

---

## What NOT to change

- **NAVY/AMBER/SLATE/BRICK/INK hex values** — these are brand colors. A line
  in one chart must match the same series in every other chart.
- **Chart figure builders** — all six builders in `streamlit_app.py` use the
  Python color constants from `theme.py`. Don't duplicate or fork those values.
- **`PLOTLY_BASE_CONFIG` / `PLOTLY_EXPLORE_CONFIG`** — Phase 5.5 decisions about
  modebar buttons and Scattergl stay in place.
