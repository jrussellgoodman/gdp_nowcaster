# Progress Checklist — GDP Nowcasting Model

Update this file as steps are completed. Claude Code should check it at the
start of a session to know where we left off, and update it after finishing
a step.

## Phase 0 — Setup
- [x] Repo structure scaffolded (src/data, src/model, src/backtest, tests,
      docs, app, notebooks)
- [x] requirements.txt created and installed into .venv
- [x] .env created (FRED_API_KEY) and confirmed gitignored
- [x] Git repo initialized, first commit made
- [ ] Pushed to GitHub (private repo)

## Phase 1 — Data Pipeline
- [x] src/data/fred_loader.py written (downloads + caches FRED series)
- [x] Stationarity transformations applied and explained
- [x] Mixed-frequency alignment (monthly panel + quarterly GDP) working
- [x] pytest for data shapes/frequencies/sanity passes
- [x] Spot-checked values against fred.stlouisfed.org

## Phase 2 — Baseline Model
- [x] AR(1) baseline model on GDPC1 growth
- [x] Single-factor DynamicFactorMQ model fits and converges (EM)
- [x] Produces a current-quarter GDP nowcast number
- [ ] Sanity-checked against latest GDPNOW value

## Phase 2 Results (2026-06-10)
- AR(1): β = −0.13 (mild mean-reversion), long-run mean = 2.44%/yr, R² = 1.8%
- DFM Q2 2026 nowcast: −0.33% annualized (vs AR(1) +2.55%)
- Compare at: https://www.atlantafed.org/cprcd/macroeconomic-research/gdpnow

## Phase 3 Results (2026-06-10)
- Blocked DFM: global factor (all 12 series) + real factor (9 real-activity series)
- LLF: −3510 vs 1-factor −3682 (2-factor global −3566); +172 pp improvement
- Q2 2026 nowcast (smoother, blocked model): −0.19% annualized
- News decomp (Apr 30 → Jun 10 vintage): revision −0.034 pp ann; dominated by
  weak Philly Fed May survey (−0.029 impact) offset by strong NY Fed (+0.019)
- Factor loadings: 10/11 signs correct; DSPIC96 negative loading explained by
  countercyclical stimulus payments (2020 CARES Act spike during recession)
- Factor plot saved at docs/figures/factor_sanity_check.png
- Validation figure saved at docs/figures/phase3_validation.png
- Chad Fulton comparison: global factor signs match exactly; CPIAUCSL uses first
  log-diff vs Fulton's second log-diff (minor, both stationary); COVID direction ✓;
  factors independent (corr = 0.107)

## Phase 3 — Full DFM + News Decomposition
- [x] Multi-factor / blocked DFM implemented (2-factor: global + real-activity)
- [x] EM convergence confirmed — blocked model LLF −3510 vs 1-factor −3682 (+172 pts)
- [x] News decomposition implemented via `compute_news()` + `print_news_summary()`
- [x] pytest confirms news impacts sum to total nowcast revision (tol 1e-5)
- [x] Validated against Chad Fulton's reference notebook output (see Phase 3 results)

## Phase 4 — Real-Time Vintage Backtest
- [x] Look-ahead bias concept explained and understood
- [x] ALFRED vintage-based backtest implemented (2015Q1–present)
- [x] Look-ahead assertions in place and passing (assert_no_lookahead in vintage.py)
- [x] RMSE/MAE with and without 2020 — both reported (design decision: not cherry-picking)
- [x] Compared against AR(1) baseline (dfm_skill_ex2020 in compute_rmse_stats)
- [x] Full 1-factor backtest run — 44/44 quarters — results in data/backtest_results.csv
- [x] Full 2-factor blocked backtest run — 44/44 quarters — data/backtest_results_2factor.csv
- [x] Plots saved to docs/figures/phase4_backtest.png and phase4_backtest_2factor.png
- [ ] GDPNow comparison added to README (defer to Phase 6)

## Phase 4 Results (2026-06-11)

### Setup
- Sample: Q1 2015 – Q4 2025, 44 quarters, ALFRED end-of-quarter evaluation
- Actual GDP: current-vintage FRED GDPC1 (latest revision) — see bias note below
- Backtest data: data/backtest_results.csv (1-factor), data/backtest_results_2factor.csv (2-factor)
- Plots: docs/figures/phase4_backtest.png, docs/figures/phase4_backtest_2factor.png

### Primary metrics (annualized pp)

| Metric                | 1-factor | 2-factor blocked | AR(1) |
|-----------------------|----------|-----------------|-------|
| RMSE — full sample    | 8.18     | 7.71            | 11.10 |
| RMSE — ex-2020 ★      | 2.92     | 2.87            | 1.95  |
| RMSE — 2020 only      | 25.49    | 23.89           | —     |
| Skill vs AR(1) ex-20  | −0.98 pp | −0.93 pp        | —     |
| Mean error (bias)     | −2.02 pp | −2.09 pp        | —     |

★ Primary metric for README and resume bullets. Use 2-factor (2.87 pp) as headline.

### Key findings
i d
1. **2-factor is marginally better on every RMSE measure** (0.05 pp ex-2020, 1.60 pp on 2020 only).
   The improvement from +172 LLF points (Phase 3) → only 0.05 pp RMSE gain is typical:
   better in-sample fit ≠ better out-of-sample forecast accuracy.
   Despite losing 24/40 quarter-by-quarter comparisons to 1-factor, the 2-factor model wins
   on RMSE because it reduces the large errors more efficiently (RMSE penalises big errors harder).

2. **AR(1) wins ex-2020 by ~1 pp.** Both models lose to the naive "predict the mean" baseline.
   This is expected for 11-series DFMs; the NY Fed runs 127 series. Not a failure — it documents
   where the model stands and motivates future expansion. Report honestly in README.

3. **Systematic −2 pp bias.** Almost every quarter the model under-predicts GDP growth.
   Likely cause: ALFRED data (pre-revision) used for nowcast vs current-vintage actual (post-2023
   BEA comprehensive revision, which lifted historical GDP ~0.3–0.5 pp/qtr). Both DFM and AR(1)
   face the same mismatch; the AR(1) is less affected because its prediction is near the historical
   mean regardless. Flagged in code (dfm.py compute_news look-ahead note), document in README.

4. **2020 COVID RMSE = 25 pp.** Q3 2020 is the single worst quarter (+44 pp error 1f, +43 pp 2f):
   model predicted continued crash, actual rebounded +30% as economy reopened. Not fixable with
   more factors; a fundamental limit of business-cycle DFMs for supply/pandemic shocks.

5. **Use 2-factor blocked model for dashboard.** Marginally better on all metrics, more
   theoretically grounded, consistent with Phase 3 design.

### Results table (formatted for README / dashboard)

| Metric | 1-factor | 2-factor | AR(1) | Δ (2f vs 1f) |
|---|---|---|---|---|
| **RMSE — ex-2020** ★ | 2.92 pp | **2.87 pp** | **1.95 pp** | −0.05 pp |
| RMSE — full sample | 8.18 pp | **7.71 pp** | 11.10 pp | −0.47 pp |
| RMSE — 2020 only | 25.49 pp | **23.89 pp** | — | −1.60 pp |
| Mean error (bias) | −2.02 pp | −2.09 pp | — | −0.07 pp (worse) |
| Skill vs AR(1) ex-20 | −0.98 pp | **−0.93 pp** | 0 (baseline) | +0.05 pp |

★ Primary metric for README and resume bullets: **2.87 pp ex-2020 RMSE (2-factor blocked DFM)**.

### Three things to notice

1. **The 2-factor improvement is marginal ex-2020 (0.05 pp).** The +172 log-likelihood-point
   in-sample gain (Phase 3) translated into almost no out-of-sample accuracy gain. This is
   completely normal — better in-sample fit rarely equals better forecasts.

2. **The 2-factor wins more clearly on full-sample RMSE (−0.47 pp) and crisis RMSE (−1.60 pp).**
   Its real-activity factor gives a sharper signal during the COVID shock. Even so, 23.89 pp
   crisis RMSE is still catastrophic — a fundamental DFM limitation for supply/pandemic shocks.

3. **The −2 pp bias is unchanged and slightly worse in the 2-factor.** Root cause: ALFRED
   pre-revision data used for nowcast vs current-vintage actual (post-2023 BEA comprehensive
   revision, which lifted historical GDP ~0.3–0.5 pp/qtr). More factors will not fix this.
   Report honestly in README: "both models under-predict by ~2 pp due to data vintage mismatch."

### Data quirks documented
- GACDFSA066MSFRBPHI absent from ALFRED before Q2 2015 → dropped for Q1 2015 vintage
- PCEC96 has no monthly data before January 2007 (BEA limitation); Kalman filter handles NaN
- ALFRED rate-limit handling: retry with 65s/90s backoff; 0.6s throttle between API calls

## Phase 5 — Dashboard (visual polish pass, 2026-06-11)
- [x] Theme foundation: `.streamlit/config.toml` + `app/theme.py` — navy/slate/amber
      palette, one hex per concept (DFM = navy #14365D, GDPNow = amber #E8A33D,
      AR(1) = slate, actual GDP = near-black, recessions = gray bands) used in every chart
- [x] All charts converted matplotlib → Plotly (interactive, full-width, 550–600 px);
      `plotly` added to requirements.txt
- [x] Landing area: plain-English intro + 4 headline stat cards (current nowcast,
      ex-2020 RMSE, indicator count, quarters backtested)
- [x] Live Nowcast: hero number + 10-year DFM-vs-GDPNow track-record chart
- [x] GDPNow comparison data: `scripts/fetch_gdpnow.py` pulls GDPNOW from ALFRED at
      each backtest quarter's evaluation date → `data/gdpnow_eoq.csv` (39/44 quarters;
      ALFRED's first GDPNow vintage is 2016-05-17, so 2015Q1–2016Q1 are NaN).
      `tests/test_gdpnow.py` (5 tests) incl. benchmark spot-checks vs published
      Atlanta Fed values (Q2 2020 ≈ −39.5, Q3 2020 ≈ +32)
- [x] News tab: default Apr 30 → Jun 10 example precomputed by
      `scripts/precompute_news_example.py` → `data/news_example.csv` (+meta JSON),
      loads instantly; diverging impact bar chart is the hero element;
      custom vintage windows still compute live (~60 s, session-cached)
- [x] Backtest tab: styled model-comparison table (2f / 1f / AR(1)), Q3 2020 worst-miss
      annotation, scatter shows 2020 quarters flagged (hollow brick markers) instead of
      hidden — per the "report both, don't cherry-pick" decision
- [x] Verified via Streamlit AppTest (no exceptions) + headless-Chrome screenshots of
      all four tabs; zero browser console errors

### Phase 5 fixes worth remembering
- **Latent news-chart bug found and fixed:** statsmodels names the impacts index level
  `"updated variable"` (space); the old app checked for `"updated_variable"` and
  silently fell back to a raw dataframe — the per-series bar chart had never rendered.
- Per-series news impacts sum to the **"impact of news"** component, NOT the full
  revision (which also includes revisions to previously-published observations,
  ~1.6e-5 quarterly for the Apr 30 window). `precompute_news_example.py` asserts
  against the correct quantity, mirroring tests/test_news.py.
- statsmodels factors come back on a monthly PeriodIndex — Plotly cannot serialize
  Period; convert with `.to_timestamp()` before plotting.
- GDPNow comparison caveat for README: GDPNow targets the advance estimate, our
  backtest actuals are current-vintage — GDPNow's measured accuracy is flattered.

## Phase 6.5 — Chart-scale fix + visual restraint pass (2026-07-01)
- [x] **Focused/Full scale toggles** on all time-series charts (`th.focused_yaxis`).
      Y-axes now default to the data's normal range computed excluding 2020;
      COVID extremes are printed on-chart (e.g. "Apr 2020 beyond this scale:
      global factor +32.5") with a one-click Full-range button. Rationale: the
      FT/Economist convention for post-COVID macro charts — autoscaling to a
      20-sigma observation flattens 95% of history into an unreadable stripe.
- [x] Backtest scatter defaults to the 40 normal quarters, "Include 2020" widens.
- [x] Visual restraint pass for academic audience: emoji removed from tabs,
      header 3.8rem→2.7rem + single subtle orb, hero 5.4→4.1rem, KPI 2.25→1.85rem,
      content column capped at 1380px, chart heights 600→440-480, inner chart
      padding. Old arrow-annotations (COVID collapse / worst miss) replaced by
      the on-chart scale notes.
- [x] Robustness: prepare_dfm_data now drops all-NaN series (failed FRED fetch)
      instead of letting DFM standardize produce inf — same guard vintage.py had.
- [x] scripts/screenshot_tabs.py added (playwright full-page captures per tab);
      docs/figures/after_redesign_tab1–4.png regenerated with the new design.

## Notes / Decisions Log
(Add dated notes here as decisions are made, e.g. "2026-06-15: reduced to 3
factors due to slow EM convergence with 5.")
