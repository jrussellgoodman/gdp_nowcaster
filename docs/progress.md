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

## Phase 3 — Full DFM + News Decomposition
- [ ] Multi-factor / blocked DFM implemented
- [ ] EM convergence confirmed (or factors reduced if unstable)
- [ ] News decomposition implemented via `news` method
- [ ] pytest confirms news impacts sum to total nowcast revision
- [ ] Validated against Chad Fulton's reference notebook output

## Phase 4 — Real-Time Vintage Backtest
- [ ] Look-ahead bias concept explained and understood
- [ ] ALFRED vintage-based backtest implemented (2015Q1–present)
- [ ] Look-ahead assertions in place and passing
- [ ] RMSE/MAE by horizon computed
- [ ] Compared against AR(1) baseline and GDPNOW

## Phase 5 — Deployment
- [ ] Streamlit app built (4 tabs: Live Nowcast, Factor, News, Backtest)
- [ ] Runs locally without errors
- [ ] Deployed to Streamlit Community Cloud
- [ ] Secrets (FRED_API_KEY) configured on Streamlit Cloud

## Phase 6 — Writeup
- [ ] README rewritten as a paper-style writeup with citation
- [ ] Resume bullet(s) drafted with real RMSE numbers
- [ ] Optional blog post drafted

## Notes / Decisions Log
(Add dated notes here as decisions are made, e.g. "2026-06-15: reduced to 3
factors due to slow EM convergence with 5.")
