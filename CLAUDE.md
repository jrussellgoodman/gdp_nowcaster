# Project: GDP Nowcasting Model

## Goal
Build a mixed-frequency dynamic factor model (DFM) with a Kalman filter and
EM estimation for real-time US GDP nowcasting, modeled on the New York Fed
Staff Nowcast methodology (Bok, Caratelli, Giannone, Sbordone & Tambalotti,
2018, "Macroeconomic Nowcasting and Forecasting with Big Data"). Use
statsmodels' DynamicFactorMQ as the primary modeling tool — do not hand-roll
the Kalman filter or EM algorithm.

The end goal is a deployed Streamlit dashboard plus a paper-style README,
intended as a portfolio piece for quant / data science / Federal Reserve
internship applications.

## Stack
- Python 3.11+
- Core libraries: statsmodels, pandas, numpy, fredapi, streamlit, pytest,
  pyarrow, matplotlib
- Always work inside the project's virtual environment (.venv)

## How I Work
I am an economics + math student with very limited Python ability. You
(Claude) are doing essentially all of the coding. For everything you build:
1. Explain in plain English, in chat AND in docstrings, what each piece of
   code does and why.
2. After writing any numerical/statistical code, write a pytest test AND a
   sanity check that compares the output against a known benchmark
   (e.g., a published GDPNow value, a documented example from Chad Fulton's
   DynamicFactorMQ notebook, or an internal consistency check).
3. If something could plausibly be wrong but you're not sure, say so rather
   than presenting it as correct.

## Data Rules (critical)
- NEVER use future or revised data when computing a historical nowcast.
- Always respect FRED/ALFRED data vintages (realtime_start / realtime_end)
  when doing anything backtest-related.
- Proactively flag any look-ahead risk you notice, even if not asked.
- The FRED API key lives in `.env` as FRED_API_KEY — never hard-code it,
  never print it, never commit it.
- ISM/PMI series (e.g. NAPM*) are NOT available on FRED (removed in 2014 due
  to licensing) — do not use them. Use GACDISA066MSFRBNY (NY Fed) and
  GACDFSA066MSFRBPHI (Philly Fed) survey series as substitutes.
- GDPC1 is REAL GDP (use this). GDP alone is nominal — do not confuse them.
- PCEC96 (monthly real PCE) vs PCECC96 (quarterly) — use the monthly one for
  the monthly panel.

## Git Conventions
- Conventional commits: feat:, fix:, docs:, test:, refactor:, chore:
- Subject line under ~72 characters
- Run `pytest -q` before committing
- Never commit .env, the data/ cache folder, or .venv/

## Commands
- Run the app: `streamlit run app/streamlit_app.py`
- Run tests: `pytest -q`
- Activate venv: `source .venv/bin/activate`

## Project Status
See docs/progress.md for the current phase and checklist. Update it as we
complete steps.
