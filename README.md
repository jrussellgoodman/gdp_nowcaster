# GDP Nowcasting Model

A mixed-frequency dynamic factor model (DFM) with a Kalman filter and EM
estimation for real-time US GDP nowcasting, modeled on the New York Fed Staff
Nowcast methodology (Bok, Caratelli, Giannone, Sbordone & Tambalotti, 2018).

> This README is a placeholder. It will be rewritten as a paper-style writeup
> in the final phase of the project (see `docs/progress.md`).

## Status
See [`docs/progress.md`](docs/progress.md) for current build progress.

## Setup
1. Create and activate a virtual environment:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and add your free FRED API key
   (get one at https://fred.stlouisfed.org).

## Project Structure
```
src/
  data/      - FRED data fetching and preprocessing
  model/     - dynamic factor model (DynamicFactorMQ) and baselines
  backtest/  - real-time vintage backtesting
app/         - Streamlit dashboard
tests/       - pytest tests and validation checks
docs/        - progress tracking and notes
notebooks/   - exploratory notebooks
```

## Run
- App: `streamlit run app/streamlit_app.py`
- Tests: `pytest -q`
