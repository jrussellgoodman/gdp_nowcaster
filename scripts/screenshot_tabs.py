"""
screenshot_tabs.py — capture full-page screenshots of all four dashboard tabs.

Used during design passes to inspect the rendered app at full resolution.
Usage:  python scripts/screenshot_tabs.py <output_prefix>
Writes: docs/figures/<prefix>_tab1.png … <prefix>_tab4.png

Assumes the app is already running on localhost:8599
(streamlit run app/streamlit_app.py --server.port 8599).
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "snapshot"
OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TAB_LABELS = ["Live Nowcast", "The Factors", "News", "Backtest"]

with sync_playwright() as p:
    browser = p.chromium.launch(channel="chrome", headless=True)
    page = browser.new_page(viewport={"width": 1600, "height": 1000})
    page.goto("http://localhost:8599", timeout=60_000)

    # Wait for the model fit to finish: the hero/stat cards appear only after.
    # Poll for the kpi-value elements to contain a % (i.e., a real nowcast).
    deadline = time.time() + 180
    while time.time() < deadline:
        page.wait_for_timeout(3000)
        body = page.inner_text("body")
        if "Model failed to load" in body:
            print("ERROR: model failed to load:", flush=True)
            for line in body.splitlines():
                if "failed" in line.lower():
                    print("  ", line)
            break
        # Model done when the spinner text is gone and a nowcast % rendered
        if "fitting the 2-factor DFM" not in body and "%" in body:
            break

    for i, label in enumerate(TAB_LABELS, start=1):
        try:
            page.click(f"button[role='tab']:has-text('{label}')", timeout=10_000)
        except Exception:
            # tab widget markup fallback
            page.click(f"[data-baseweb='tab']:has-text('{label}')", timeout=10_000)
        page.wait_for_timeout(2500)
        path = OUT / f"{PREFIX}_tab{i}.png"
        page.screenshot(path=str(path), full_page=True)
        print(f"saved {path}")

    browser.close()
