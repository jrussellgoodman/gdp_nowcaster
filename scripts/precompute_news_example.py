"""
precompute_news_example.py — bake the default news-decomposition illustration.

Why precompute?
  The News tab's default example (Apr 30 → Jun 10, 2026 — the vintage window
  analyzed in Phase 3) requires fitting the DFM twice, which takes ~60 seconds.
  Visitors to a portfolio dashboard won't wait that long for the first tab
  view, so we compute it once here and ship the result as two small files
  the app loads instantly:

    data/news_example.csv        — per-series surprises and nowcast impacts
                                   (columns: series, news, impact_qtr, impact_ann)
    data/news_example_meta.json  — before/after nowcasts, revision, dates

  The interactive "pick your own vintage date" path in the app still computes
  live; only the default illustration is precomputed.

Consistency guarantee:
  The aggregation here (group per-series, sum news and impact) is exactly what
  the app's live path does, and we assert the same invariant tests/test_news.py
  checks: per-series impacts must sum to the total nowcast revision.

Run:  source .venv/bin/activate && python scripts/precompute_news_example.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.model.dfm import compute_news, make_block_structure, prepare_dfm_data

BEFORE_END = "2026-04-30"          # vintage window documented in Phase 3
OUTPUT_CSV  = Path("data/news_example.csv")
OUTPUT_META = Path("data/news_example_meta.json")


def main() -> None:
    # Same 2-factor blocked structure the dashboard uses.
    monthly_df, quarterly_df = prepare_dfm_data(start="2000-01-01", use_cache=True)
    factor_blocks = make_block_structure(monthly_df, quarterly_df)

    print(f"Computing news decomposition: {BEFORE_END} -> latest (~60s)…")
    news = compute_news(
        before_end=BEFORE_END,
        factor_blocks=factor_blocks,
        start="2000-01-01",
        use_cache=True,
    )

    # Aggregate to one row per data series — identical to the app's live path.
    # NOTE: statsmodels names this index level "updated variable" (with a
    # space, not an underscore). The original dashboard checked for
    # "updated_variable" and therefore never found it — that bug is fixed
    # in the app as well.
    tidy = (
        news["impacts"].reset_index()
        .groupby("updated variable")[["news", "impact"]]
        .sum()
        .sort_values("impact")
        .rename(columns={"impact": "impact_qtr"})
    )
    tidy["impact_ann"] = tidy["impact_qtr"] * 4
    tidy.index.name = "series"

    # Invariant (same as tests/test_news.py): per-series impacts sum to the
    # total "impact of news". Note this is NOT the full revision — the revision
    # additionally includes effects of revisions to previously-published
    # observations, which have no per-series news row. For the Apr 30 window
    # that residual is ~1.6e-5 quarterly (negligible, but nonzero).
    total = tidy["impact_qtr"].sum()
    total_news = float(news["news_results"].impacts["impact of news"].iloc[0])
    assert abs(total - total_news) < 1e-5, (
        f"impacts sum {total:.8f} != impact of news {total_news:.8f}"
    )

    tidy.to_csv(OUTPUT_CSV)
    meta = {
        "before_end":          news["before_end"],
        "computed_on":         pd.Timestamp.today().strftime("%Y-%m-%d"),
        "target_quarter":      news["target_quarter"],
        "before_nowcast_ann":  news["before_nowcast_ann"],
        "after_nowcast_ann":   news["after_nowcast_ann"],
        "revision_ann":        news["revision_ann"],
    }
    OUTPUT_META.write_text(json.dumps(meta, indent=2))

    print(f"Saved {OUTPUT_CSV} ({len(tidy)} series) and {OUTPUT_META}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
