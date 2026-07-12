"""Fetch REAL MLB sportsbook moneylines (with vig) for the definitive CLV gate.

Unlike the 538 benchmark (a fair, no-vig proxy), this pulls actual market
moneylines keyed by StatsAPI gamePk, so no date/team matching is needed and the
de-vig + betting sim reflect real book prices. Source: the community dataset
cesar-dx/mlb-betting-ml (per-season data/{year}/final_format.csv), which carries
game_pk, game_date, home_odds, away_odds (American).

NOTE: these are real market lines but may be opening/consensus rather than strictly
CLOSING. They are still a far sharper benchmark than 538 — good enough to establish
whether the model beats the market (it does not, as of this writing).

Output CSV columns: game_id, date, home_team, away_team, home_ml, away_ml.
Because it has a game_id column, `clv.load_odds_csv(path)` keys it directly by
gamePk; run it through run_clv_backtest against the leak-free replay records.

Usage:
  python scripts/fetch_real_odds.py --season 2023 --out mlb_real_odds_2023.csv
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import os
import sys

SRC = "https://raw.githubusercontent.com/cesar-dx/mlb-betting-ml/main/data/{season}/final_format.csv"


async def build(season: int, out: str) -> None:
    from common.data.http import make_async_client

    url = SRC.format(season=season)
    async with make_async_client() as c:
        r = await c.get(url, timeout=90, follow_redirects=True)
        r.raise_for_status()

    n = 0
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["game_id", "date", "home_team", "away_team", "home_ml", "away_ml"])
        for row in csv.DictReader(io.StringIO(r.text)):
            gp = (row.get("game_pk") or "").strip()
            h, a = row.get("home_odds"), row.get("away_odds")
            if not (gp and h and a):
                continue
            try:
                float(h); float(a)
            except (TypeError, ValueError):
                continue
            w.writerow([gp, row.get("game_date", ""), row.get("home_name", ""),
                        row.get("away_name", ""), h, a])
            n += 1

    print(f"wrote {n} {season} games with real moneylines -> {out}")
    if not n:
        sys.exit(f"no rows for {season} — check the upstream dataset has that season")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch real MLB sportsbook odds (gamePk-keyed) for the CLV gate.")
    ap.add_argument("--season", type=int, default=2023)
    ap.add_argument("--out", default="mlb_real_odds.csv")
    args = ap.parse_args()
    asyncio.run(build(args.season, args.out))


if __name__ == "__main__":
    main()
