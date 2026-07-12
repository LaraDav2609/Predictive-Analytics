"""Build a CLV benchmark odds file from FiveThirtyEight's MLB Elo ratings.

The true CLV gate wants sportsbook CLOSING moneylines. Those are hard to get free
(538's own site is down; the SBR archive is 404). As an INDEPENDENT, reproducible
stand-in this script turns 538's *pitcher-adjusted* game ratings (`rating_prob1/2`,
which account for starting pitchers) into a fair two-way moneyline CSV that the
`/api/baseball/clv` endpoint (and `odds_ingest.load_and_match`) can consume.

IMPORTANT: 538 ratings are a strong independent benchmark, NOT the market close.
- If our model can't beat 538 -> it certainly won't beat the sharper closing line.
- If it does beat 538 -> necessary but NOT sufficient; still need real closing odds.
The generated line has NO vig (probs sum to 1), so any betting-sim ROI is optimistic
and should be ignored; read the Brier comparison, not the ROI.

Usage:
  python scripts/fetch_538_benchmark_odds.py --season 2023 --out mlb_538_2023.csv
Then:
  GET /api/baseball/clv?season=2023&odds=<abs path to mlb_538_2023.csv>
  (or scripts/refit_baseball_models.ps1 -Season 2023 -OddsCsv <path>)
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import os
import sys

# 538 team abbreviation -> MLB StatsAPI full name (matched to the schedule via normalize_team).
FULL = {
    "ARI": "Arizona Diamondbacks", "ATL": "Atlanta Braves", "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox", "CHC": "Chicago Cubs", "CHW": "Chicago White Sox",
    "CIN": "Cincinnati Reds", "CLE": "Cleveland Guardians", "COL": "Colorado Rockies",
    "DET": "Detroit Tigers", "HOU": "Houston Astros", "KCR": "Kansas City Royals",
    "LAA": "Los Angeles Angels", "LAD": "Los Angeles Dodgers", "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers", "MIN": "Minnesota Twins", "NYM": "New York Mets",
    "NYY": "New York Yankees", "OAK": "Oakland Athletics", "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates", "SDP": "San Diego Padres", "SEA": "Seattle Mariners",
    "SFG": "San Francisco Giants", "STL": "St. Louis Cardinals", "TBR": "Tampa Bay Rays",
    "TEX": "Texas Rangers", "TOR": "Toronto Blue Jays", "WSN": "Washington Nationals",
    # historical / alternate codes
    "ANA": "Los Angeles Angels", "TBD": "Tampa Bay Rays", "FLA": "Miami Marlins",
    "KCA": "Kansas City Royals", "SDG": "San Diego Padres", "SFO": "San Francisco Giants",
    "WAS": "Washington Nationals", "MON": "Washington Nationals",
}

# Mirror of 538's mlb_elo.csv (their own projects.fivethirtyeight.com endpoint is dead).
MIRROR = "https://raw.githubusercontent.com/sportstensor/MLB/main/data_and_models/mlb_elo.csv"


def prob_to_american(p: float) -> int:
    p = max(0.02, min(0.98, p))
    return -round(100 * p / (1 - p)) if p >= 0.5 else round(100 * (1 - p) / p)


async def build(season: int, out: str, url: str) -> None:
    from common.data.http import make_async_client

    async with make_async_client() as c:
        r = await c.get(url, timeout=90, follow_redirects=True)
        r.raise_for_status()

    rows = list(csv.DictReader(io.StringIO(r.text)))
    n, unmapped = 0, set()
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "home_team", "away_team", "home_ml", "away_ml"])
        for row in rows:
            if row.get("season") != str(season):
                continue
            if (row.get("playoff") or "").strip():          # regular season only
                continue
            t1, t2 = row.get("team1"), row.get("team2")       # team1 = home, team2 = away
            p1 = row.get("rating_prob1") or row.get("elo_prob1")
            p2 = row.get("rating_prob2") or row.get("elo_prob2")
            if not (t1 and t2 and p1 and p2):
                continue
            home, away = FULL.get(t1), FULL.get(t2)
            if not home:
                unmapped.add(t1)
            if not away:
                unmapped.add(t2)
            if not (home and away):
                continue
            w.writerow([row["date"], home, away, prob_to_american(float(p1)), prob_to_american(float(p2))])
            n += 1

    print(f"wrote {n} {season} regular-season rows -> {out}")
    if unmapped:
        print(f"unmapped 538 codes (skipped): {sorted(unmapped)}", file=sys.stderr)
    if not n:
        sys.exit(f"no rows for season {season} — is the mirror current?")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a 538 benchmark odds CSV for the CLV gate.")
    ap.add_argument("--season", type=int, default=2023)
    ap.add_argument("--out", default="mlb_538_benchmark.csv")
    ap.add_argument("--url", default=MIRROR, help="mlb_elo.csv source URL")
    args = ap.parse_args()
    asyncio.run(build(args.season, args.out, args.url))


if __name__ == "__main__":
    main()
