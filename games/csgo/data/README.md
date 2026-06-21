# CS2 / CSGO data providers

The model, backtest, Model Health panel, and single-game replay all run on whatever
`CsgoDataClient` the factory selects. **By default that's the in-memory `stub`**, which
generates deterministic synthetic history so the service always boots — useful for
development, but the Brier / Brier-skill / ROI numbers it produces are *not real*. The
dashboard flags this with a **"synthetic stub data"** badge on the Backtest and Model
Health tabs; a real feed shows **"live feed: <provider>"** instead.

To get real numbers, point `CSGO_DATA_PROVIDER` at a real source:

| `CSGO_DATA_PROVIDER` | Needs | Cost | Notes |
|----------------------|-------|------|-------|
| `stub` (default)     | nothing | free | synthetic; for dev only |
| `localfile`          | `CSGO_HISTORY_FILE=/path/to/results.csv` (or `.json`) | free | a CSV/JSON results export (e.g. a Kaggle HLTV dump) |
| `liquipedia`         | `LIQUIPEDIA_API_KEY` (free, signup) + `LIQUIPEDIA_USER_AGENT` (contact) | free | rate-limited; be polite |
| `pandascore`         | `PANDASCORE_TOKEN` | paid | richest feed (rosters, games, odds) |

Any provider that's selected but missing its credential/file **falls back to the stub**
(and is reported as synthetic), so a misconfiguration never takes the service down.

## localfile CSV format

Header row required; one row per finished match. Minimum columns:

```csv
date,event,team1,team2,best_of,team1_score,team2_score
2026-01-10,IEM Katowice 2026,Natus Vincere,FaZe Clan,3,2,1
```

`date` is ISO (`YYYY-MM-DD`); `team1_score`/`team2_score` are maps won. A JSON export may
additionally carry per-map detail (`map_scores`) which enriches the per-map Glicko ratings
and the replay's map-by-map curve. See `sample_history.csv` for a starter file and
`tests/games/csgo/test_localfile_client.py` for the parser's accepted shapes.

You need ~20+ finished matches (per team pairing) before the walk-forward backtest scores
anything — the first matches are spent warming up leak-free ratings.

## Quick start (real data, free)

```bash
# 1. Drop a results export anywhere, e.g. ./cs2_results.csv
export CSGO_DATA_PROVIDER=localfile
export CSGO_HISTORY_FILE=$PWD/cs2_results.csv
# 2. Restart the Sports API. The Backtest + Model Health tabs now show real numbers
#    with a green "live feed: localfile" badge instead of the amber stub badge.
```

The active provider is also exposed at `GET /api/csgo/source`
(`{provider, synthetic, history_matches, teams}`).
