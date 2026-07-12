"""Live venue-edge probe: our model vs LIVE Kalshi MLB game prices.

The definitive CLV gate showed the model does not beat sharp VEGAS lines. But the
platform trades on Kalshi/Polymarket, which may be SOFTER. This probe takes a first
live read: for each near-term MLB game it pulls Kalshi's two per-team winner markets,
de-vigs them into a fair market probability + spread, fetches our model's win prob
from the Sports API, and reports the model-vs-Kalshi edge and the spread (cost).

It does NOT place bets — it quantifies (a) how much the model disagrees with Kalshi
and (b) how wide/thin Kalshi's MLB markets are. The definitive answer (do we beat the
venue's CLOSE?) comes from the paper auto-trader's CLV-by-venue ledger over settled
games; this is the fast snapshot that tells you whether that's worth running.

Usage:  python scripts/venue_edge_probe.py [--api http://127.0.0.1:8100] [--min-edge 0.05]
"""
from __future__ import annotations

import argparse
import asyncio
import itertools
import re

from common.data.http import make_async_client
from sports.baseball.analytics.clv import devig_pair

KALSHI = "https://api.elections.kalshi.com/trade-api/v2/markets?series_ticker=KXMLBGAME&status=open&limit=200"


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _yes_mid_and_spread(m: dict):
    bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2.0, round((ask - bid) * 100)
    last = _f(m.get("last_price_dollars"))
    return (last, None) if last is not None else (None, None)


def _teams_from_title(title: str):
    t = re.sub(r"\s*winner\??\s*$", "", (title or ""), flags=re.I)
    parts = re.split(r"\s+vs\.?\s+", t, flags=re.I)
    return (parts[0].strip(), parts[1].strip()) if len(parts) == 2 else (None, None)


def _squash(s: str) -> str:
    return re.sub(r"[^A-Z]", "", (s or "").upper())


def _subseq_score(needle: str, hay: str) -> int:
    """How many letters of `needle` appear IN ORDER within `hay` (subsequence)."""
    it = iter(hay)
    return sum(1 for ch in needle if any(h == ch for h in it))


async def kalshi_games(c):
    """Per MLB game event: {'a','b' title teams, 'by_team': {title_team: (mid, spread)}}.
    The two side-market suffixes are assigned to the two title teams by the permutation
    that maximizes total subsequence-match score (robust for SF/TB/KC-style codes)."""
    r = await c.get(KALSHI, timeout=25, follow_redirects=True)
    r.raise_for_status()
    raw: dict[str, dict] = {}
    for m in r.json().get("markets", []):
        base, _, side = m.get("ticker", "").rpartition("-")
        if not side:
            continue
        a, b = _teams_from_title(m.get("title", ""))
        mid, spread = _yes_mid_and_spread(m)
        if mid is None or not (a and b):
            continue
        raw.setdefault(base, {"a": a, "b": b, "sides": []})["sides"].append((side, mid, spread))

    out = {}
    for base, ev in raw.items():
        if len(ev["sides"]) != 2:
            continue
        a, b = ev["a"], ev["b"]
        sa, sb = _squash(a), _squash(b)
        (s0, m0, sp0), (s1, m1, sp1) = ev["sides"]
        # score both permutations of (side -> team) and keep the better assignment
        p1 = _subseq_score(_squash(s0), sa) + _subseq_score(_squash(s1), sb)
        p2 = _subseq_score(_squash(s0), sb) + _subseq_score(_squash(s1), sa)
        if p1 >= p2:
            by_team = {a: (m0, sp0), b: (m1, sp1)}
        else:
            by_team = {a: (m1, sp1), b: (m0, sp0)}
        out[base] = {"a": a, "b": b, "by_team": by_team}
    return out


async def model_games(c, api):
    r = await c.get(f"{api}/api/baseball/schedule", timeout=30, follow_redirects=True)
    r.raise_for_status()
    out = []
    for g in r.json().get("games", []):
        pred = g.get("prediction") or {}
        home, away, hp = g.get("home_team"), g.get("away_team"), pred.get("home_win_prob")
        if home and away and hp is not None:
            out.append({"home": home, "away": away, "home_prob": float(hp),
                        "model": pred.get("model_version")})
    return out


def _match_event(kal, home, away):
    """Find the Kalshi event whose two title teams are substrings of home/away."""
    hu, au = home.upper(), away.upper()
    for ev in kal.values():
        a, b = ev["a"].upper(), ev["b"].upper()
        if (a in hu and b in au) or (a in au and b in hu):
            home_title = ev["a"] if ev["a"].upper() in hu else ev["b"]
            away_title = ev["b"] if home_title == ev["a"] else ev["a"]
            return ev, home_title, away_title
    return None, None, None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8100")
    ap.add_argument("--min-edge", type=float, default=0.05)
    args = ap.parse_args()

    async with make_async_client() as c:
        kal = await kalshi_games(c)
        try:
            mdl = await model_games(c, args.api)
        except Exception as e:
            print(f"(!) Sports API {args.api} unreachable ({type(e).__name__}) — Kalshi only.")
            mdl = []

    print(f"\nKalshi MLB game events: {len(kal)} | model games: {len(mdl)}\n")
    print(f"{'Matchup':<30}{'Model':>7}{'Kalshi':>8}{'Edge':>7}{'Sprd':>6}  Flag")
    print("-" * 66)
    edges, spreads, hits, matched = [], [], 0, 0
    seen = set()
    for ev in kal.values():
        # nearest model game for this event (first schedule game with both teams)
        mg = next((g for g in mdl
                   if (ev["a"].upper() in g["home"].upper() and ev["b"].upper() in g["away"].upper())
                   or (ev["a"].upper() in g["away"].upper() and ev["b"].upper() in g["home"].upper())), None)
        if not mg:
            continue
        key = (mg["home"], mg["away"])
        if key in seen:
            continue
        seen.add(key)
        home_title = ev["a"] if ev["a"].upper() in mg["home"].upper() else ev["b"]
        away_title = ev["b"] if home_title == ev["a"] else ev["a"]
        home_mid, sp = ev["by_team"][home_title]
        away_mid, _ = ev["by_team"][away_title]
        fair_home, _fa = devig_pair(home_mid, away_mid)
        edge = mg["home_prob"] - fair_home
        matched += 1
        edges.append(abs(edge))
        if sp is not None:
            spreads.append(sp)
        flag = "VALUE" if abs(edge) >= args.min_edge else ""
        hits += 1 if flag else 0
        side = mg["home"] if edge > 0 else mg["away"]
        row = f"{(mg['away']+' @ '+mg['home'])[:29]:<30}{mg['home_prob']*100:>6.1f}%{fair_home*100:>7.1f}%{edge*100:>+6.1f}"
        row += f"{('' if sp is None else str(sp)+'c'):>6}  {flag}{(' -> '+side) if flag else ''}"
        print(row)

    if matched:
        avg_sp = (sum(spreads) / len(spreads)) if spreads else float("nan")
        print("-" * 66)
        print(f"matched {matched} games | mean |edge| {sum(edges)/len(edges)*100:.1f}% | "
              f"mean spread {avg_sp:.1f}c | >= {args.min_edge*100:.0f}% edge: {hits}")
        print("\nRead: a large model-vs-Kalshi edge is EITHER venue softness OR model error "
              "(the model loses to Vegas). Wide spreads are the round-trip cost. The real\n"
              "answer is CLV over settled games — run the paper auto-trader, read CLV-by-venue.")
    else:
        print("(no model game matched a live Kalshi event — off-hours, or API stale)")


if __name__ == "__main__":
    asyncio.run(main())
