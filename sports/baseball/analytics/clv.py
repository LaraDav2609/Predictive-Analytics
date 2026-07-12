"""Model-vs-market backtest — the real trading-readiness gate for baseball.

Beating the home base rate (the leak-free backtest gate) is NOT the same as beating
the market. MLB moneylines are set by sharp books and are far tighter than a
base-rate baseline, so the honest question is: **do the model's probabilities beat
the closing line?** This module answers it, given historical closing moneyline odds:

1. De-vig the closing line into the market's fair probability (proportional / Shin).
2. Compare model vs market on Brier / log-loss over the same games — does the model
   predict outcomes better than the market's own fair number?
3. Simulate betting the model's edge picks AT the closing line (flat + Kelly): ROI,
   win rate, P&L, average edge. Positive ROI at the close ≈ genuine alpha.

Gate: ``model_beats_market`` (lower Brier) AND ``roi > 0``. Pure-python; no odds
source is bundled — supply a season's closing lines (see ``load_odds_csv``).
"""
from __future__ import annotations

import csv
import math
import os
from typing import Iterable, Optional


# ─────────────────────────── odds math ───────────────────────────
def american_to_prob(ml: float) -> float:
    """American moneyline → implied probability (vig included)."""
    ml = float(ml)
    return (-ml) / (-ml + 100.0) if ml < 0 else 100.0 / (ml + 100.0)


def american_to_decimal(ml: float) -> float:
    """American moneyline → decimal odds (payout multiple incl. stake)."""
    ml = float(ml)
    return 1.0 + (100.0 / -ml if ml < 0 else ml / 100.0)


def devig_pair(home_imp: float, away_imp: float, method: str = "proportional") -> tuple[float, float]:
    """Remove the bookmaker's vig from a two-way market → fair probabilities that
    sum to 1. 'proportional' (normalize) is the standard default; 'shin' backs out an
    insider-trading factor and is a touch fairer on lopsided lines."""
    home_imp = max(1e-9, float(home_imp)); away_imp = max(1e-9, float(away_imp))
    if method == "shin":
        s = home_imp + away_imp
        z_num = (s - 1.0)
        # Shin's z (fraction of informed money); closed form for two outcomes.
        disc = z_num * z_num + 4.0 * (1.0 - z_num) * home_imp * away_imp / s
        z = max(0.0, min(0.5, (z_num) / (s) if s > 0 else 0.0))
        def shin(p):
            return (math.sqrt(z * z + 4.0 * (1.0 - z) * p * p / s) - z) / (2.0 * (1.0 - z)) if z < 1 else p
        h, a = shin(home_imp), shin(away_imp)
        tot = h + a
        return (h / tot, a / tot) if tot > 0 else (0.5, 0.5)
    tot = home_imp + away_imp
    return home_imp / tot, away_imp / tot


def market_fair_home(row: dict, method: str = "proportional") -> Optional[float]:
    """Fair (de-vigged) home win probability from an odds row. Accepts American
    (``home_ml``/``away_ml``), decimal (``home_dec``/``away_dec``) or pre-computed
    implied (``home_implied``/``away_implied``) columns."""
    def num(*keys):
        for k in keys:
            v = row.get(k)
            if v not in (None, ""):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return None

    hml, aml = num("home_ml", "home_moneyline"), num("away_ml", "away_moneyline")
    if hml is not None and aml is not None:
        return devig_pair(american_to_prob(hml), american_to_prob(aml), method)[0]
    hdec, adec = num("home_dec", "home_decimal"), num("away_dec", "away_decimal")
    if hdec and adec and hdec > 1 and adec > 1:
        return devig_pair(1.0 / hdec, 1.0 / adec, method)[0]
    himp, aimp = num("home_implied"), num("away_implied")
    if himp is not None and aimp is not None:
        return devig_pair(himp, aimp, method)[0]
    return None


def _decimal_for(row: dict, side: str) -> Optional[float]:
    v = row.get(f"{side}_ml") or row.get(f"{side}_moneyline")
    if v not in (None, ""):
        try:
            return american_to_decimal(float(v))
        except (TypeError, ValueError):
            pass
    d = row.get(f"{side}_dec") or row.get(f"{side}_decimal")
    try:
        return float(d) if d not in (None, "") and float(d) > 1 else None
    except (TypeError, ValueError):
        return None


def _kelly(p: float, dec: float) -> float:
    """Fraction of bankroll for prob ``p`` at decimal odds ``dec`` (0 if no edge)."""
    if dec <= 1:
        return 0.0
    f = (p * dec - 1.0) / (dec - 1.0)
    return max(0.0, f)


# ─────────────────────────── the backtest ───────────────────────────
def run_clv_backtest(records, odds_by_game: dict, *, edge_threshold_bps: float = 200.0,
                     kelly_fraction: float = 0.25, devig: str = "proportional",
                     flat_stake: float = 1.0) -> dict:
    """``records``: leak-free backtest records (need ``.game_id``, ``.prob_home``,
    ``.actual``). ``odds_by_game``: {game_id: odds_row}. Returns model-vs-market
    accuracy plus a flat + fractional-Kelly betting simulation at the closing line."""
    from common.ml.calibration import brier_score, log_loss
    import numpy as np

    thr = edge_threshold_bps / 10_000.0
    mp, kp, my, mm = [], [], [], []      # model probs, market probs, outcomes
    flat_pnl = flat_bets = flat_wins = 0.0
    kelly_bankroll = 1.0
    edges = []
    matched = 0

    for r in records:
        row = odds_by_game.get(r.game_id) or odds_by_game.get(str(r.game_id))
        if not row:
            continue
        fair_home = market_fair_home(row, devig)
        if fair_home is None:
            continue
        matched += 1
        p_home = float(r.prob_home); y = float(r.actual)
        mp.append(p_home); kp.append(fair_home); my.append(y)

        # Which side (if any) does the model think is mispriced by > threshold?
        for side, p_model, p_mkt, won in (
            ("home", p_home, fair_home, y == 1.0),
            ("away", 1.0 - p_home, 1.0 - fair_home, y == 0.0),
        ):
            edge = p_model - p_mkt
            if edge <= thr:
                continue
            dec = _decimal_for(row, side)
            if not dec:
                continue
            edges.append(edge)
            # Flat stake
            flat_bets += 1
            flat_pnl += flat_stake * (dec - 1.0) if won else -flat_stake
            if won:
                flat_wins += 1
            # Fractional Kelly on the running bankroll
            stake = kelly_bankroll * kelly_fraction * _kelly(p_model, dec)
            kelly_bankroll += stake * (dec - 1.0) if won else -stake

    n = len(my)
    if n == 0:
        return {"available": False, "reason": "no games matched to odds",
                "matched": 0, "records": len(list(records)) if hasattr(records, "__len__") else None}

    model = np.array(mp); market = np.array(kp); outcome = np.array(my)
    model_brier = brier_score(model, outcome); market_brier = brier_score(market, outcome)
    result = {
        "available": True,
        "games_scored": n,
        "matched_to_odds": matched,
        "devig": devig,
        "model": {"brier": round(model_brier, 4), "log_loss": round(log_loss(model, outcome), 4)},
        "market": {"brier": round(market_brier, 4), "log_loss": round(log_loss(market, outcome), 4)},
        "model_beats_market": bool(model_brier < market_brier),
        "brier_vs_market": round(market_brier - model_brier, 4),   # >0 → model sharper
        "betting": {
            "edge_threshold_bps": edge_threshold_bps,
            "bets": int(flat_bets),
            "flat_roi": round(flat_pnl / flat_bets, 4) if flat_bets else 0.0,
            "flat_pnl_units": round(flat_pnl, 2),
            "win_rate": round(flat_wins / flat_bets, 4) if flat_bets else 0.0,
            "avg_edge_bps": round(float(np.mean(edges)) * 10_000, 1) if edges else 0.0,
            "kelly_fraction": kelly_fraction,
            "kelly_final_bankroll": round(kelly_bankroll, 4),
            "kelly_return": round(kelly_bankroll - 1.0, 4),
        },
    }
    roi = result["betting"]["flat_roi"]
    result["gate"] = {
        "beats_market_brier": result["model_beats_market"],
        "positive_roi": bool(roi > 0),
        "tradeable": bool(result["model_beats_market"] and roi > 0 and flat_bets >= 20),
        "verdict": (
            "Model beats the closing line AND flat-stake edge picks are profitable — "
            "evidence of genuine edge (validate on more seasons before sizing up)."
            if result["model_beats_market"] and roi > 0 else
            "Model does NOT beat the closing line — the observed base-rate skill does "
            "not survive against sharp market prices. Not tradeable."
        ),
    }
    return result


# ─────────────────────────── odds loading ───────────────────────────
def load_odds_csv(path: str) -> dict:
    """Load closing odds keyed by game id from a CSV with a ``game_id`` column plus
    American (``home_ml``/``away_ml``), decimal or implied columns. Returns {} if the
    file is absent so callers can degrade to 'supply odds' messaging."""
    if not path or not os.path.exists(path):
        return {}
    out: dict = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            gid = (row.get("game_id") or row.get("gamePk") or row.get("id") or "").strip()
            if gid:
                out[gid] = row
                try:
                    out[int(gid)] = row
                except ValueError:
                    pass
    return out
