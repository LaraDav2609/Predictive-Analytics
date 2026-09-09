"""Leakage-safe, ML-ready basketball game feature model.

The coefficients are an interpretable v1 logistic score.  The feature contract is
deliberately stable so it can later be fed to a trained/calibrated model without
changing API consumers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import exp, log
from typing import Any


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_date(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def flatten_team_stats(payload: dict[str, Any]) -> dict[str, float]:
    result: dict[str, float] = {}
    categories = (((payload.get("results") or {}).get("stats") or {}).get("categories") or [])
    for category in categories:
        for stat in category.get("stats") or []:
            name = stat.get("name")
            if name:
                result[name] = _number(stat.get("value", stat.get("displayValue")))
    return result


def recent_form(schedule: dict[str, Any], team_id: str, before: datetime, limit: int = 10) -> dict[str, Any]:
    completed: list[dict[str, Any]] = []
    for event in schedule.get("events") or []:
        event_date = _parse_date(event.get("date"))
        competition = (event.get("competitions") or [{}])[0]
        if not event_date or event_date >= before or not ((competition.get("status") or {}).get("type") or {}).get("completed"):
            continue
        competitors = competition.get("competitors") or []
        own = next((x for x in competitors if str((x.get("team") or {}).get("id")) == str(team_id)), None)
        other = next((x for x in competitors if x is not own), None)
        if not own or not other:
            continue
        own_score, other_score = _number((own.get("score") or {}).get("value", own.get("score"))), _number((other.get("score") or {}).get("value", other.get("score")))
        completed.append({"date": event_date, "win": 1.0 if own_score > other_score else 0.0,
                          "points": own_score, "allowed": other_score, "margin": own_score - other_score})

    completed.sort(key=lambda row: row["date"], reverse=True)
    rows = completed[:limit]
    count = len(rows)
    if not count:
        return {"games": 0, "win_pct": 0.5, "avg_margin": 0.0, "points_for": 0.0,
                "points_against": 0.0, "rest_days": None}
    rest = max(0.0, (before - rows[0]["date"]).total_seconds() / 86400.0)
    return {
        "games": count,
        "win_pct": sum(x["win"] for x in rows) / count,
        "avg_margin": sum(x["margin"] for x in rows) / count,
        "points_for": sum(x["points"] for x in rows) / count,
        "points_against": sum(x["allowed"] for x in rows) / count,
        "rest_days": round(rest, 2),
    }


def _record_pct(record: str | None) -> float:
    if not record or "-" not in record:
        return 0.5
    try:
        wins, losses = [int(part) for part in record.split("-")[:2]]
        return wins / (wins + losses) if wins + losses else 0.5
    except (TypeError, ValueError):
        return 0.5


def _logit(probability: float) -> float:
    p = min(0.95, max(0.05, probability))
    return log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + exp(-value))


def _rate_features(stats: dict[str, float]) -> dict[str, float]:
    fga = stats.get("avgFieldGoalsAttempted", 0.0)
    fgm = stats.get("avgFieldGoalsMade", 0.0)
    three_made = stats.get("avgThreePointFieldGoalsMade", 0.0)
    fta = stats.get("avgFreeThrowsAttempted", 0.0)
    oreb = stats.get("avgOffensiveRebounds", 0.0)
    turnovers = stats.get("avgTurnovers", 0.0)
    possessions = fga + 0.44 * fta - oreb + turnovers
    return {
        "effective_fg_pct": (fgm + 0.5 * three_made) / fga if fga else 0.5,
        "turnover_rate": turnovers / possessions if possessions else 0.15,
        "rebound_rate_proxy": stats.get("avgRebounds", 0.0),
        "points_per_100": stats.get("avgPoints", 0.0) / possessions * 100 if possessions else 100.0,
        "possessions": possessions,
    }


def predict_game(*, home: dict[str, Any], away: dict[str, Any], home_schedule: dict[str, Any],
                 away_schedule: dict[str, Any], home_stats: dict[str, Any], away_stats: dict[str, Any],
                 game_date: str | None, home_edge: float, neutral_site: bool = False) -> dict[str, Any]:
    before = _parse_date(game_date) or datetime.now(timezone.utc)
    home_form = recent_form(home_schedule, str(home.get("id") or ""), before)
    away_form = recent_form(away_schedule, str(away.get("id") or ""), before)
    hs, aws = flatten_team_stats(home_stats), flatten_team_stats(away_stats)
    home_rates, away_rates = _rate_features(hs), _rate_features(aws)
    home_record, away_record = _record_pct(home.get("record")), _record_pct(away.get("record"))

    contributions = {
        "season_record": 0.55 * (_logit(home_record) - _logit(away_record)),
        "recent_form": 0.85 * (home_form["win_pct"] - away_form["win_pct"]),
        "recent_margin": 0.035 * (home_form["avg_margin"] - away_form["avg_margin"]),
        "shooting_efficiency": 2.2 * (home_rates["effective_fg_pct"] - away_rates["effective_fg_pct"]),
        "turnover_control": 1.5 * (away_rates["turnover_rate"] - home_rates["turnover_rate"]),
        "rebounding": 0.012 * (home_rates["rebound_rate_proxy"] - away_rates["rebound_rate_proxy"]),
        "rest": 0.035 * max(-3.0, min(3.0, (home_form["rest_days"] or 2.0) - (away_form["rest_days"] or 2.0))),
        "home_court": 0.0 if neutral_site else home_edge * 4.0,
    }
    raw_probability = _sigmoid(sum(contributions.values()))
    sample_games = min(home_form["games"], away_form["games"])
    stats_ready = bool(hs and aws)
    reliability = min(1.0, sample_games / 10.0) * (1.0 if stats_ready else 0.72)
    baseline = 0.5 + (home_record - away_record) * 0.38 + (0.0 if neutral_site else home_edge)
    home_probability = baseline * (1.0 - reliability) + raw_probability * reliability
    home_probability = min(0.92, max(0.08, home_probability))

    factor_context = {
        "season_record": (home_record, away_record, "win_pct",
                          "Season win percentage supplies the long-run team-strength anchor."),
        "recent_form": (home_form["win_pct"], away_form["win_pct"], "win_pct",
                        "Win rate over up to 10 completed games before tip-off captures current form."),
        "recent_margin": (home_form["avg_margin"], away_form["avg_margin"], "points",
                          "Average recent scoring margin separates narrow results from decisive performance."),
        "shooting_efficiency": (home_rates["effective_fg_pct"], away_rates["effective_fg_pct"], "percentage",
                                "Effective field-goal percentage rewards made threes for their extra point value."),
        "turnover_control": (home_rates["turnover_rate"], away_rates["turnover_rate"], "percentage_lower_better",
                             "Estimated turnovers per possession measure how reliably each offense protects the ball."),
        "rebounding": (home_rates["rebound_rate_proxy"], away_rates["rebound_rate_proxy"], "rebounds",
                       "Rebounds per game proxy second-chance creation and possession control."),
        "rest": (home_form["rest_days"], away_form["rest_days"], "days",
                 "Days since the latest completed game approximate fatigue and recovery."),
        "home_court": (0.0 if neutral_site else home_edge, 0.0, "probability_adjustment",
                       "League-calibrated venue advantage; it is zero for neutral-site games."),
    }
    factors = sorted(({
        "name": name,
        "contribution": round(value, 4),
        "direction": "home" if value > 0 else "away" if value < 0 else "neutral",
        "home_value": factor_context[name][0],
        "away_value": factor_context[name][1],
        "unit": factor_context[name][2],
        "explanation": factor_context[name][3],
    } for name, value in contributions.items()), key=lambda x: abs(x["contribution"]), reverse=True)
    return {
        "home_win_probability": round(home_probability, 4),
        "away_win_probability": round(1.0 - home_probability, 4),
        "pick": "home" if home_probability >= 0.5 else "away",
        "confidence": "high" if reliability >= 0.8 and abs(home_probability - 0.5) >= 0.12 else "medium" if reliability >= 0.5 else "low",
        "model": "basketball multi-factor logistic v1",
        "model_version": "bb-form-v1",
        "reliability": round(reliability, 3),
        "features": {"home": {"record_pct": round(home_record, 3), "recent": home_form, **home_rates},
                     "away": {"record_pct": round(away_record, 3), "recent": away_form, **away_rates}},
        "factors": factors,
        "guardrails": {"pregame_only": True, "probability_floor": 0.08, "probability_ceiling": 0.92,
                       "fallback": "record-strength baseline when history is sparse",
                       "method": "weighted factors -> logistic probability -> reliability blend -> probability bounds"},
    }
