"""Page-aware context builder for the F1 assistant."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from .schemas import F1AssistantContext


def parse_f1_route(route: str) -> dict[str, Any]:
    parsed = urlparse(route or "/F1")
    path = parsed.path or "/F1"
    query = parse_qs(parsed.query or "")
    info: dict[str, Any] = {
        "route": route or "/F1",
        "path": path,
        "page_type": "home",
        "tab": (query.get("tab") or [None])[0],
        "session": (query.get("session") or [None])[0] or "race",
    }
    if match := re.match(r"^/F1/Race/(\d+)", path, re.IGNORECASE):
        info["page_type"] = "race"
        info["round"] = int(match.group(1))
    elif match := re.match(r"^/F1/Driver/([^/?#]+)", path, re.IGNORECASE):
        info["page_type"] = "driver"
        info["driver_id"] = match.group(1)
    elif match := re.match(r"^/F1/Constructor/([^/?#]+)", path, re.IGNORECASE):
        info["page_type"] = "constructor"
        info["constructor_id"] = match.group(1)
    elif re.match(r"^/F1/ModelLab", path, re.IGNORECASE):
        info["page_type"] = "model_lab"
    elif re.match(r"^/F1/Markets", path, re.IGNORECASE):
        info["page_type"] = "markets"
    elif re.match(r"^/F1/Workspace", path, re.IGNORECASE):
        info["page_type"] = "workspace"
    elif match := re.match(r"^/F1/(Replay|Simulation|Live)/?(\d+)?", path, re.IGNORECASE):
        info["page_type"] = match.group(1).lower()
        if match.group(2):
            info["round"] = int(match.group(2))
    return info


async def build_context(
    *,
    route: str,
    client,
    predictor=None,
    live_engine=None,
    storage=None,
) -> F1AssistantContext:
    route_info = parse_f1_route(route)
    current: dict[str, Any] = {}
    citations: list[dict[str, str]] = []
    warnings: list[str] = []
    title = "Formula 1 Command Wall"
    source_mode: str | None = None
    confidence: float | None = None
    missing_groups: list[str] = []

    races = client.get_races() if client else []
    drivers = client.get_drivers() if client else []
    constructors = client.get_constructors() if client else []
    next_race = _next_race(races)
    navigation = {
        "home": "/F1",
        "model_lab": "/F1/ModelLab",
        "markets": "/F1/Markets",
        "workspace": "/F1/Workspace",
    }
    if next_race is not None:
        navigation["next_gp"] = f"/F1/Race/{getattr(next_race, 'round', '')}"

    try:
        if route_info["page_type"] == "race" and route_info.get("round"):
            round_num = int(route_info["round"])
            profile = await client.get_race_profile(round_num, predictor) if client else {}
            title = str(profile.get("race", {}).get("name") or f"Round {round_num}")
            truth_summary = _summarize_profile(profile)
            current.update({"profile": truth_summary, "round": round_num})
            confidence = _to_float(((profile.get("race") or {}).get("prediction") or {}).get("confidence"))
            citations.append({"label": "Race profile", "source": f"/api/f1/races/{round_num}/profile"})
            navigation.update({
                "overview": f"/F1/Race/{round_num}?tab=overview",
                "qualifying": f"/F1/Race/{round_num}?tab=qualifying",
                "race": f"/F1/Race/{round_num}?tab=race",
                "live": f"/F1/Race/{round_num}?tab=live",
            })
            if live_engine is not None:
                try:
                    race_obj = client.get_race_by_round(round_num) if client else None
                    state = (
                        await live_engine.get_state(race_obj, drivers, route_info.get("session") or "race", force=False)
                        if race_obj is not None
                        else {}
                    )
                    source_mode = state.get("source_mode") or state.get("mode")
                    confidence = _to_float(state.get("confidence")) if state.get("confidence") is not None else confidence
                    missing_groups = list(state.get("missing_groups") or [])
                    current["live"] = _compact_live_state(state)
                    citations.append({"label": "Live state", "source": f"/api/f1/live/{round_num}"})
                except Exception as exc:
                    warnings.append(f"Live state unavailable: {exc.__class__.__name__}")
        elif route_info["page_type"] == "driver" and route_info.get("driver_id"):
            profile = await client.get_driver_profile(str(route_info["driver_id"])) if client else {}
            driver = profile.get("driver") or {}
            title = driver.get("name") or driver.get("last_name") or str(route_info["driver_id"])
            current["driver"] = {
                "id": driver.get("id"),
                "code": driver.get("code"),
                "team": driver.get("team"),
                "points": driver.get("points"),
                "position": driver.get("position"),
                "season": profile.get("selected_season"),
                "stats": profile.get("stats") or {},
            }
            citations.append({"label": "Driver profile", "source": f"/api/f1/drivers/{route_info['driver_id']}"})
        elif route_info["page_type"] == "constructor" and route_info.get("constructor_id"):
            profile = await client.get_constructor_profile(str(route_info["constructor_id"])) if client else {}
            constructor = profile.get("constructor") or {}
            title = constructor.get("name") or str(route_info["constructor_id"])
            current["constructor"] = {
                "id": constructor.get("id"),
                "name": constructor.get("name"),
                "points": constructor.get("points"),
                "position": constructor.get("position"),
                "season": profile.get("selected_season"),
                "stats": profile.get("stats") or {},
            }
            citations.append({"label": "Constructor profile", "source": f"/api/f1/constructors/{route_info['constructor_id']}"})
        else:
            title = {
                "model_lab": "Global Model Lab",
                "markets": "F1 Markets",
                "workspace": "F1 Workspace",
                "replay": "F1 Replay",
                "simulation": "F1 Simulation",
                "live": "F1 Live",
            }.get(route_info["page_type"], "Formula 1 Command Wall")
            current["summary"] = {
                "drivers": len(drivers),
                "constructors": len(constructors),
                "races": len(races),
                "next_race": _race_label(next_race),
            }
            citations.append({"label": "F1 overview", "source": "/api/f1/calendar + /api/f1/drivers"})
    except Exception as exc:
        warnings.append(f"Context degraded: {exc.__class__.__name__}")

    return F1AssistantContext(
        route=route or "/F1",
        page_type=route_info["page_type"],
        round=route_info.get("round"),
        driver_id=route_info.get("driver_id"),
        constructor_id=route_info.get("constructor_id"),
        tab=route_info.get("tab"),
        session=route_info.get("session") or "race",
        title=title,
        source_mode=source_mode,
        confidence=confidence,
        missing_groups=missing_groups,
        current=current,
        navigation=navigation,
        citations=citations,
        warnings=warnings,
    )


def _next_race(races: list) -> Any | None:
    for race in races:
        status = str(getattr(race, "status", "") or "").lower()
        if status not in {"completed", "done", "finished"}:
            return race
    return races[-1] if races else None


def _race_label(race: Any | None) -> dict[str, Any] | None:
    if race is None:
        return None
    return {
        "round": getattr(race, "round", None),
        "name": getattr(race, "name", None),
        "date": str(getattr(race, "date", "") or ""),
        "circuit": getattr(race, "circuit", None),
        "country": getattr(race, "country", None),
    }


def _summarize_profile(profile: dict[str, Any]) -> dict[str, Any]:
    race = profile.get("race") or {}
    prediction = race.get("prediction") or profile.get("prediction") or {}
    return {
        "name": race.get("name"),
        "round": race.get("round"),
        "date": race.get("date"),
        "format": race.get("format"),
        "status": race.get("status"),
        "top_pick": prediction.get("top_pick") or prediction.get("winner") or race.get("top_pick"),
        "confidence": prediction.get("confidence") or race.get("confidence"),
        "source_mode": profile.get("source_mode") or prediction.get("source_mode"),
    }


def _compact_live_state(state: dict[str, Any]) -> dict[str, Any]:
    drivers = state.get("drivers") or state.get("running_order") or []
    return {
        "source_mode": state.get("source_mode") or state.get("mode"),
        "confidence": state.get("confidence"),
        "data_age_seconds": state.get("data_age_seconds"),
        "fallback_reason": state.get("fallback_reason"),
        "driver_count": len(drivers) if isinstance(drivers, list) else 0,
        "top": drivers[:5] if isinstance(drivers, list) else [],
    }


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
