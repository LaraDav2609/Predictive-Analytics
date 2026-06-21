"""Safe F1 assistant tools."""

from __future__ import annotations

from typing import Any

from .schemas import AssistantToolSpec


TOOL_SPECS = [
    AssistantToolSpec(
        id="f1_health",
        label="Backend Health",
        description="Check F1 backend readiness, loaded driver/constructor/race counts, and source availability.",
    ),
    AssistantToolSpec(
        id="storage_health",
        label="Storage Health",
        description="Check optional Redis and ClickHouse storage/cache availability.",
    ),
    AssistantToolSpec(
        id="live_diagnostics",
        label="Live Diagnostics",
        description="Check live source mode, recorder status, OpenF1 rows, missing live groups, and confidence blockers.",
    ),
    AssistantToolSpec(
        id="route_context",
        label="Route Context",
        description="Explain what the assistant knows about the current F1 page.",
    ),
    AssistantToolSpec(
        id="car_performance_analysis",
        label="Car Performance Analyst",
        description="Analyze F1 car model traits: low-speed performance, tyre degradation, top speed, sector strengths, reliability, and teammate deltas.",
    ),
]


def list_tools() -> list[dict[str, Any]]:
    return [tool.model_dump() for tool in TOOL_SPECS]


async def run_diagnostics(*, context, client, openf1=None, live_engine=None, live_recorder=None, storage=None) -> dict[str, Any]:
    storage_health = await storage.health() if storage is not None else {"available": False, "reason": "storage_not_configured"}
    recorder_status = live_recorder.status() if live_recorder is not None else {"available": False, "reason": "recorder_not_configured"}
    live_payload: dict[str, Any] = {}
    if context.round and live_engine is not None:
        try:
            race = client.get_race_by_round(context.round) if client else None
            live_payload = (
                await live_engine.get_state(race, client.get_drivers(), context.session or "race", force=False)
                if race is not None and client is not None
                else {}
            )
        except Exception as exc:
            live_payload = {"ok": False, "reason": f"live_diagnostics_error:{exc.__class__.__name__}"}
    return {
        "ok": True,
        "backend": {
            "drivers": len(client.get_drivers()) if client else 0,
            "constructors": len(client.get_constructors()) if client else 0,
            "races": len(client.get_races()) if client else 0,
            "openf1_available": openf1 is not None,
        },
        "storage": storage_health,
        "recorder": recorder_status,
        "live": {
            "source_mode": live_payload.get("source_mode") or live_payload.get("mode") or "unavailable",
            "confidence": live_payload.get("confidence"),
            "fallback_reason": live_payload.get("fallback_reason") or live_payload.get("reason"),
            "missing_groups": live_payload.get("missing_groups") or [],
            "driver_count": len(live_payload.get("drivers") or []),
        },
        "route": context.model_dump(),
    }
