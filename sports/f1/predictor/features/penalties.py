"""Structured grid-penalty / pit-lane / disqualification parsing.

Derived from REAL official fields only:
  - grid penalties + pit-lane starts from the Jolpica qualifying-vs-grid delta
    (computed by ``weekend._grid_evidence``, confidence ~0.86);
  - disqualifications from the official result ``status`` string.

Penalty REASONS, steward decisions, and component/PU-element changes are NOT
available as structured data in any connected feed, so they are deliberately
omitted here (labelled in the report) rather than guessed.
"""
from __future__ import annotations

from typing import Any

_DSQ_TOKENS = ("disqualif", "excluded", "dsq")
_MECH_TOKENS = (
    "engine", "gearbox", "hydraulic", "power unit", "powerunit", "electrical",
    "transmission", "brake", "suspension", "oil", "water", "fuel", "wheel",
    "puncture", "tyre", "tire", "overheating",
)
_INCIDENT_TOKENS = ("accident", "collision", "spun", "damage", "crash", "withdrew")


def is_disqualified(status: str | None) -> bool:
    s = str(status or "").lower()
    return any(t in s for t in _DSQ_TOKENS)


def classify_result_status(status: str | None) -> str:
    """Categorize an official result status: finished / disqualified / mechanical /
    incident / other / unknown. Fixes DSQ being lumped in with generic DNFs."""
    s = str(status or "").lower()
    if not s:
        return "unknown"
    if is_disqualified(s):
        return "disqualified"
    if s == "finished" or s.startswith("+") or "lap" in s:
        return "finished"
    if any(t in s for t in _MECH_TOKENS):
        return "mechanical"
    if any(t in s for t in _INCIDENT_TOKENS):
        return "incident"
    return "other"


def build_penalty_report(grid_evidence: dict[str, Any] | None, results: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Combine grid evidence (from ``weekend._grid_evidence``) with result statuses
    into a structured penalty report: grid penalties, pit-lane starts, DSQs."""
    grid_evidence = grid_evidence or {}
    grid_drivers = grid_evidence.get("drivers") or {}

    grid_penalties: list[dict[str, Any]] = []
    pit_lane_starts: list[dict[str, Any]] = []
    for item in grid_drivers.values():
        if (item.get("grid_penalty") or 0) > 0:
            grid_penalties.append({
                "driver_id": item.get("driver_id"),
                "driver_code": item.get("driver_code"),
                "qualifying_position": item.get("qualifying_position"),
                "grid_position": item.get("grid_position"),
                "grid_penalty": item.get("grid_penalty"),
                "confidence": item.get("confidence"),
            })
        if item.get("pit_lane_start"):
            pit_lane_starts.append({
                "driver_id": item.get("driver_id"),
                "driver_code": item.get("driver_code"),
                "grid_position": item.get("grid_position"),
                "confidence": item.get("confidence"),
            })

    disqualifications: list[dict[str, Any]] = []
    for row in results or []:
        if is_disqualified(row.get("status")):
            disqualifications.append({
                "driver_id": row.get("driver_id"),
                "driver_code": row.get("driver_code"),
                "position": row.get("position"),
                "status": row.get("status"),
                "source": "official_results",
                "confidence": 0.95,
            })

    grid_penalties.sort(key=lambda x: -(x.get("grid_penalty") or 0))
    return {
        "grid_penalties": grid_penalties,
        "pit_lane_starts": pit_lane_starts,
        "disqualifications": disqualifications,
        "summary": {
            "grid_penalties": len(grid_penalties),
            "pit_lane_starts": len(pit_lane_starts),
            "disqualifications": len(disqualifications),
        },
        "source": grid_evidence.get("source") or "jolpica_qualifying_grid",
        "confidence": grid_evidence.get("confidence") or 0.0,
        "available": bool(grid_drivers) or bool(disqualifications),
        "note": "Penalty reasons and component/PU-element changes are not available as structured data and are omitted.",
    }
