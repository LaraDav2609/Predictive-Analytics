"""Race weekend stage detection for probability calibration."""

from __future__ import annotations

from typing import Any

VALID_STAGES = {"pre_weekend", "practice_available", "post_qualifying", "live", "completed"}


def detect_stage(
    profile: dict[str, Any] | None = None,
    truth: dict[str, Any] | None = None,
    *,
    live: bool = False,
    requested_stage: str | None = "auto",
) -> str:
    """Infer the best probability stage from race profile and truth data."""

    requested = (requested_stage or "auto").strip().lower()
    if requested in VALID_STAGES:
        return requested

    profile = profile or {}
    truth = truth or {}
    context = profile.get("context") or {}
    status = str((profile.get("race") or {}).get("status") or truth.get("status") or "").lower()
    source_mode = str(truth.get("source_mode") or truth.get("mode") or "").lower()

    if _has_final_classification(profile, truth) or status in {"completed", "final", "classified"}:
        return "completed"
    if live and source_mode in {"live", "recent", "recorded", "recorded_confident"} and bool(truth.get("drivers") or truth.get("by_driver_id")):
        return "live"
    if _has_rows(profile.get("qualifying")) or _has_grid(profile):
        return "post_qualifying"
    if _has_completed_practice(profile) or int(context.get("completed_sessions") or 0) > 0:
        return "practice_available"
    return "pre_weekend"


def _has_rows(value: Any) -> bool:
    return isinstance(value, list) and len(value) > 0


def _has_grid(profile: dict[str, Any]) -> bool:
    for row in profile.get("results") or []:
        if row.get("grid") not in {None, "", "0", 0}:
            return True
    return False


def _has_final_classification(profile: dict[str, Any], truth: dict[str, Any]) -> bool:
    if _has_rows(profile.get("results")):
        return True
    if truth.get("source_mode") == "historical" and bool(truth.get("drivers") or truth.get("by_driver_id")):
        return True
    return False


def _has_completed_practice(profile: dict[str, Any]) -> bool:
    for session in profile.get("sessions") or []:
        name = str(session.get("name") or session.get("session") or "").lower()
        status = str(session.get("status") or "").lower()
        if ("practice" in name or name in {"fp1", "fp2", "fp3"}) and status in {"completed", "done", "final"}:
            return True
    return False
