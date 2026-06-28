"""Map telemetry model output into simulator initial-state adjustments."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from sports.f1.ml.telemetry.types import TelemetryModelOutput


def apply_telemetry_adjustments(
    initial_state: dict[str, Any],
    output: TelemetryModelOutput | None,
    *,
    min_confidence: float = 0.20,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply telemetry adjustments to the simulator initial state.

    Returns the adjusted state plus metadata describing whether telemetry was
    applied. The function is side-effect free.
    """

    state = deepcopy(initial_state)
    if output is None:
        return state, _metadata(False, "telemetry_output_missing", None)
    if output.confidence < min_confidence:
        return state, _metadata(False, "telemetry_confidence_below_threshold", output)

    codes = list(state.get("driver_codes") or [])
    means = list(state.get("driver_mean_pace_s") or [])
    sigmas = list(state.get("driver_pace_sigma_s") or [])
    dnfs = list(state.get("driver_dnf_rate_per_lap") or [])
    if not (len(codes) == len(means) == len(sigmas) == len(dnfs)):
        return state, _metadata(False, "initial_state_shape_mismatch", output)

    start_deg = list(state.get("driver_starting_deg_slope") or [0.0] * len(codes))
    pit_values = list(state.get("driver_pit_window_value_s") or [0.0] * len(codes))
    applied: list[str] = []
    for index, code in enumerate(codes):
        adjustment = output.driver_adjustments.get(str(code).upper()) or output.driver_adjustments.get(str(code))
        if not adjustment:
            continue
        means[index] = round(float(means[index]) + float(adjustment.get("clean_air_pace_shift_s") or 0.0), 5)
        sigmas[index] = round(max(0.05, float(sigmas[index]) * float(adjustment.get("pace_sigma_multiplier") or 1.0)), 5)
        dnfs[index] = round(max(0.00001, min(0.25, float(dnfs[index]) * float(adjustment.get("dnf_hazard_multiplier") or 1.0))), 7)
        start_deg[index] = round(float(start_deg[index]) + float(adjustment.get("tire_deg_slope_delta") or 0.0), 5)
        pit_values[index] = round(float(pit_values[index]) + float(adjustment.get("pit_window_value_s") or 0.0), 5)
        applied.append(str(code).upper())

    state["driver_mean_pace_s"] = means
    state["driver_pace_sigma_s"] = sigmas
    state["driver_dnf_rate_per_lap"] = dnfs
    state["driver_starting_deg_slope"] = start_deg
    state["driver_pit_window_value_s"] = pit_values
    state["telemetry_adjustments_applied"] = applied
    return state, {
        **_metadata(bool(applied), None if applied else "no_matching_driver_adjustments", output),
        "applied_driver_codes": applied,
    }


def _metadata(applied: bool, reason: str | None, output: TelemetryModelOutput | None) -> dict[str, Any]:
    return {
        "telemetry_model_used": bool(applied),
        "telemetry_model_id": output.model_id if output else None,
        "telemetry_model_version": output.model_version if output else None,
        "telemetry_confidence": output.confidence if output else None,
        "telemetry_source_mode": output.source_mode if output else None,
        "telemetry_missing_groups": output.missing_groups if output else [],
        "telemetry_fallback_reason": reason,
    }
