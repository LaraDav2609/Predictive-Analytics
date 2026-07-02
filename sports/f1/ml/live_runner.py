"""Live/replay runner for the F1 ML simulator.

This module is intentionally small and source-conservative: it reuses the
existing simulator, artifact loader, synthetic fixtures, and shared probability
publisher.  It does not reach into paid or login-protected feeds.  When real
timing rows are unavailable, the runner emits an explicitly estimated payload
with a low confidence ceiling.
"""

from __future__ import annotations

import json
import ast
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from common.ml.bridge.outcome_publisher import InMemoryOutcomePublisher, OutcomePublisher
from common.ml.types import OutcomeProbability
from sports.f1.ml.artifacts.loader import (
    apply_ml_trained_inputs_to_initial_state,
    load_ml_trained_inputs,
    load_simulator_model_bundle,
)
from sports.f1.ml.common.types import Race, SessionType
from sports.f1.ml.markets.mapper import (
    dnf_probabilities,
    fastest_lap_probabilities,
    podium_probabilities,
    top_k_probabilities,
    winner_probabilities,
)
from sports.f1.ml.providers.synthetic_provider import SyntheticProvider, SyntheticRaceConfig
from sports.f1.ml.simulator.race_sim import PhysicalParams, SimConfig, simulate_race


SOURCE_MODES = {"auto", "openf1", "fastf1-recorded", "replay", "estimated"}


@dataclass
class LiveRunnerConfig:
    race: str
    season: int | None = None
    round_num: int | None = None
    session: str = "race"
    model_id: str = "ml_simulator_v1"
    artifact_path: str | None = None
    source: str = "auto"
    recording_path: str | None = None
    poll_seconds: float = 10.0
    once: bool = False
    publish: bool = False
    redis_url: str = "redis://localhost:6379/0"
    dry_run: bool = False
    n_iterations: int = 800
    physical: bool = True
    lap: int | None = None
    race_obj: Any | None = None
    drivers: list[Any] | None = None
    constructors: list[Any] | None = None
    features: dict[str, Any] | None = None
    sentiment: dict[str, Any] | None = None
    race_truth: dict[str, Any] | None = None
    weekend_evidence: dict[str, Any] | None = None
    live_state: dict[str, Any] | None = None
    track: dict[str, Any] | None = None
    weather: dict[str, Any] | None = None
    tires: dict[str, Any] | None = None


@dataclass
class LiveRunnerResult:
    payload: dict[str, Any]
    published_records: list[OutcomeProbability] = field(default_factory=list)


def run_live_once(config: LiveRunnerConfig, *, publisher: Any | None = None) -> LiveRunnerResult:
    """Run one live/replay update and optionally publish market probabilities."""

    source = _normalise_source(config.source)
    if _has_production_context(config):
        return _run_production_live_once(config, source=source, publisher=publisher)

    provider, race_obj, initial_state = _base_synthetic_state(config)
    artifact_meta, model_bundle = _apply_artifact_inputs(config, initial_state)
    live_state = _resolve_live_state(config, provider, race_obj, initial_state, source)
    adjusted_state = _apply_live_state_to_initial_state(initial_state, live_state)

    models = {"simulator_model_bundle": model_bundle} if model_bundle is not None else {}
    sim_result = simulate_race(
        race=race_obj,
        initial_state=adjusted_state,
        models=models,
        config=SimConfig(
            n_iterations=max(20, int(config.n_iterations)),
            seed=_seed(config, live_state),
            enable_physical=bool(config.physical),
            physical=PhysicalParams(),
        ),
    )
    payload = _payload_from_result(
        config=config,
        race_obj=race_obj,
        sim_result=sim_result,
        live_state=live_state,
        artifact_meta=artifact_meta,
        adjusted_state=adjusted_state,
    )
    records = _to_outcome_probabilities(payload)
    publish_status = _publish_records(config, records, publisher=publisher)
    payload["publish_status"] = publish_status
    return LiveRunnerResult(payload=payload, published_records=records)


def _has_production_context(config: LiveRunnerConfig) -> bool:
    return bool(config.race_obj is not None and config.drivers)


def _run_production_live_once(
    config: LiveRunnerConfig,
    *,
    source: str,
    publisher: Any | None = None,
) -> LiveRunnerResult:
    from sports.f1.predictor.models.ml_simulator import _build_simulation_context, _ml_race

    features = dict(config.features or {})
    if config.weekend_evidence is not None:
        features["weekend_evidence"] = config.weekend_evidence
    if config.race_truth is not None:
        features["race_truth"] = config.race_truth
    if config.live_state is not None:
        features["live_state"] = config.live_state
    if config.artifact_path:
        features["ml_artifact_path"] = config.artifact_path
    features["session"] = config.session
    features["ml_simulator_iterations"] = max(20, int(config.n_iterations))

    context = _build_simulation_context(
        config.race_obj,
        list(config.drivers or []),
        list(config.constructors or []),
        features,
        dict(config.sentiment or {}),
    )
    sim_config = context["config"]
    sim_config.n_iterations = max(20, int(config.n_iterations))
    sim_config.enable_physical = bool(config.physical)
    race_obj = _ml_race(config.race_obj, context.get("track") or {})
    live_state = _production_live_state(config, context, source=source)
    sim_result = simulate_race(
        race=race_obj,
        initial_state=context["initial_state"],
        models=context["models"],
        config=sim_config,
    )
    adjusted_state = dict(context["initial_state"])
    adjusted_state["live_lap"] = live_state.get("lap")
    artifact_meta = dict(context.get("ml_metadata") or {})
    artifact_meta.setdefault("ml_input_source", artifact_meta.get("input_source"))
    payload = _payload_from_result(
        config=config,
        race_obj=race_obj,
        sim_result=sim_result,
        live_state=live_state,
        artifact_meta=artifact_meta,
        adjusted_state=adjusted_state,
    )
    payload.update({
        "production_context": True,
        "race_truth_used": bool((config.race_truth or features.get("race_truth") or {}).get("ok", True)),
        "weekend_evidence_used": bool((config.weekend_evidence or features.get("weekend_evidence") or {}).get("ok", True)),
        "live_state_used": bool(config.live_state or features.get("live_state")),
        "race_truth_source_mode": (config.race_truth or features.get("race_truth") or {}).get("source_mode"),
        "weekend_evidence_source_mode": (config.weekend_evidence or features.get("weekend_evidence") or {}).get("source_mode"),
        "ml_live_runner_version": "production_live_runner_v1",
        "track": config.track or context.get("track") or {},
        "weather": config.weather or context.get("weather") or {},
        "tires": config.tires or context.get("tires") or {},
    })
    records = _to_outcome_probabilities(payload)
    publish_status = _publish_records(config, records, publisher=publisher)
    payload["publish_status"] = publish_status
    return LiveRunnerResult(payload=payload, published_records=records)


def _production_live_state(config: LiveRunnerConfig, context: dict[str, Any], *, source: str) -> dict[str, Any]:
    features = dict(config.features or {})
    truth = config.race_truth or features.get("race_truth") or {}
    evidence = config.weekend_evidence or features.get("weekend_evidence") or {}
    live_state = config.live_state or features.get("live_state") or {}
    source_mode = (
        live_state.get("source_mode")
        or live_state.get("mode")
        or truth.get("source_mode")
        or evidence.get("source_mode")
        or ("estimated" if source == "auto" else source)
    )
    confidence_ceiling = _confidence_ceiling(source_mode)
    confidence = _safe_float(live_state.get("confidence"))
    if confidence is None:
        confidence = _safe_float(truth.get("confidence"))
    if confidence is None:
        confidence = _safe_float(evidence.get("confidence"))
    if confidence is None:
        confidence = _safe_float((context.get("ml_metadata") or {}).get("ml_confidence")) or 0.22
    confidence = round(min(float(confidence), confidence_ceiling), 4)

    truth_drivers = _driver_state_maps(truth)
    evidence_drivers = (evidence.get("drivers") or {}) if isinstance(evidence.get("drivers"), dict) else {}
    live_drivers = _driver_state_maps(live_state)
    rows: list[dict[str, Any]] = []
    max_lap = 0
    for index, driver in enumerate(config.drivers or [], start=1):
        row = _merge_driver_state(driver, index, truth_drivers, evidence_drivers, live_drivers, source_mode, confidence)
        max_lap = max(max_lap, int(row.get("lap") or 0))
        rows.append(row)

    groups = set((context.get("ml_metadata") or {}).get("evidence_groups_used") or [])
    if truth:
        groups.add("race_truth")
    if evidence:
        groups.add("weekend_evidence")
    if live_state:
        groups.add("live_state")
    if any(row.get("gap_to_leader") is not None or row.get("interval") is not None for row in rows):
        groups.add("gaps")
    if any(row.get("compound") for row in rows):
        groups.add("tyres")
    if any(row.get("representative_lap") for row in rows):
        groups.add("lap_times")
    if rows:
        groups.add("positions")
    missing = set(truth.get("missing_groups") or [])
    missing.update(evidence.get("missing_groups") or [])
    missing.update(_missing_groups(sorted(groups)))
    return {
        "source_mode": source_mode,
        "confidence": confidence,
        "confidence_ceiling": confidence_ceiling,
        "confidence_reason": _confidence_reason(source_mode, groups, confidence_ceiling),
        "fallback_reason": live_state.get("fallback_reason") or truth.get("fallback_reason") or evidence.get("fallback_reason"),
        "lap": max_lap or _safe_int(truth.get("lap")) or _safe_int(live_state.get("lap")) or 0,
        "session_status": live_state.get("session_status") or truth.get("status") or ("live" if source_is_reliable(source_mode) else "estimated"),
        "drivers": sorted(rows, key=lambda row: row.get("position") or 99),
        "evidence_groups_used": sorted(groups),
        "missing_evidence_groups": sorted(item for item in missing if item),
    }


def run_live_loop(
    config: LiveRunnerConfig,
    *,
    publisher: Any | None = None,
    on_result: Any | None = None,
) -> list[LiveRunnerResult]:
    """Run the live loop.  In tests and smoke mode, ``once`` keeps this finite."""

    results: list[LiveRunnerResult] = []
    while True:
        result = run_live_once(config, publisher=publisher)
        results.append(result)
        if on_result is not None:
            on_result(result)
        if config.once:
            return results
        if len(results) > 1:
            results[:] = results[-1:]
        time.sleep(max(1.0, float(config.poll_seconds)))


def _normalise_source(source: str) -> str:
    value = (source or "auto").strip().lower()
    if value not in SOURCE_MODES:
        raise ValueError(f"source must be one of {sorted(SOURCE_MODES)}")
    return value


def _base_synthetic_state(config: LiveRunnerConfig) -> tuple[SyntheticProvider, Race, dict[str, Any]]:
    race_id = _normalise_synthetic_race_id(config.race, config.season, config.round_num)
    season, round_num = _synthetic_season_round(race_id)
    provider = SyntheticProvider(SyntheticRaceConfig(race_id=race_id, season=season, round=round_num))
    race = provider.list_races(season)[0]
    pace = provider.driver_pace_table()
    sigma = provider.driver_sigma_table()
    dnf = provider.dnf_rate_table()
    drivers = list(pace.keys())
    initial_state = {
        "driver_codes": drivers,
        "driver_mean_pace_s": [float(pace[code]) for code in drivers],
        "driver_pace_sigma_s": [float(sigma.get(code, 0.35)) for code in drivers],
        "driver_dnf_rate_per_lap": [float(dnf.get(code, 0.0008)) for code in drivers],
        "driver_starting_compound": ["MEDIUM"] * len(drivers),
        "driver_pit_lap": [max(8, int(provider.config.n_laps * 0.42))] * len(drivers),
        "driver_pit_compound": ["HARD"] * len(drivers),
        "total_laps": int(provider.config.n_laps),
    }
    return provider, race, initial_state


def _apply_artifact_inputs(config: LiveRunnerConfig, initial_state: dict[str, Any]) -> tuple[dict[str, Any], Any | None]:
    meta = {
        "ml_input_source": "evidence_fallback",
        "ml_provider_sources": ["synthetic_initial_state"],
        "artifact_id": None,
        "artifact_version": None,
        "ml_artifact_readiness": {},
        "trained_artifacts_used": False,
        "ml_fallback_reason": None,
        "ml_model_contract_used": False,
        "ml_model_fallback_reason": None,
    }
    if not config.artifact_path:
        return meta, None
    trained_inputs = load_ml_trained_inputs(config.artifact_path, target_race=config.race)
    updated, apply_meta = apply_ml_trained_inputs_to_initial_state(initial_state, trained_inputs)
    initial_state.clear()
    initial_state.update(updated)
    bundle = load_simulator_model_bundle(config.artifact_path, target_race=config.race)
    summary = bundle.source_summary()
    meta.update({
        "artifact_id": trained_inputs.get("artifact_id") or apply_meta.get("artifact_id"),
        "artifact_version": trained_inputs.get("artifact_version") or apply_meta.get("artifact_version"),
        "ml_artifact_readiness": summary.get("ml_artifact_readiness") or trained_inputs.get("artifact_readiness") or {},
        "ml_model_contract_used": summary.get("ml_model_contract_used"),
        "ml_model_fallback_reason": summary.get("ml_model_fallback_reason"),
    })
    if int(apply_meta.get("applied_rows") or 0) > 0:
        meta.update({
            "ml_input_source": "trained_artifacts",
            "ml_provider_sources": ["synthetic_initial_state", "artifact_bundle"],
            "trained_artifacts_used": True,
        })
    else:
        meta.update({
            "ml_fallback_reason": apply_meta.get("fallback_reason") or trained_inputs.get("artifact_load_error") or "artifact_unusable",
            "ml_provider_sources": ["synthetic_initial_state", "artifact_invalid"],
        })
    return meta, bundle


def _resolve_live_state(
    config: LiveRunnerConfig,
    provider: SyntheticProvider,
    race: Race,
    initial_state: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    if source in {"auto", "fastf1-recorded"} and config.recording_path:
        recorded = _recorded_state(config.recording_path, initial_state)
        if recorded["source_mode"] != "estimated" or source == "fastf1-recorded":
            return recorded
    if source in {"auto", "replay"}:
        return _replay_state(config, provider, race, initial_state)
    if source == "openf1":
        return _estimated_state(initial_state, reason="openf1_live_not_available_in_ml_cli")
    return _estimated_state(initial_state, reason="estimated_source_selected")


def _replay_state(
    config: LiveRunnerConfig,
    provider: SyntheticProvider,
    race: Race,
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    laps = provider.laps(race, SessionType.RACE)
    target_lap = int(config.lap or max(1, min(6, provider.config.n_laps // 5)))
    rows = [lap for lap in laps if lap.lap_number <= target_lap]
    latest: dict[str, Any] = {}
    for lap in rows:
        latest[lap.driver_code] = lap
    if not latest:
        return _estimated_state(initial_state, reason="replay_has_no_rows")
    drivers = []
    for index, code in enumerate(initial_state["driver_codes"]):
        lap = latest.get(code)
        drivers.append({
            "driver_code": code,
            "position": int(lap.position) if lap else index + 1,
            "lap": target_lap,
            "gap_to_leader": float(lap.gap_to_leader_s or 0.0) if lap else None,
            "interval": None,
            "representative_lap": float(lap.lap_time_s) if lap else None,
            "compound": str(lap.compound.value if hasattr(lap.compound, "value") else lap.compound) if lap else "MEDIUM",
            "tyre_age": int(lap.tire_age_laps) if lap else target_lap,
            "pit_stops": 0,
        })
    return _state_from_driver_rows(
        drivers,
        source_mode="recorded",
        confidence=0.62,
        confidence_ceiling=0.75,
        fallback_reason=None,
        lap=target_lap,
        session_status="replay",
        evidence_groups=["replay_laps", "positions", "lap_times", "tyres"],
    )


def _recorded_state(path: str, initial_state: dict[str, Any]) -> dict[str, Any]:
    file = Path(path)
    if not file.exists():
        return _estimated_state(initial_state, reason="recording_file_missing")
    rows = _read_recording_rows(file)
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = _first(row, "driver_code", "DriverCode", "Tla", "tla")
        if not code:
            number = _safe_int(_first(row, "driver_number", "RacingNumber", "racing_number", "number", "Number"))
            if number is not None:
                codes = list(initial_state.get("driver_codes") or [])
                if 1 <= number <= len(codes):
                    code = codes[number - 1]
        if not code:
            continue
        latest[str(code).upper()] = row
    if not latest:
        return _estimated_state(initial_state, reason="recording_empty_or_unparsed")
    driver_rows = []
    max_lap = 0
    for index, code in enumerate(initial_state.get("driver_codes") or []):
        row = latest.get(str(code).upper()) or {}
        lap = _safe_int(_first(row, "lap", "laps", "lap_number", "LapNumber", "NumberOfLaps")) or 0
        max_lap = max(max_lap, lap)
        driver_rows.append({
            "driver_code": code,
            "position": _safe_int(_first(row, "position", "Position")) or index + 1,
            "lap": lap,
            "gap_to_leader": _first(row, "gap_to_leader", "GapToLeader", "gap"),
            "interval": _first(row, "interval", "IntervalToPositionAhead"),
            "representative_lap": _first(row, "representative_lap", "LastLapTime", "LapTime"),
            "compound": _first(row, "compound", "Compound", "tyre_compound") or "MEDIUM",
            "tyre_age": _safe_int(_first(row, "tyre_age", "TireAge", "TotalLaps")) or lap,
            "pit_stops": _safe_int(_first(row, "pit_stops", "NumberOfPitStops")) or 0,
        })
    parsed = len(latest)
    return _state_from_driver_rows(
        driver_rows,
        source_mode="recorded_confident" if parsed >= 18 else "recorded",
        confidence=min(0.82 if parsed >= 18 else 0.70, 0.38 + parsed / 60.0),
        confidence_ceiling=0.82 if parsed >= 18 else 0.75,
        fallback_reason=None if parsed else "recording_has_no_matching_drivers",
        lap=max_lap,
        session_status="recorded",
        evidence_groups=["recorded_timing", "positions"],
    )


def _read_recording_rows(path: Path) -> list[dict[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return []
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            return _flatten_records(parser(text))
        except Exception:
            pass
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for parser in (json.loads, ast.literal_eval):
            try:
                rows.extend(_flatten_records(parser(line)))
                break
            except Exception:
                continue
    return rows


def _flatten_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        if len(value) >= 2 and isinstance(value[0], str):
            rows = _flatten_records(value[1])
            for row in rows:
                row.setdefault("category", value[0])
                if len(value) > 2:
                    row.setdefault("timestamp", value[2])
            return rows
        rows: list[dict[str, Any]] = []
        for item in value:
            rows.extend(_flatten_records(item))
        return rows
    if isinstance(value, dict):
        if value and all(str(key).isdigit() for key in value):
            rows = []
            for key, item in value.items():
                if isinstance(item, dict):
                    rows.append({"driver_number": int(key), **item})
            if rows:
                return rows
        rows: list[dict[str, Any]] = []
        for key in ("data", "Data", "TimingData", "Lines", "Position", "PositionData", "TimingAppData"):
            if isinstance(value.get(key), (list, dict)):
                rows.extend(_flatten_records(value[key]))
        return rows or [value]
    return []


def _state_from_driver_rows(
    drivers: list[dict[str, Any]],
    *,
    source_mode: str,
    confidence: float,
    confidence_ceiling: float,
    fallback_reason: str | None,
    lap: int,
    session_status: str,
    evidence_groups: list[str],
) -> dict[str, Any]:
    drivers = sorted(drivers, key=lambda row: row.get("position") or 99)
    return {
        "source_mode": source_mode,
        "confidence": round(min(float(confidence), float(confidence_ceiling)), 4),
        "confidence_ceiling": confidence_ceiling,
        "confidence_reason": f"{source_mode}_timing_rows",
        "fallback_reason": fallback_reason,
        "lap": int(lap or 0),
        "session_status": session_status,
        "drivers": drivers,
        "evidence_groups_used": evidence_groups,
        "missing_evidence_groups": _missing_groups(evidence_groups),
    }


def _estimated_state(initial_state: dict[str, Any], *, reason: str) -> dict[str, Any]:
    drivers = [
        {
            "driver_code": code,
            "position": index + 1,
            "lap": 0,
            "gap_to_leader": None,
            "interval": None,
            "representative_lap": None,
            "compound": "MEDIUM",
            "tyre_age": 0,
            "pit_stops": 0,
        }
        for index, code in enumerate(initial_state.get("driver_codes") or [])
    ]
    return _state_from_driver_rows(
        drivers,
        source_mode="estimated",
        confidence=0.22,
        confidence_ceiling=0.25,
        fallback_reason=reason,
        lap=0,
        session_status="estimated",
        evidence_groups=["estimated_grid"],
    )


def _confidence_ceiling(source_mode: Any) -> float:
    mode = str(source_mode or "").lower()
    return {
        "unavailable": 0.05,
        "estimated": 0.25,
        "recording_pending": 0.25,
        "recent": 0.55,
        "historical": 0.65,
        "recorded": 0.75,
        "recorded_confident": 0.85,
        "live": 0.90,
    }.get(mode, 0.25)


def _confidence_reason(source_mode: Any, groups: set[str], ceiling: float) -> str:
    group_text = ",".join(sorted(groups)) or "no_real_evidence"
    return f"{source_mode or 'estimated'} evidence capped at {ceiling:.2f}; groups={group_text}"


def _driver_state_maps(payload: dict[str, Any] | None) -> dict[str, dict[str, dict[str, Any]]]:
    payload = payload or {}
    rows: list[dict[str, Any]] = []
    if isinstance(payload.get("drivers"), list):
        rows.extend(row for row in payload.get("drivers") if isinstance(row, dict))
    if isinstance(payload.get("by_driver_id"), dict):
        rows.extend(row for row in payload.get("by_driver_id").values() if isinstance(row, dict))
    if isinstance(payload.get("live_positions"), dict):
        rows.extend(
            {"driver_id": driver_id, **row}
            for driver_id, row in payload.get("live_positions").items()
            if isinstance(row, dict)
        )
    by_id: dict[str, dict[str, Any]] = {}
    by_code: dict[str, dict[str, Any]] = {}
    by_number: dict[str, dict[str, Any]] = {}
    for row in rows:
        driver_id = row.get("driver_id") or row.get("id")
        code = row.get("driver_code") or row.get("code")
        number = row.get("driver_number") or row.get("number")
        if driver_id:
            by_id[str(driver_id)] = row
        if code:
            by_code[str(code).upper()] = row
        if number is not None:
            by_number[str(number)] = row
    return {"by_id": by_id, "by_code": by_code, "by_number": by_number}


def _merge_driver_state(
    driver: Any,
    index: int,
    truth_maps: dict[str, dict[str, dict[str, Any]]],
    evidence_drivers: dict[str, Any],
    live_maps: dict[str, dict[str, dict[str, Any]]],
    source_mode: Any,
    confidence: float,
) -> dict[str, Any]:
    driver_id = str(getattr(driver, "id", "") or "")
    code = str(getattr(driver, "code", "") or "").upper()
    number = getattr(driver, "number", None)
    truth = _lookup_driver_state(driver_id, code, number, truth_maps)
    live = _lookup_driver_state(driver_id, code, number, live_maps)
    evidence = evidence_drivers.get(driver_id) or evidence_drivers.get(code) or {}
    race_inputs = evidence.get("race_inputs") or {}
    practice = evidence.get("practice") or {}
    grid = evidence.get("grid") or {}
    source = live or truth or race_inputs or grid or practice
    position = _safe_int(_first_non_none(live.get("position"), truth.get("position"), race_inputs.get("position"), grid.get("grid_position"))) or index
    lap = _safe_int(_first_non_none(live.get("lap"), truth.get("lap"), race_inputs.get("lap"), truth.get("laps"))) or 0
    compound = _first_non_none(live.get("compound"), truth.get("compound"), race_inputs.get("compound"), race_inputs.get("final_compound"))
    return {
        "driver_id": driver_id,
        "driver_code": code,
        "driver_number": number,
        "team": getattr(driver, "team", None),
        "position": position,
        "lap": lap,
        "gap_to_leader": _first_non_none(live.get("gap_to_leader"), truth.get("gap_to_leader"), race_inputs.get("gap_to_leader")),
        "interval": _first_non_none(live.get("interval"), truth.get("interval"), race_inputs.get("interval")),
        "representative_lap": _first_non_none(
            live.get("representative_lap"),
            truth.get("representative_lap"),
            race_inputs.get("representative_lap"),
            practice.get("representative_lap"),
            practice.get("long_run_lap"),
        ),
        "compound": str(compound).upper() if compound else "MEDIUM",
        "tyre_age": _safe_int(_first_non_none(live.get("tyre_age"), truth.get("tyre_age"), race_inputs.get("tyre_age"))) or 0,
        "pit_stops": _safe_int(_first_non_none(live.get("pit_stops"), truth.get("pit_stops"), race_inputs.get("pit_stops"))) or 0,
        "estimated_progress": _first_non_none(live.get("estimated_progress"), truth.get("estimated_progress"), race_inputs.get("estimated_progress")),
        "x": _first_non_none(live.get("x"), truth.get("x"), race_inputs.get("x")),
        "y": _first_non_none(live.get("y"), truth.get("y"), race_inputs.get("y")),
        "source_mode": source.get("source_mode") or source.get("source") or source_mode,
        "confidence": round(min(float(source.get("confidence") or confidence), _confidence_ceiling(source.get("source_mode") or source_mode)), 4),
    }


def _lookup_driver_state(
    driver_id: str,
    code: str,
    number: Any,
    maps: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    return (
        (maps.get("by_id") or {}).get(driver_id)
        or (maps.get("by_code") or {}).get(code)
        or (maps.get("by_number") or {}).get(str(number))
        or {}
    )


def _first_non_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _apply_live_state_to_initial_state(initial_state: dict[str, Any], live_state: dict[str, Any]) -> dict[str, Any]:
    state = {key: list(value) if isinstance(value, list) else value for key, value in initial_state.items()}
    codes = list(state.get("driver_codes") or [])
    means = list(state.get("driver_mean_pace_s") or [])
    sigmas = list(state.get("driver_pace_sigma_s") or [])
    dnfs = list(state.get("driver_dnf_rate_per_lap") or [])
    compounds = list(state.get("driver_starting_compound") or ["MEDIUM"] * len(codes))
    pit_laps = list(state.get("driver_pit_lap") or [None] * len(codes))
    total_laps = int(state.get("total_laps") or 1)
    current_lap = max(0, int(live_state.get("lap") or 0))
    remaining = max(1, total_laps - current_lap)
    confidence = float(live_state.get("confidence") or 0.0)
    mode = live_state.get("source_mode")
    if mode == "estimated":
        confidence = min(confidence, 0.25)
    by_code = {row.get("driver_code"): row for row in live_state.get("drivers") or []}
    field_size = max(1, len(codes))
    median_pos = (field_size + 1) / 2.0
    for index, code in enumerate(codes):
        row = by_code.get(code) or {}
        position = _safe_int(row.get("position")) or index + 1
        lap_time = _safe_float(row.get("representative_lap"))
        if lap_time and 45.0 <= lap_time <= 140.0:
            means[index] = round((1.0 - confidence * 0.35) * float(means[index]) + confidence * 0.35 * lap_time, 4)
        position_delta = (position - median_pos) / field_size
        means[index] = round(float(means[index]) + confidence * position_delta * 1.4, 4)
        gap = _gap_seconds(row.get("gap_to_leader"))
        if gap is not None:
            means[index] = round(float(means[index]) + min(1.25, gap / max(20.0, remaining) * 0.25) * confidence, 4)
        tyre_age = _safe_int(row.get("tyre_age")) or 0
        compound = str(row.get("compound") or "").upper()
        if compound in {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}:
            compounds[index] = compound
        if compound == "SOFT" and tyre_age > 14:
            means[index] = round(float(means[index]) + min(0.9, (tyre_age - 14) * 0.045) * confidence, 4)
        if _safe_int(row.get("pit_stops")) and pit_laps[index] and current_lap >= int(pit_laps[index]):
            pit_laps[index] = None
        if source_is_reliable(mode):
            sigmas[index] = round(max(0.12, float(sigmas[index]) * (1.0 - min(0.22, confidence * 0.18))), 4)
        else:
            sigmas[index] = round(min(2.0, float(sigmas[index]) * 1.08), 4)
        dnfs[index] = round(max(0.00005, min(0.05, float(dnfs[index]))), 6)
    state["driver_mean_pace_s"] = means
    state["driver_pace_sigma_s"] = sigmas
    state["driver_dnf_rate_per_lap"] = dnfs
    state["driver_starting_compound"] = compounds
    state["driver_pit_lap"] = pit_laps
    state["total_laps"] = remaining
    state["live_lap"] = current_lap
    return state


def _payload_from_result(
    *,
    config: LiveRunnerConfig,
    race_obj: Race,
    sim_result: Any,
    live_state: dict[str, Any],
    artifact_meta: dict[str, Any],
    adjusted_state: dict[str, Any],
) -> dict[str, Any]:
    winner = winner_probabilities(sim_result)
    podium = podium_probabilities(sim_result)
    top5 = top_k_probabilities(sim_result, min(5, len(sim_result.driver_codes)))
    dnf = dnf_probabilities(sim_result)
    fastest = fastest_lap_probabilities(sim_result)
    expected = {
        code: float(sim_result.finish_positions[:, index].mean())
        for index, code in enumerate(sim_result.driver_codes)
    }
    rows = []
    for code in sim_result.driver_codes:
        rows.append({
            "driver_code": code,
            "win_probability": round(winner.get(code, 0.0), 6),
            "podium_probability": round(podium.get(code, 0.0), 6),
            "top5_probability": round(top5.get(code, 0.0), 6),
            "dnf_probability": round(dnf.get(code, 0.0), 6),
            "fastest_lap_probability": round(fastest.get(code, 0.0), 6),
            "expected_finish": round(expected.get(code, 99.0), 3),
        })
    rows.sort(key=lambda row: row["win_probability"], reverse=True)
    top_movers = _top_movers(rows, live_state)
    return {
        "ok": True,
        "model_id": config.model_id,
        "model_version": f"{config.model_id}+live_runner_v1",
        "race": config.race,
        "race_id": _payload_race_id(config),
        "season": race_obj.season,
        "round": race_obj.round,
        "session": config.session,
        "stage": "live" if source_is_reliable(live_state.get("source_mode")) else "estimated",
        "source_mode": live_state.get("source_mode"),
        "confidence": live_state.get("confidence"),
        "confidence_ceiling": live_state.get("confidence_ceiling"),
        "confidence_reason": live_state.get("confidence_reason"),
        "fallback_reason": live_state.get("fallback_reason") or artifact_meta.get("ml_fallback_reason"),
        "lap": live_state.get("lap"),
        "remaining_laps": adjusted_state.get("total_laps"),
        "session_status": live_state.get("session_status"),
        "ml_input_source": artifact_meta.get("ml_input_source") or artifact_meta.get("input_source"),
        "ml_provider_sources": artifact_meta.get("ml_provider_sources") or [],
        "ml_fallback_reason": artifact_meta.get("ml_fallback_reason"),
        "ml_confidence": artifact_meta.get("ml_confidence"),
        "simulator_iterations": artifact_meta.get("simulator_iterations"),
        "artifact_id": artifact_meta.get("artifact_id") or artifact_meta.get("ml_artifact_id"),
        "artifact_version": artifact_meta.get("artifact_version") or artifact_meta.get("ml_artifact_version"),
        "ml_artifact_id": artifact_meta.get("ml_artifact_id") or artifact_meta.get("artifact_id"),
        "ml_artifact_version": artifact_meta.get("ml_artifact_version") or artifact_meta.get("artifact_version"),
        "ml_artifact_readiness": artifact_meta.get("ml_artifact_readiness") or {},
        "trained_artifacts_used": artifact_meta.get("trained_artifacts_used"),
        "ml_model_contract_used": artifact_meta.get("ml_model_contract_used"),
        "ml_model_adapters_used": artifact_meta.get("ml_model_adapters_used") or [],
        "ml_model_fallback_reason": artifact_meta.get("ml_model_fallback_reason"),
        "pace_adapter_source": artifact_meta.get("pace_adapter_source"),
        "dnf_adapter_source": artifact_meta.get("dnf_adapter_source"),
        "rating_adapter_source": artifact_meta.get("rating_adapter_source"),
        "evidence_groups_used": sorted(set((live_state.get("evidence_groups_used") or []) + ["simulator"])),
        "missing_evidence_groups": live_state.get("missing_evidence_groups") or [],
        "probabilities": rows,
        "top_probability_movers": top_movers,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _top_movers(rows: list[dict[str, Any]], live_state: dict[str, Any]) -> list[dict[str, Any]]:
    position_by_code = {row.get("driver_code"): row.get("position") for row in live_state.get("drivers") or []}
    movers = []
    for row in rows[:8]:
        code = row["driver_code"]
        position = _safe_int(position_by_code.get(code))
        if position is None:
            continue
        mover_score = round(row["win_probability"] * (1.0 + max(0, 6 - position) * 0.08), 6)
        movers.append({
            "driver_code": code,
            "position": position,
            "win_probability": row["win_probability"],
            "movement_score": mover_score,
            "reason": f"P{position} live/replay order with {live_state.get('source_mode')} confidence",
        })
    return sorted(movers, key=lambda item: item["movement_score"], reverse=True)[:5]


def _to_outcome_probabilities(payload: dict[str, Any]) -> list[OutcomeProbability]:
    now = datetime.now(timezone.utc)
    records = []
    race_id = str(payload.get("race_id") or payload.get("race"))
    version = str(payload.get("model_version") or payload.get("model_id") or "ml_simulator_v1")
    for row in payload.get("probabilities") or []:
        code = str(row.get("driver_code"))
        for market, key in (
            ("winner", "win_probability"),
            ("podium", "podium_probability"),
            ("top5", "top5_probability"),
            ("fastest_lap", "fastest_lap_probability"),
            ("dnf", "dnf_probability"),
        ):
            records.append(OutcomeProbability(
                domain="f1",
                entity_id=race_id,
                entity_code=code,
                market=market,
                probability=float(row.get(key) or 0.0),
                knowable_as_of=now,
                model_version=version,
            ))
    return records


def _payload_race_id(config: LiveRunnerConfig) -> str:
    text = str(config.race or "").strip()
    if text and not text.lower().startswith("synthetic_") and not text.upper().startswith("SYN-"):
        return text
    return _normalise_synthetic_race_id(text, config.season, config.round_num)


def _publish_records(config: LiveRunnerConfig, records: list[OutcomeProbability], *, publisher: Any | None = None) -> dict[str, Any]:
    if publisher is None:
        publisher = _publisher_from_config(config)
    try:
        publisher.publish_batch(records)
        return {
            "ok": True,
            "mode": "redis" if config.publish and not config.dry_run else "dry_run",
            "records": len(records),
        }
    except Exception as exc:
        return {
            "ok": False,
            "mode": "redis",
            "records": len(records),
            "reason": f"publish_failed:{exc.__class__.__name__}",
        }


def _publisher_from_config(config: LiveRunnerConfig) -> Any:
    if not config.publish or config.dry_run:
        return InMemoryOutcomePublisher()
    parsed = urlparse(config.redis_url or "redis://localhost:6379/0")
    return OutcomePublisher(
        host=parsed.hostname or "localhost",
        port=int(parsed.port or 6379),
        db=int((parsed.path or "/0").lstrip("/") or 0),
    )


def _normalise_synthetic_race_id(race: str, season: int | None = None, round_num: int | None = None) -> str:
    text = (race or "").strip()
    lower = text.lower()
    if lower.startswith("synthetic_"):
        parts = lower.replace("synthetic_", "").replace("r", "").split("_")
        try:
            return f"SYN-{int(parts[0])}-{int(parts[1]):02d}"
        except Exception:
            return "SYN-2026-01"
    if text.startswith("SYN-"):
        return text
    if season and round_num:
        return f"SYN-{int(season)}-{int(round_num):02d}"
    return "SYN-2026-01"


def _synthetic_season_round(race_id: str) -> tuple[int, int]:
    parts = race_id.split("-")
    try:
        return int(parts[1]), int(parts[2])
    except Exception:
        return 2026, 1


def _seed(config: LiveRunnerConfig, live_state: dict[str, Any]) -> int:
    _, round_num = _synthetic_season_round(_normalise_synthetic_race_id(config.race, config.season, config.round_num))
    return 9000 + round_num * 37 + int(live_state.get("lap") or 0)


def _missing_groups(groups: list[str]) -> list[str]:
    required = {"positions", "lap_times", "tyres", "gaps", "race_control"}
    present = set(groups)
    if "replay_laps" in present or "recorded_timing" in present:
        present.update({"positions", "lap_times"})
    return sorted(required - present)


def source_is_reliable(source_mode: Any) -> bool:
    return str(source_mode or "") in {"live", "recent", "recorded", "recorded_confident"}


def _first(row: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row and row.get(key) is not None:
            return row.get(key)
    return None


def _gap_seconds(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("+", "")
    if text in {"", "leader", "interval"}:
        return None
    if "lap" in text:
        return 90.0
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("Value") or value.get("value")
    if isinstance(value, str) and ":" in value:
        try:
            minutes, seconds = value.split(":", 1)
            return int(minutes) * 60.0 + float(seconds)
        except ValueError:
            return None
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def payload_to_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, default=str)
