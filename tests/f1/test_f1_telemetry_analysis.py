import asyncio
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

from sports.f1.ml.common.types import Race, SessionType
from sports.f1.ml.providers.fastf1_telemetry_provider import FastF1TelemetryProvider
from sports.f1.ml.providers.fastf1_provider import FastF1Provider, _derive_longitudinal_accel_g
from sports.f1.ml.providers.openf1_provider import OpenF1TelemetryProvider
from sports.f1.ml.telemetry import (
    TelemetryCache,
    apply_telemetry_adjustments,
    build_telemetry_features,
    build_telemetry_features_from_openf1_session,
    build_telemetry_model_output,
    load_telemetry_artifact_manifest,
    normalize_openf1_trace_points,
    score_telemetry_artifact_heads,
    telemetry_leakage_guard_status,
)
from sports.f1.ml.telemetry.types import TelemetryFeaturePayload, TelemetryFeatureVector, TelemetryModelOutput


class F1TelemetryAnalysisTests(unittest.TestCase):
    def test_openf1_normalizer_joins_car_and_location_rows(self):
        points = normalize_openf1_trace_points(
            race_id="2026-01-BAHRAIN",
            session="race",
            driver_lookup={16: "LEC"},
            car_rows=[
                {
                    "driver_number": 16,
                    "date": "2026-03-01T14:00:00Z",
                    "speed": 285,
                    "throttle": 100,
                    "brake": 0,
                    "rpm": 11200,
                    "n_gear": 8,
                    "drs": 12,
                }
            ],
            location_rows=[
                {"driver_number": 16, "date": "2026-03-01T14:00:00.600Z", "x": 10, "y": 20, "z": 1}
            ],
        )

        self.assertEqual(1, len(points))
        self.assertEqual("LEC", points[0].driver_code)
        self.assertEqual(10, points[0].x)
        self.assertTrue(points[0].drs_active)

    def test_feature_builder_detects_faster_driver(self):
        points = normalize_openf1_trace_points(
            race_id="2026-01-BAHRAIN",
            session="race",
            driver_lookup={16: "LEC", 44: "HAM"},
            car_rows=[
                {"driver_number": 16, "date": "2026-03-01T14:00:00Z", "speed": 290, "throttle": 100, "brake": 0},
                {"driver_number": 16, "date": "2026-03-01T14:00:01Z", "speed": 160, "throttle": 86, "brake": 0},
                {"driver_number": 16, "date": "2026-03-01T14:00:02Z", "speed": 295, "throttle": 100, "brake": 0},
                {"driver_number": 44, "date": "2026-03-01T14:00:00Z", "speed": 282, "throttle": 95, "brake": 0},
                {"driver_number": 44, "date": "2026-03-01T14:00:01Z", "speed": 145, "throttle": 70, "brake": 80},
                {"driver_number": 44, "date": "2026-03-01T14:00:02Z", "speed": 286, "throttle": 94, "brake": 0},
            ],
            location_rows=[],
        )
        payload = build_telemetry_features(
            race_id="2026-01-BAHRAIN",
            session="race",
            source_mode="openf1_historical",
            trace_points=points,
            laps=[
                {"driver_code": "LEC", "lap_time_s": 83.1},
                {"driver_code": "LEC", "lap_time_s": 83.0},
                {"driver_code": "HAM", "lap_time_s": 83.8},
                {"driver_code": "HAM", "lap_time_s": 83.9},
            ],
            intervals=[{"driver_code": "LEC", "interval": 0.8}],
        )

        self.assertTrue(payload.ok)
        self.assertLess(payload.driver_features["LEC"].clean_air_pace_delta_s, 0)
        self.assertGreater(payload.driver_features["LEC"].overtake_pressure, 0)
        self.assertIn("location", payload.missing_groups)

    def test_v0_model_builds_bounded_adjustments(self):
        points = normalize_openf1_trace_points(
            race_id="2026-01-BAHRAIN",
            session="race",
            driver_lookup={16: "LEC", 44: "HAM"},
            car_rows=[
                {"driver_number": 16, "date": "2026-03-01T14:00:00Z", "speed": 300, "throttle": 100},
                {"driver_number": 16, "date": "2026-03-01T14:00:01Z", "speed": 170, "throttle": 92},
                {"driver_number": 44, "date": "2026-03-01T14:00:00Z", "speed": 275, "throttle": 92},
                {"driver_number": 44, "date": "2026-03-01T14:00:01Z", "speed": 130, "throttle": 55, "brake": 100},
            ],
            location_rows=[],
        )
        payload = build_telemetry_features(
            race_id="2026-01-BAHRAIN",
            session="race",
            trace_points=points,
            source_mode="openf1_historical",
            laps=[
                {"driver_code": "LEC", "lap_time_s": 82.0},
                {"driver_code": "HAM", "lap_time_s": 84.0},
            ],
        )
        output = build_telemetry_model_output(payload, stage="live")

        self.assertIn("LEC", output.driver_adjustments)
        self.assertLess(output.driver_adjustments["LEC"]["clean_air_pace_shift_s"], 0)
        self.assertLessEqual(abs(output.driver_adjustments["HAM"]["clean_air_pace_shift_s"]), 1.20)
        self.assertTrue(output.probability_delta_explanations)
        self.assertTrue(output.probability_delta_explanations[0]["factors"])
        self.assertTrue(output.metadata["feature_importance"])
        self.assertEqual("ok", output.metadata["telemetry_leakage_guard_status"]["status"])
        self.assertFalse(output.metadata["learned_artifact_status"]["ok"])

    def test_telemetry_guard_blocks_race_timing_before_live_stage(self):
        payload = build_telemetry_features_from_openf1_session(
            race_id="2026-01-BAHRAIN",
            session="race",
            openf1_session={
                "ok": True,
                "source": "openf1",
                "laps": {
                    "drivers": {
                        "16": {"driver_number": 16, "driver_code": "LEC", "laps": 9, "representative_lap": 90.8},
                    },
                },
                "raw_counts": {"laps": 9},
            },
        )

        guard = telemetry_leakage_guard_status(payload, stage="pre_weekend")
        output = build_telemetry_model_output(payload, stage="pre_weekend")

        self.assertEqual("blocked", guard["status"])
        self.assertIn("live_timing", guard["blocked_feature_groups"])
        self.assertEqual("blocked", output.metadata["telemetry_leakage_guard_status"]["status"])

    def test_telemetry_artifact_manifest_reports_missing_and_present_heads(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text(json.dumps({
                "artifact_id": "telemetry-fixture",
                "artifact_version": "v1",
                "heads": {
                    "pace_delta": {"path": "pace.pkl"},
                    "pace_quantile": {"path": "quantile.pkl"},
                    "dnf_hazard": {"path": "dnf.pkl"},
                    "overtake": {"path": "overtake.pkl"},
                    "pit_value": {"path": "pit.pkl"},
                },
            }), encoding="utf-8")

            status = load_telemetry_artifact_manifest(tmp)

        self.assertTrue(status["ok"])
        self.assertEqual("telemetry-fixture", status["artifact_id"])
        self.assertEqual([], status["missing_heads"])

    def test_telemetry_artifact_heads_score_and_drive_adjustments(self):
        payload = TelemetryFeaturePayload(
            race_id="2026-01-BAHRAIN",
            session="fp1",
            source_mode="fixture",
            confidence=0.8,
            driver_features={
                "LEC": TelemetryFeatureVector(
                    race_id="2026-01-BAHRAIN",
                    session="fp1",
                    driver_code="LEC",
                    source="fixture",
                    clean_air_pace_delta_s=0.15,
                    confidence=0.8,
                    samples=12,
                )
            },
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text(json.dumps({
                "artifact_id": "telemetry-linear-fixture",
                "heads": {
                    "pace_delta": {"intercept": -0.4, "coefficients": {}, "clamp": [-1.2, 1.2]},
                    "pace_quantile": {"intercept": 0.9, "coefficients": {}, "clamp": [0.78, 1.25]},
                    "dnf_hazard": {"intercept": 1.2, "coefficients": {}, "clamp": [0.75, 1.5]},
                    "overtake": {"intercept": 0.04, "coefficients": {}, "clamp": [-0.07, 0.10]},
                    "pit_value": {"intercept": 0.2, "coefficients": {}, "clamp": [-0.25, 0.55]},
                },
            }), encoding="utf-8")
            status = load_telemetry_artifact_manifest(tmp)
            scored = score_telemetry_artifact_heads(status, payload.driver_features["LEC"])
            output = build_telemetry_model_output(payload, stage="live", artifact_path=tmp)

        self.assertEqual(-0.4, scored["pace_delta"])
        self.assertTrue(output.metadata["learned_artifacts_used"])
        self.assertEqual("telemetry-linear-fixture", output.metadata["learned_artifact_status"]["artifact_id"])
        self.assertAlmostEqual(-0.32, output.driver_adjustments["LEC"]["clean_air_pace_shift_s"])
        self.assertAlmostEqual(0.9, output.driver_adjustments["LEC"]["pace_sigma_multiplier"])
        self.assertAlmostEqual(1.2, output.driver_adjustments["LEC"]["dnf_hazard_multiplier"])

    def test_adapter_applies_adjustments_without_mutating_input(self):
        initial = {
            "driver_codes": ["LEC", "HAM"],
            "driver_mean_pace_s": [83.5, 83.7],
            "driver_pace_sigma_s": [0.4, 0.4],
            "driver_dnf_rate_per_lap": [0.001, 0.001],
        }
        output = TelemetryModelOutput(
            race_id="2026-01-BAHRAIN",
            session="race",
            confidence=0.7,
            source_mode="fixture",
            driver_adjustments={
                "LEC": {
                    "clean_air_pace_shift_s": -0.2,
                    "pace_sigma_multiplier": 0.9,
                    "dnf_hazard_multiplier": 1.1,
                    "tire_deg_slope_delta": 0.01,
                    "pit_window_value_s": 0.05,
                }
            },
        )

        adjusted, meta = apply_telemetry_adjustments(initial, output)

        self.assertTrue(meta["telemetry_model_used"])
        self.assertEqual([83.5, 83.7], initial["driver_mean_pace_s"])
        self.assertAlmostEqual(83.3, adjusted["driver_mean_pace_s"][0])
        self.assertAlmostEqual(0.36, adjusted["driver_pace_sigma_s"][0])
        self.assertEqual(["LEC"], adjusted["telemetry_adjustments_applied"])

    def test_adapter_ignores_low_confidence_output(self):
        initial = {
            "driver_codes": ["LEC"],
            "driver_mean_pace_s": [83.5],
            "driver_pace_sigma_s": [0.4],
            "driver_dnf_rate_per_lap": [0.001],
        }
        output = TelemetryModelOutput(
            race_id="2026-01-BAHRAIN",
            session="race",
            confidence=0.1,
            source_mode="estimated",
            driver_adjustments={"LEC": {"clean_air_pace_shift_s": -1.0}},
        )

        adjusted, meta = apply_telemetry_adjustments(initial, output)

        self.assertFalse(meta["telemetry_model_used"])
        self.assertEqual("telemetry_confidence_below_threshold", meta["telemetry_fallback_reason"])
        self.assertEqual(initial, adjusted)

    def test_openf1_summary_builder_produces_driver_features(self):
        payload = build_telemetry_features_from_openf1_session(
            race_id="2026-01-BAHRAIN",
            session="fp1",
            openf1_session={
                "ok": True,
                "source": "openf1",
                "laps": {
                    "drivers": {
                        "16": {
                            "driver_number": 16,
                            "driver_code": "LEC",
                            "laps": 9,
                            "representative_lap": 90.8,
                            "median_lap": 91.2,
                            "lap_time_stddev": 0.4,
                            "pace_stability": 0.86,
                        },
                        "44": {
                            "driver_number": 44,
                            "driver_code": "HAM",
                            "laps": 9,
                            "representative_lap": 91.6,
                            "median_lap": 91.9,
                            "lap_time_stddev": 0.9,
                            "pace_stability": 0.70,
                        },
                    },
                    "source": "openf1_laps",
                },
                "intervals": {
                    "drivers": {
                        "16": {"driver_number": 16, "driver_code": "LEC", "interval": 0.7},
                    },
                    "source": "openf1_intervals",
                },
                "stints": {
                    "drivers": {
                        "16": {"driver_number": 16, "driver_code": "LEC", "final_stint_laps": 18},
                    },
                    "source": "openf1_stints",
                },
                "raw_counts": {"laps": 18, "intervals": 2, "stints": 2},
            },
        )

        self.assertTrue(payload.ok)
        self.assertEqual("openf1_historical", payload.source_mode)
        self.assertIn("LEC", payload.driver_features)
        self.assertLess(payload.driver_features["LEC"].clean_air_pace_delta_s, 0)
        self.assertGreater(payload.driver_features["LEC"].traffic_penalty_s, 0)

    def test_telemetry_cache_roundtrips_snapshot_manifest(self):
        with TemporaryDirectory() as tmp:
            cache = TelemetryCache(tmp)
            written = cache.write_snapshot(
                source="openf1",
                season=2026,
                round_num=1,
                session="fp1",
                payload={
                    "raw_counts": {"car_data": 2},
                    "car_data": [{"driver_number": 16, "speed": 300}],
                    "trace_points": [],
                },
            )

            cached = cache.read_snapshot(source="openf1", season=2026, round_num=1, session="fp1")

            self.assertFalse(written["cache"]["hit"])
            self.assertTrue(cached["cache"]["hit"])
            self.assertEqual("f1_telemetry_cache_v1", cached["manifest"]["schema_version"])
            self.assertEqual(2, cached["raw_counts"]["car_data"])

    def test_openf1_provider_builds_raw_features_and_uses_cache(self):
        async def run_case():
            with TemporaryDirectory() as tmp:
                client = _RawOpenF1()
                provider = OpenF1TelemetryProvider(client, cache=TelemetryCache(tmp))
                race = SimpleNamespace()
                drivers = [
                    SimpleNamespace(permanent_number=16, code="LEC"),
                    SimpleNamespace(permanent_number=44, code="HAM"),
                ]

                payload, snapshot = await provider.telemetry_features(
                    race=race,
                    race_id="2026-01-BAHRAIN",
                    season=2026,
                    round_num=1,
                    session="fp1",
                    drivers=drivers,
                    live=False,
                )
                cached_payload, cached_snapshot = await provider.telemetry_features(
                    race=race,
                    race_id="2026-01-BAHRAIN",
                    season=2026,
                    round_num=1,
                    session="fp1",
                    drivers=drivers,
                    live=False,
                )
                return client, payload, snapshot, cached_payload, cached_snapshot

        client, payload, snapshot, cached_payload, cached_snapshot = asyncio.run(run_case())

        self.assertTrue(payload.ok)
        self.assertEqual("openf1_historical_raw", payload.source_mode)
        self.assertTrue(payload.data_quality["raw_trace"])
        self.assertIn("LEC", payload.driver_features)
        self.assertEqual(1, client.session_feature_calls)
        self.assertFalse(snapshot["cache"]["hit"])
        self.assertTrue(cached_snapshot["cache"]["hit"])
        self.assertTrue(cached_payload.ok)

    def test_fastf1_provider_derives_acceleration(self):
        previous = {"Speed": 100, "Date": "2026-03-01T14:00:00+00:00"}
        current = {"Speed": 136, "Date": "2026-03-01T14:00:01+00:00"}

        accel = _derive_longitudinal_accel_g(previous, current)

        self.assertAlmostEqual(1.0197, accel, places=3)

    def test_fastf1_provider_trace_points_join_position_and_quality_flags(self):
        with TemporaryDirectory() as tmp:
            provider = FastF1Provider(cache_dir=tmp)
            provider._fastf1 = _FakeFastF1()
            race = Race(
                season=2026,
                round=1,
                track_code="BAHRAIN",
                name="Bahrain GP",
                scheduled_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
            )

            points = provider.trace_points(race, SessionType.FP1)
            diagnostics = provider.session_diagnostics(race, SessionType.FP1)

        self.assertEqual(2, len(points))
        self.assertEqual("LEC", points[0].driver_code)
        self.assertEqual(10, points[0].x)
        self.assertIn("pit_out_lap", points[0].data_quality_flags)
        self.assertGreater(points[1].accel_long_g, 0)
        self.assertTrue(diagnostics["data_quality"]["track_status_available"])
        self.assertTrue(diagnostics["data_quality"]["race_control_available"])

    def test_fastf1_telemetry_bridge_builds_features_and_uses_cache(self):
        with TemporaryDirectory() as tmp:
            provider = FastF1Provider(cache_dir=tmp)
            provider._fastf1 = _FakeFastF1()
            bridge = FastF1TelemetryProvider(provider, cache=TelemetryCache(tmp))
            race = Race(
                season=2026,
                round=1,
                track_code="BAHRAIN",
                name="Bahrain GP",
                scheduled_start=datetime(2026, 3, 1, tzinfo=timezone.utc),
            )

            payload, snapshot = bridge.telemetry_features(
                race=race,
                season=2026,
                round_num=1,
                session="fp1",
            )
            cached_payload, cached_snapshot = bridge.telemetry_features(
                race=race,
                season=2026,
                round_num=1,
                session="fp1",
            )

        self.assertTrue(payload.ok)
        self.assertEqual("fastf1_historical_raw", payload.source_mode)
        self.assertIn("LEC", payload.driver_features)
        self.assertFalse(snapshot["cache"]["hit"])
        self.assertTrue(cached_snapshot["cache"]["hit"])
        self.assertTrue(cached_payload.ok)

class _RawOpenF1:
    def __init__(self):
        self.session_feature_calls = 0

    async def get_session_features(self, race, session="race", drivers=None, live=False):
        self.session_feature_calls += 1
        return {
            "ok": True,
            "source": "openf1",
            "session": session,
            "session_key": 101,
            "laps": {
                "drivers": {
                    "16": {"driver_number": 16, "driver_code": "LEC", "representative_lap": 83.0, "laps": 2},
                    "44": {"driver_number": 44, "driver_code": "HAM", "representative_lap": 84.0, "laps": 2},
                }
            },
            "intervals": {"drivers": {"16": {"driver_number": 16, "driver_code": "LEC", "interval": 0.8}}},
            "stints": {"drivers": {}},
            "pits": {"drivers": {}},
            "raw_counts": {"laps": 4, "intervals": 1},
        }

    async def get_car_data(self, session_key, **filters):
        return [
            {"driver_number": 16, "date": "2026-03-01T14:00:00Z", "speed": 300, "throttle": 100, "brake": 0},
            {"driver_number": 16, "date": "2026-03-01T14:00:01Z", "speed": 160, "throttle": 86, "brake": 0},
            {"driver_number": 44, "date": "2026-03-01T14:00:00Z", "speed": 280, "throttle": 95, "brake": 0},
            {"driver_number": 44, "date": "2026-03-01T14:00:01Z", "speed": 145, "throttle": 70, "brake": 80},
        ]

    async def get_location(self, session_key, **filters):
        return [
            {"driver_number": 16, "date": "2026-03-01T14:00:00.300Z", "x": 10, "y": 20},
            {"driver_number": 44, "date": "2026-03-01T14:00:00.300Z", "x": 12, "y": 22},
        ]


class _FakeFrame:
    def __init__(self, rows):
        self.rows = rows
        self.empty = not bool(rows)

    def iterrows(self):
        for index, row in enumerate(self.rows):
            yield index, row

    def add_distance(self):
        return self

    def pick_drivers(self, driver_code):
        return _FakeFrame([row for row in self.rows if row.get("Driver") == driver_code])


class _FakeLap(dict):
    def get_car_data(self):
        return _FakeFrame([
            {"Date": "2026-03-01T14:00:00+00:00", "Speed": 100, "Distance": 0, "Throttle": 50, "Brake": False, "RPM": 9000, "nGear": 4, "DRS": 0},
            {"Date": "2026-03-01T14:00:01+00:00", "Speed": 136, "Distance": 35, "Throttle": 90, "Brake": False, "RPM": 10500, "nGear": 5, "DRS": 12},
        ])

    def get_pos_data(self):
        return _FakeFrame([
            {"Date": "2026-03-01T14:00:00.200+00:00", "X": 10, "Y": 20, "Z": 1},
            {"Date": "2026-03-01T14:00:01.100+00:00", "X": 18, "Y": 24, "Z": 1},
        ])


class _FakeFastF1Session:
    drivers = ["LEC"]
    date = datetime(2026, 3, 1, tzinfo=timezone.utc)

    def __init__(self):
        self.laps = _FakeFrame([
            _FakeLap({
                "Driver": "LEC",
                "LapNumber": 3,
                "TrackStatus": "1",
                "Deleted": False,
                "PitInTime": None,
                "PitOutTime": "2026-03-01T14:00:00+00:00",
                "LapTime": 92.1,
            })
        ])
        self.track_status = _FakeFrame([{"Time": "0:01:00", "Status": "1", "Message": "AllClear"}])
        self.race_control_messages = _FakeFrame([{"Time": "0:01:10", "Category": "Flag", "Message": "Green"}])

    def load(self, **kwargs):
        return None


class _FakeFastF1:
    def get_session(self, *args, **kwargs):
        return _FakeFastF1Session()


if __name__ == "__main__":
    unittest.main()
