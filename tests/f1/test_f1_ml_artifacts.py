import json
import unittest
from datetime import datetime, timezone
from tempfile import TemporaryDirectory
from pathlib import Path

try:
    from typer.testing import CliRunner
    from sports.f1.ml.cli import app
except ModuleNotFoundError:
    CliRunner = None
    app = None

from sports.f1.ml.artifacts.builder import build_synthetic_artifact_bundle
from sports.f1.ml.artifacts.builder import build_fastf1_artifact_bundle
from sports.f1.ml.artifacts.loader import artifact_to_ml_trained_inputs, artifact_to_simulator_model_bundle, load_ml_trained_inputs
from sports.f1.ml.artifacts.schema import F1MLArtifactBundle, F1MLDriverArtifact
from sports.f1.ml.artifacts.store import artifact_readiness, load_artifact_bundle, load_artifact_readiness, save_artifact_bundle, validate_artifact_bundle
from sports.f1.ml.common.types import Lap, Race as MLSimRace, SessionType, TireCompound
from sports.f1.ml.markets.mapper import dnf_probabilities, winner_probabilities
from sports.f1.ml.simulator.model_contract import PaceDistribution, SimulatorModelBundle
from sports.f1.ml.simulator.race_sim import SimConfig, simulate_race
from sports.f1.ml.telemetry import (
    build_telemetry_artifact_manifest,
    export_telemetry_training_rows,
    load_telemetry_artifact_manifest,
    load_telemetry_label_rows,
    load_telemetry_training_rows,
    load_telemetry_training_payloads,
    save_telemetry_artifact_manifest,
)
from sports.f1.ml.telemetry.types import TelemetryFeaturePayload, TelemetryFeatureVector
from sports.f1.models.f1 import Constructor, Driver, Race
from sports.f1.predictor.models.baseline import BaselineRaceModel
from sports.f1.predictor.models.registry import F1ModelRegistry

try:
    from sports.f1.predictor.service import F1PredictionService
except ModuleNotFoundError:
    F1PredictionService = None


class F1MLArtifactTests(unittest.TestCase):
    def setUp(self):
        self.drivers = [
            Driver(id="antonelli", number=12, code="ANT", first_name="Kimi", last_name="Antonelli", nationality="Italian", team="Mercedes", points=75, position=1),
            Driver(id="russell", number=63, code="RUS", first_name="George", last_name="Russell", nationality="British", team="Mercedes", points=62, position=2),
        ]
        self.constructors = [Constructor(id="mercedes", name="Mercedes", nationality="German", points=137, position=1)]
        self.race = Race(round=4, name="Miami Grand Prix", circuit="Miami", country="USA", date=datetime.now(timezone.utc))
        self.features = {
            "completed_races": 3,
            "total_races": 22,
            "ml_simulator_iterations": 300,
            "drivers": {
                "antonelli": {"form_score": 0.82, "reliability_score": 0.90},
                "russell": {"form_score": 0.70, "reliability_score": 0.86},
            },
            "constructors": {"mercedes": {"team_score": 0.84}},
        }

    def test_artifact_save_load_round_trip(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            bundle = build_synthetic_artifact_bundle()
            save_artifact_bundle(path, bundle)
            loaded = load_artifact_bundle(path)
            validation = validate_artifact_bundle(loaded)

        self.assertTrue(validation.ok)
        self.assertEqual(bundle.artifact_id, loaded.artifact_id)
        self.assertTrue(loaded.feature_columns)
        self.assertTrue(loaded.drivers)

    def test_artifact_readiness_separates_serving_from_live_trading(self):
        bundle = build_synthetic_artifact_bundle()

        readiness = artifact_readiness(bundle)

        self.assertTrue(readiness["serving_ready"])
        self.assertFalse(readiness["live_trading_ready"])
        self.assertIn("out_of_sample_gate_missing", readiness["promotion_blockers"])
        self.assertGreater(readiness["coverage"]["pace_coverage"], 0.0)

    def test_artifact_readiness_passes_live_gate_when_embedded(self):
        bundle = build_synthetic_artifact_bundle(metrics={
            "out_of_sample_gate": {
                "passed": True,
                "status": "passed",
                "candidate_model_id": "ml_simulator_v1",
                "baseline_model_id": "uniform",
            }
        })

        readiness = artifact_readiness(bundle)

        self.assertTrue(readiness["serving_ready"])
        self.assertTrue(readiness["live_trading_ready"])
        self.assertEqual([], readiness["promotion_blockers"])

    def test_artifact_readiness_blocks_live_when_quality_underperforms(self):
        bundle = build_synthetic_artifact_bundle(metrics={
            "out_of_sample_gate": {
                "passed": True,
                "status": "passed",
                "candidate_model_id": "ml_simulator_v1",
                "baseline_model_id": "uniform",
            },
            "model_quality": {
                "pace_beats_baseline": False,
                "dnf_beats_baseline": True,
                "overtake_beats_baseline": False,
            },
        })

        readiness = artifact_readiness(bundle)

        self.assertTrue(readiness["serving_ready"])
        self.assertFalse(readiness["live_trading_ready"])
        self.assertIn("pace_model_underperformed_baseline", readiness["promotion_blockers"])
        self.assertIn("overtake_model_underperformed_baseline", readiness["promotion_blockers"])

    def test_artifact_readiness_reports_passing_calibration_status(self):
        bundle = build_synthetic_artifact_bundle(metrics={
            "out_of_sample_gate": {
                "passed": True,
                "status": "passed",
                "candidate_model_id": "ml_simulator_v1",
                "baseline_model_id": "uniform",
            },
            "calibration": {
                "method": "isotonic",
                "baseline_brier": 0.210,
                "calibrated_brier": 0.184,
                "sample_count": 48,
            },
        })

        readiness = artifact_readiness(bundle)

        self.assertTrue(readiness["live_trading_ready"])
        self.assertEqual("passed", readiness["calibration_status"]["status"])
        self.assertAlmostEqual(0.026, readiness["calibration_status"]["brier_improvement"])
        self.assertEqual([], readiness["promotion_blockers"])

    def test_artifact_readiness_blocks_live_when_calibration_underperforms(self):
        bundle = build_synthetic_artifact_bundle(metrics={
            "out_of_sample_gate": {
                "passed": True,
                "status": "passed",
                "candidate_model_id": "ml_simulator_v1",
                "baseline_model_id": "uniform",
            },
            "calibration": {
                "method": "isotonic",
                "heldout_brier_improvement": -0.004,
                "sample_count": 36,
            },
        })

        readiness = artifact_readiness(bundle)

        self.assertTrue(readiness["serving_ready"])
        self.assertFalse(readiness["live_trading_ready"])
        self.assertEqual("blocked", readiness["calibration_status"]["status"])
        self.assertIn("calibration_underperformed_baseline", readiness["promotion_blockers"])

    def test_load_artifact_readiness_reports_missing_artifact(self):
        readiness = load_artifact_readiness("missing-artifact.json")

        self.assertFalse(readiness["serving_ready"])
        self.assertEqual("artifact_load_failed", readiness["reason"])

    def test_invalid_artifact_returns_fallback_metadata(self):
        result = load_ml_trained_inputs("missing-artifact.json")

        self.assertEqual({}, result["drivers"])
        self.assertEqual("artifact_load_failed", result["artifact_load_error"])

    def test_stale_artifact_returns_fallback_metadata(self):
        stale = self._two_driver_bundle()
        stale.source_metadata["stale"] = True
        result = artifact_to_ml_trained_inputs(stale)

        self.assertEqual({}, result["drivers"])
        self.assertEqual("stale_artifact", result["artifact_load_error"])

    def test_ml_simulator_uses_artifact_path_and_exposes_metadata(self):
        if F1PredictionService is None:
            self.skipTest("httpx/API prediction stack is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            save_artifact_bundle(path, self._two_driver_bundle())
            service = F1PredictionService(model_id="ml_simulator_v1")
            service.load(self.drivers, self.constructors, {**self.features, "ml_artifact_path": str(path)}, sentiment={})

            prediction = service.predict_race(self.race)

        self.assertEqual("trained_artifacts", prediction.ml_input_source)
        self.assertEqual("two-driver-test-artifact", prediction.ml_artifact_id)
        self.assertTrue(prediction.trained_artifacts_used)
        self.assertTrue(prediction.ml_model_contract_used)
        self.assertTrue(prediction.ml_artifact_readiness["serving_ready"])
        self.assertFalse(prediction.ml_artifact_readiness["live_trading_ready"])
        self.assertIn("pace", prediction.ml_model_adapters_used)
        self.assertEqual("artifact_static_dnf", prediction.dnf_adapter_source)
        self.assertIn("artifact_bundle", prediction.ml_provider_sources)
        self.assertGreater(prediction.driver_predictions["russell"].win_prob, prediction.driver_predictions["antonelli"].win_prob)

    def test_artifact_dnf_input_changes_dnf_probability(self):
        if F1PredictionService is None:
            self.skipTest("httpx/API prediction stack is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            save_artifact_bundle(path, self._two_driver_bundle())
            service = F1PredictionService(model_id="ml_simulator_v1")
            service.load(self.drivers, self.constructors, {**self.features, "ml_artifact_path": str(path)}, sentiment={})

            prediction = service.predict_race(self.race)

        self.assertGreater(prediction.driver_predictions["russell"].dnf_prob, prediction.driver_predictions["antonelli"].dnf_prob)

    def test_ml_simulator_uses_artifact_bundle_object(self):
        if F1PredictionService is None:
            self.skipTest("httpx/API prediction stack is not installed in this test runtime")
        service = F1PredictionService(model_id="ml_simulator_v1")
        service.load(self.drivers, self.constructors, {**self.features, "ml_artifact_bundle": self._two_driver_bundle()}, sentiment={})

        prediction = service.predict_race(self.race)

        self.assertEqual("trained_artifacts", prediction.ml_input_source)
        self.assertEqual("two-driver-test-artifact", prediction.ml_artifact_id)
        self.assertTrue(prediction.ml_model_contract_used)

    def test_missing_artifact_path_falls_back_without_crashing(self):
        if F1PredictionService is None:
            self.skipTest("httpx/API prediction stack is not installed in this test runtime")
        service = F1PredictionService(model_id="ml_simulator_v1")
        service.load(self.drivers, self.constructors, {**self.features, "ml_artifact_path": "missing.json"}, sentiment={})

        prediction = service.predict_race(self.race)

        self.assertEqual("evidence_fallback", prediction.ml_input_source)
        self.assertEqual("artifact_load_failed", prediction.ml_fallback_reason)
        self.assertFalse(prediction.trained_artifacts_used)
        self.assertFalse(prediction.ml_model_contract_used)
        self.assertEqual("artifact_load_failed", prediction.ml_model_fallback_reason)

    def test_artifact_to_simulator_model_bundle_builds_adapters(self):
        bundle = artifact_to_simulator_model_bundle(self._two_driver_bundle())

        self.assertTrue(bundle.source_summary()["ml_model_contract_used"])
        self.assertIn("pace", bundle.adapters_used())
        self.assertIn("dnf", bundle.adapters_used())

    def test_invalid_artifact_model_payload_falls_back_visibly(self):
        stale = self._two_driver_bundle()
        stale.source_metadata["stale"] = True
        bundle = artifact_to_simulator_model_bundle(stale)

        self.assertFalse(bundle.source_summary()["ml_model_contract_used"])
        self.assertEqual("stale_artifact", bundle.fallback_reason)

    def test_empty_models_preserves_raw_simulator_behavior(self):
        race = MLSimRace(season=2026, round=1, track_code="SYN", name="Synthetic", scheduled_start=datetime.now(timezone.utc))
        initial_state = self._tiny_initial_state()
        config = SimConfig(n_iterations=120, seed=7)

        raw = simulate_race(race, initial_state, models={}, config=config)
        inert = simulate_race(race, initial_state, models={"unused": object()}, config=config)

        self.assertTrue((raw.finish_positions == inert.finish_positions).all())
        self.assertTrue((raw.dnf_mask == inert.dnf_mask).all())
        self.assertFalse(inert.metadata.get("ml_model_contract_used"))

    def test_fake_pace_adapter_changes_win_probability(self):
        race = MLSimRace(season=2026, round=1, track_code="SYN", name="Synthetic", scheduled_start=datetime.now(timezone.utc))
        initial_state = self._tiny_initial_state()
        config = SimConfig(n_iterations=400, seed=11)

        raw = winner_probabilities(simulate_race(race, initial_state, models={}, config=config))
        adapted = winner_probabilities(simulate_race(
            race,
            initial_state,
            models={"simulator_model_bundle": SimulatorModelBundle(pace_adapter=_FakePaceAdapter())},
            config=config,
        ))

        self.assertGreater(adapted["BBB"], raw["BBB"] + 0.25)

    def test_fake_dnf_adapter_changes_dnf_probability(self):
        race = MLSimRace(season=2026, round=1, track_code="SYN", name="Synthetic", scheduled_start=datetime.now(timezone.utc))
        initial_state = self._tiny_initial_state()
        initial_state["total_laps"] = 25
        config = SimConfig(n_iterations=500, seed=13)

        adapted = dnf_probabilities(simulate_race(
            race,
            initial_state,
            models={"simulator_model_bundle": SimulatorModelBundle(dnf_adapter=_FakeDNFAdapter())},
            config=config,
        ))

        self.assertGreater(adapted["AAA"], adapted["BBB"] + 0.25)

    def test_adapter_exception_falls_back_safely(self):
        race = MLSimRace(season=2026, round=1, track_code="SYN", name="Synthetic", scheduled_start=datetime.now(timezone.utc))
        initial_state = self._tiny_initial_state()
        config = SimConfig(n_iterations=120, seed=17)

        raw = simulate_race(race, initial_state, models={}, config=config)
        broken = simulate_race(
            race,
            initial_state,
            models={"simulator_model_bundle": SimulatorModelBundle(pace_adapter=_BrokenPaceAdapter())},
            config=config,
        )

        self.assertTrue((raw.finish_positions == broken.finish_positions).all())
        self.assertIn("pace", broken.metadata.get("model_contract_fallback_counts") or {})
        self.assertTrue(broken.metadata.get("model_contract_errors"))

    def test_production_model_remains_baseline_default(self):
        self.assertIsInstance(F1ModelRegistry()._model, BaselineRaceModel)
        self.assertEqual("production_v1", F1ModelRegistry().model_id)

    def test_cli_predict_no_models(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        result = CliRunner().invoke(app, ["predict", "--provider", "synthetic", "--no-models", "--n-iterations", "50"])

        self.assertEqual(0, result.exit_code, result.output)
        self.assertIn("artifact mode: no-models", result.output)

    def test_cli_predict_with_artifact_path(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            save_artifact_bundle(path, build_synthetic_artifact_bundle())

            result = CliRunner().invoke(app, ["predict", "--provider", "synthetic", "--artifact-path", str(path), "--n-iterations", "50"])

        self.assertEqual(0, result.exit_code, result.output)
        self.assertIn("artifact mode:", result.output)
        self.assertIn("applied rows=", result.output)
        self.assertIn("model contract: used", result.output)
        self.assertIn("adapters=pace,dnf,rating", result.output)

    def test_cli_train_artifacts_writes_bundle(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trained.json"
            result = CliRunner().invoke(app, ["train-artifacts", "--output", str(path)])
            loaded = load_artifact_bundle(path)

        self.assertEqual(0, result.exit_code, result.output)
        self.assertEqual("ml_simulator_v1", loaded.model_id)
        self.assertTrue(loaded.drivers)

    def test_cli_artifact_status_reports_readiness(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "trained.json"
            save_artifact_bundle(path, build_synthetic_artifact_bundle())
            result = CliRunner().invoke(app, ["artifact-status", "--path", str(path)])

        self.assertEqual(0, result.exit_code, result.output)
        self.assertIn("serving: ready", result.output)
        self.assertIn("live: blocked", result.output)
        self.assertIn("calibration: missing", result.output)

    def test_telemetry_artifact_builder_writes_complete_manifest(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            manifest = build_telemetry_artifact_manifest(artifact_id="telemetry-test")
            result = save_telemetry_artifact_manifest(path, manifest)
            loaded = load_telemetry_artifact_manifest(path)

        self.assertTrue(result["ok"])
        self.assertTrue(loaded["ok"])
        self.assertEqual("telemetry-test", loaded["artifact_id"])
        self.assertEqual([], loaded["missing_heads"])
        self.assertEqual({"pace_delta", "pace_quantile", "dnf_hazard", "overtake", "pit_value"}, set(loaded["trained_heads"]))

    def test_telemetry_artifact_builder_fits_supervised_head_rows(self):
        rows = [
            {"features": {"clean_air_pace_delta_s": 0.0}, "targets": {"pace_delta": 0.10}},
            {"features": {"clean_air_pace_delta_s": 1.0}, "targets": {"pace_delta": 0.68}},
            {"features": {"clean_air_pace_delta_s": 2.0}, "targets": {"pace_delta": 1.20}},
        ]
        manifest = build_telemetry_artifact_manifest(artifact_id="telemetry-supervised-test", training_rows=rows)
        pace = manifest["heads"]["pace_delta"]

        self.assertEqual("supervised_linear", manifest["calibration"]["kind"])
        self.assertEqual(["pace_delta"], manifest["metrics"]["supervised_heads_trained"])
        self.assertEqual("supervised_ridge_linear", pace["calibration"]["kind"])
        self.assertGreater(pace["coefficients"]["clean_air_pace_delta_s"], 0.3)

    def test_cli_train_telemetry_artifacts_writes_manifest(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "telemetry"
            result = CliRunner().invoke(app, ["train-telemetry-artifacts", "--output", str(path), "--artifact-id", "telemetry-cli-test"])
            loaded = load_telemetry_artifact_manifest(path)

        self.assertEqual(0, result.exit_code, result.output)
        self.assertTrue(loaded["ok"])
        self.assertEqual("telemetry-cli-test", loaded["artifact_id"])
        self.assertIn("telemetry artifact written:", result.output)

    def test_cli_train_telemetry_artifacts_accepts_supervised_rows(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            row_path = Path(tmp) / "rows.json"
            row_path.write_text(json.dumps({
                "training_rows": [
                    {"features": {"clean_air_pace_delta_s": 0.0}, "targets": {"pace_delta": 0.10}},
                    {"features": {"clean_air_pace_delta_s": 1.0}, "targets": {"pace_delta": 0.68}},
                    {"features": {"clean_air_pace_delta_s": 2.0}, "targets": {"pace_delta": 1.20}},
                ]
            }), encoding="utf-8")
            output_path = Path(tmp) / "telemetry"
            result = CliRunner().invoke(app, [
                "train-telemetry-artifacts",
                "--output", str(output_path),
                "--artifact-id", "telemetry-cli-supervised",
                "--training-row-path", str(row_path),
            ])
            loaded = load_telemetry_artifact_manifest(output_path)

        self.assertEqual(0, result.exit_code, result.output)
        self.assertTrue(loaded["ok"])
        self.assertEqual("supervised_ridge_linear", loaded["head_definitions"]["pace_delta"]["calibration"]["kind"])
        self.assertIn("supervised heads: pace_delta", result.output)

    def test_cli_train_telemetry_artifacts_can_join_features_and_labels(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            feature_path = Path(tmp) / "features.json"
            label_path = Path(tmp) / "labels.json"
            output_path = Path(tmp) / "telemetry"
            feature_path.write_text(json.dumps([
                _telemetry_payload_fixture("LEC", 0.0).model_dump(mode="json"),
                _telemetry_payload_fixture("VER", 1.0).model_dump(mode="json"),
                _telemetry_payload_fixture("HAM", 2.0).model_dump(mode="json"),
            ]), encoding="utf-8")
            label_path.write_text(json.dumps({
                "label_rows": [
                    {"race_id": "2026-01-TEST", "session": "fp1", "driver_code": "LEC", "pace_delta": 0.10},
                    {"race_id": "2026-01-TEST", "session": "fp1", "driver_code": "VER", "pace_delta": 0.68},
                    {"race_id": "2026-01-TEST", "session": "fp1", "driver_code": "HAM", "pace_delta": 1.20},
                ]
            }), encoding="utf-8")

            result = CliRunner().invoke(app, [
                "train-telemetry-artifacts",
                "--output", str(output_path),
                "--artifact-id", "telemetry-cli-joined",
                "--feature-payload-path", str(feature_path),
                "--label-path", str(label_path),
            ])
            loaded = load_telemetry_artifact_manifest(output_path)

        self.assertEqual(0, result.exit_code, result.output)
        self.assertTrue(loaded["ok"])
        self.assertEqual("supervised_ridge_linear", loaded["head_definitions"]["pace_delta"]["calibration"]["kind"])
        self.assertIn("joined labels: 3 rows: 3", result.output)

    def test_export_telemetry_training_rows_joins_features_and_labels(self):
        payload = _telemetry_payload_fixture()
        rows = export_telemetry_training_rows(
            [payload],
            [{
                "race_id": "2026-01-TEST",
                "session": "fp1",
                "driver_code": "LEC",
                "targets": {"pace_delta": -0.18, "overtake": 0.04},
            }],
        )

        self.assertEqual(1, len(rows))
        self.assertEqual("LEC", rows[0]["driver_code"])
        self.assertAlmostEqual(0.25, rows[0]["features"]["clean_air_pace_delta_s"])
        self.assertEqual({"pace_delta": -0.18, "overtake": 0.04}, rows[0]["targets"])

    def test_cli_export_telemetry_training_rows_writes_joined_rows(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            feature_path = Path(tmp) / "features.json"
            label_path = Path(tmp) / "labels.json"
            output_path = Path(tmp) / "training_rows.json"
            feature_path.write_text(_telemetry_payload_fixture().model_dump_json(), encoding="utf-8")
            label_path.write_text(json.dumps({
                "label_rows": [{
                    "race_id": "2026-01-TEST",
                    "session": "fp1",
                    "driver_code": "LEC",
                    "pace_delta": -0.18,
                    "overtake": 0.04,
                }]
            }), encoding="utf-8")

            result = CliRunner().invoke(app, [
                "export-telemetry-training-rows",
                "--feature-payload-path", str(feature_path),
                "--label-path", str(label_path),
                "--output", str(output_path),
            ])
            rows = load_telemetry_training_rows(output_path)

        self.assertEqual(0, result.exit_code, result.output)
        self.assertEqual(1, len(rows))
        self.assertEqual("LEC", rows[0]["driver_code"])
        self.assertIn("rows: 1", result.output)

    def test_telemetry_training_loaders_accept_utf8_bom_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_path = root / "features.json"
            label_path = root / "labels.json"
            row_path = root / "rows.json"
            _write_bom_json(feature_path, [_telemetry_payload_fixture().model_dump(mode="json")])
            _write_bom_json(label_path, {
                "label_rows": [{
                    "race_id": "2026-01-TEST",
                    "session": "fp1",
                    "driver_code": "LEC",
                    "pace_delta": -0.18,
                    "overtake": 0.04,
                }]
            })
            _write_bom_json(row_path, {
                "training_rows": [{
                    "features": {"clean_air_pace_delta_s": 0.25},
                    "targets": {"pace_delta": -0.18},
                }]
            })

            payloads = load_telemetry_training_payloads(feature_path)
            labels = load_telemetry_label_rows(label_path)
            rows = load_telemetry_training_rows(row_path)

        self.assertEqual(1, len(payloads))
        self.assertEqual(1, len(labels))
        self.assertEqual(1, len(rows))
        self.assertEqual("LEC", next(iter(payloads[0].driver_features.keys())))
        self.assertEqual("LEC", labels[0]["driver_code"])
        self.assertAlmostEqual(-0.18, rows[0]["targets"]["pace_delta"])

    def test_cli_export_telemetry_training_rows_accepts_utf8_bom_inputs(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_path = root / "features.json"
            label_path = root / "labels.json"
            output_path = root / "training_rows.json"
            _write_bom_json(feature_path, [_telemetry_payload_fixture().model_dump(mode="json")])
            _write_bom_json(label_path, {
                "label_rows": [{
                    "race_id": "2026-01-TEST",
                    "session": "fp1",
                    "driver_code": "LEC",
                    "pace_delta": -0.18,
                    "overtake": 0.04,
                }]
            })

            result = CliRunner().invoke(app, [
                "export-telemetry-training-rows",
                "--feature-payload-path", str(feature_path),
                "--label-path", str(label_path),
                "--output", str(output_path),
            ])
            rows = load_telemetry_training_rows(output_path)

        self.assertEqual(0, result.exit_code, result.output)
        self.assertEqual(1, len(rows))
        self.assertIn("payloads: 1 labels: 1 rows: 1", result.output)

    def test_cli_export_telemetry_training_rows_fails_when_no_rows_join(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_path = root / "features.json"
            label_path = root / "labels.json"
            output_path = root / "training_rows.json"
            feature_path.write_text(_telemetry_payload_fixture("LEC").model_dump_json(), encoding="utf-8")
            label_path.write_text(json.dumps({
                "label_rows": [{
                    "race_id": "2026-01-TEST",
                    "session": "fp1",
                    "driver_code": "VER",
                    "pace_delta": -0.18,
                }]
            }), encoding="utf-8")

            result = CliRunner().invoke(app, [
                "export-telemetry-training-rows",
                "--feature-payload-path", str(feature_path),
                "--label-path", str(label_path),
                "--output", str(output_path),
            ])

        self.assertEqual(1, result.exit_code, result.output)
        self.assertIn("no telemetry training rows exported", result.output)
        self.assertFalse(output_path.exists())

    def test_cli_train_telemetry_artifacts_fails_when_supplied_labels_do_not_join(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            feature_path = root / "features.json"
            label_path = root / "labels.json"
            output_path = root / "telemetry"
            feature_path.write_text(_telemetry_payload_fixture("LEC").model_dump_json(), encoding="utf-8")
            label_path.write_text(json.dumps({
                "label_rows": [{
                    "race_id": "2026-01-TEST",
                    "session": "race",
                    "driver_code": "VER",
                    "pace_delta": -0.18,
                }]
            }), encoding="utf-8")

            result = CliRunner().invoke(app, [
                "train-telemetry-artifacts",
                "--output", str(output_path),
                "--artifact-id", "telemetry-empty-join",
                "--feature-payload-path", str(feature_path),
                "--label-path", str(label_path),
            ])

        self.assertEqual(1, result.exit_code, result.output)
        self.assertIn("no telemetry training rows found", result.output)
        self.assertFalse((output_path / "metadata.json").exists())

    def test_fastf1_artifact_builder_produces_metadata_from_fixture_provider(self):
        provider = _FixtureTelemetryProvider()
        races = provider.list_races(2025)

        bundle = build_fastf1_artifact_bundle(
            provider,
            training_races=races,
            target_race_id="2026-01-MIA",
            knowable_as_of=datetime(2026, 1, 1, tzinfo=timezone.utc),
            artifact_id="fixture-fastf1-artifact",
            season_start=2025,
            season_end=2025,
        )
        validation = validate_artifact_bundle(bundle, target_race="2026-01-MIA")

        self.assertTrue(validation.ok, validation.reason)
        self.assertEqual("fastf1", bundle.source_metadata["provider"])
        self.assertEqual("guarded", bundle.leakage_status["status"])
        self.assertGreaterEqual(bundle.source_metadata["coverage_counts"]["lap_rows"], 8)
        self.assertIn("AAA", bundle.drivers)
        self.assertIsNotNone(bundle.pace_model)
        self.assertIsNotNone(bundle.dnf_model)
        self.assertEqual("empirical_driver_median_pace_v1", bundle.pace_model.kind)
        self.assertEqual("smoothed_driver_dnf_hazard_v1", bundle.dnf_model.kind)
        self.assertIsNotNone(bundle.overtake_model)
        self.assertEqual("empirical_position_change_overtake_v1", bundle.overtake_model.kind)
        self.assertTrue(bundle.validation_metrics["pace_model"]["trained"])
        self.assertTrue(bundle.validation_metrics["dnf_model"]["trained"])
        self.assertTrue(bundle.validation_metrics["overtake_model"]["trained"])
        self.assertIn("mae_seconds", bundle.validation_metrics["pace_model"])
        self.assertIn("brier", bundle.validation_metrics["dnf_model"])
        self.assertIn("positive_passes", bundle.validation_metrics["overtake_model"])
        self.assertIn("model_quality", bundle.validation_metrics)

    def test_fastf1_artifact_builder_marks_future_training_as_leakage(self):
        provider = _FixtureTelemetryProvider()
        races = provider.list_races(2025)

        bundle = build_fastf1_artifact_bundle(
            provider,
            training_races=races,
            target_race_id="2025-01-MIA",
            knowable_as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
            artifact_id="fixture-leaky-artifact",
            season_start=2025,
            season_end=2025,
        )
        validation = validate_artifact_bundle(bundle, target_race="2025-01-MIA")

        self.assertFalse(validation.ok)
        self.assertEqual("leakage_guard_failed", validation.reason)

    def _two_driver_bundle(self):
        return F1MLArtifactBundle(
            artifact_id="two-driver-test-artifact",
            model_version="test-artifact-v1",
            training_seasons=[2026],
            training_races=["2026-04-MIA"],
            training_cutoff=datetime(2026, 5, 1, tzinfo=timezone.utc),
            knowable_as_of=datetime(2026, 5, 1, tzinfo=timezone.utc),
            feature_columns=["race_pace_score", "reliability_score", "track_laps"],
            drivers={
                "ANT": F1MLDriverArtifact(driver_code="ANT", pace_mean_seconds=74.5, pace_sigma_seconds=0.35, dnf_hazard_per_lap=0.0002, confidence=0.82, sources=["test_pace", "test_dnf"]),
                "RUS": F1MLDriverArtifact(driver_code="RUS", pace_mean_seconds=71.5, pace_sigma_seconds=0.28, dnf_hazard_per_lap=0.004, confidence=0.84, sources=["test_pace", "test_dnf"]),
            },
            validation_metrics={"source": "unit_test"},
            leakage_status={"status": "guarded", "future_results_excluded": True},
        )

    def _tiny_initial_state(self):
        return {
            "driver_codes": ["AAA", "BBB"],
            "driver_mean_pace_s": [80.0, 80.0],
            "driver_pace_sigma_s": [0.08, 0.08],
            "driver_dnf_rate_per_lap": [0.0001, 0.0001],
            "total_laps": 12,
        }


def _telemetry_payload_fixture(driver_code: str = "LEC", pace_delta: float = 0.25):
    driver_code = driver_code.upper()
    return TelemetryFeaturePayload(
        race_id="2026-01-TEST",
        session="fp1",
        source_mode="fixture",
        confidence=0.82,
        driver_features={
            driver_code: TelemetryFeatureVector(
                race_id="2026-01-TEST",
                session="fp1",
                driver_code=driver_code,
                source="fixture",
                clean_air_pace_delta_s=pace_delta,
                pace_sigma_delta=0.04,
                top_speed_delta_kph=3.0,
                corner_min_speed_delta_kph=1.5,
                traction_score=0.72,
                stability_score=0.88,
                tire_deg_slope_delta=0.01,
                traffic_penalty_s=0.03,
                overtake_pressure=0.42,
                dnf_hazard_multiplier=1.05,
                confidence=0.80,
                samples=24,
            )
        },
    )


def _write_bom_json(path: Path, payload) -> None:
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))


class _FakePaceAdapter:
    source = "fake_pace"
    confidence = 0.9

    def pace_distribution(self, driver_code, lap, state, features):
        if driver_code == "BBB":
            return PaceDistribution(mean_seconds=78.8, sigma_seconds=0.08, source=self.source, confidence=self.confidence)
        return PaceDistribution(mean_seconds=80.6, sigma_seconds=0.08, source=self.source, confidence=self.confidence)


class _FakeDNFAdapter:
    source = "fake_dnf"
    confidence = 0.9

    def dnf_hazard(self, driver_code, lap, state, features):
        return 0.08 if driver_code == "AAA" else 0.0001


class _BrokenPaceAdapter:
    source = "broken_pace"
    confidence = 0.1

    def pace_distribution(self, driver_code, lap, state, features):
        raise RuntimeError("boom")


class _FixtureTelemetryProvider:
    def list_races(self, season):
        return [
            MLSimRace(season=season, round=1, track_code="MIA", name="Fixture GP", scheduled_start=datetime(season, 5, 1, tzinfo=timezone.utc)),
        ]

    def laps(self, race, session):
        if session != SessionType.RACE:
            return []
        rows = []
        for code, base in (("AAA", 80.0), ("BBB", 81.0)):
            for lap in range(1, 8):
                if code == "BBB" and lap >= 5:
                    position = 1
                elif code == "AAA" and lap >= 5:
                    position = 2
                else:
                    position = 1 if code == "AAA" else 2
                rows.append(Lap(
                    race_id=f"{race.season}-{race.round:02d}-{race.track_code}",
                    driver_code=code,
                    lap_number=lap,
                    lap_time_s=base + lap * 0.04,
                    sector1_s=25.0,
                    sector2_s=30.0,
                    sector3_s=25.0,
                    compound=TireCompound.MEDIUM,
                    tire_age_laps=lap,
                    position=position,
                ))
        return rows

    def stream_telemetry(self, race, session):
        return iter(())

    def weather(self, race):
        return []

    def is_live(self):
        return False


if __name__ == "__main__":
    unittest.main()
