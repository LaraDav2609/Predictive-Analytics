import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from common.ml.bridge.outcome_publisher import InMemoryOutcomePublisher
from common.ml.types import OutcomeProbability
from sports.f1.ml.artifacts.builder import build_synthetic_artifact_bundle
from sports.f1.ml.artifacts.store import save_artifact_bundle
from sports.f1.ml.live_runner import LiveRunnerConfig, run_live_once
from sports.f1.models.f1 import Constructor, Driver, Race
from sports.f1.predictor.models.baseline import BaselineRaceModel
from sports.f1.predictor.models.registry import F1ModelRegistry

try:
    from typer.testing import CliRunner
    from sports.f1.ml.cli import app
except ModuleNotFoundError:
    CliRunner = None
    app = None


class F1MLLiveRunnerTests(unittest.TestCase):
    def test_estimated_live_once_is_low_confidence_and_publishes_in_memory(self):
        publisher = InMemoryOutcomePublisher()
        result = run_live_once(
            LiveRunnerConfig(
                race="synthetic_2026_r01",
                source="estimated",
                once=True,
                dry_run=True,
                n_iterations=40,
                physical=False,
            ),
            publisher=publisher,
        )

        payload = result.payload
        self.assertEqual("estimated", payload["source_mode"])
        self.assertLessEqual(payload["confidence"], 0.25)
        self.assertEqual("ml_simulator_v1", payload["model_id"])
        self.assertGreater(len(result.published_records), 0)
        self.assertTrue(all(isinstance(item, OutcomeProbability) for item in result.published_records))
        self.assertTrue(publisher.snapshots)

    def test_replay_lap_changes_probability_output(self):
        early = run_live_once(
            LiveRunnerConfig(race="synthetic_2026_r01", source="replay", once=True, dry_run=True, n_iterations=60, physical=False, lap=1)
        ).payload
        later = run_live_once(
            LiveRunnerConfig(race="synthetic_2026_r01", source="replay", once=True, dry_run=True, n_iterations=60, physical=False, lap=12)
        ).payload

        self.assertEqual("recorded", early["source_mode"])
        self.assertEqual("recorded", later["source_mode"])
        self.assertNotEqual(
            early["probabilities"][0]["win_probability"],
            later["probabilities"][0]["win_probability"],
        )
        self.assertIn("replay_laps", later["evidence_groups_used"])

    def test_artifact_path_is_used_and_metadata_is_visible(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            save_artifact_bundle(path, build_synthetic_artifact_bundle())
            payload = run_live_once(
                LiveRunnerConfig(
                    race="SYN-2026-01",
                    source="replay",
                    artifact_path=str(path),
                    once=True,
                    dry_run=True,
                    n_iterations=40,
                    physical=False,
                )
            ).payload

        self.assertTrue(payload["trained_artifacts_used"])
        self.assertEqual("trained_artifacts", payload["ml_input_source"])
        self.assertEqual("synthetic-ml-simulator-artifact", payload["artifact_id"])
        self.assertTrue(payload["ml_model_contract_used"])

    def test_missing_artifact_falls_back_visibly(self):
        payload = run_live_once(
            LiveRunnerConfig(
                race="SYN-2026-01",
                source="estimated",
                artifact_path="missing-artifact.json",
                once=True,
                dry_run=True,
                n_iterations=40,
                physical=False,
            )
        ).payload

        self.assertFalse(payload["trained_artifacts_used"])
        self.assertEqual("evidence_fallback", payload["ml_input_source"])
        self.assertEqual("artifact_load_failed", payload["ml_fallback_reason"])

    def test_malformed_recording_falls_back_to_estimated(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "round_01_race.txt"
            path.write_text("not-json\nalso-not-json", encoding="utf-8")
            payload = run_live_once(
                LiveRunnerConfig(
                    race="SYN-2026-01",
                    source="fastf1-recorded",
                    recording_path=str(path),
                    once=True,
                    dry_run=True,
                    n_iterations=40,
                    physical=False,
                )
            ).payload

        self.assertEqual("estimated", payload["source_mode"])
        self.assertIn("recording_empty_or_unparsed", payload["fallback_reason"])

    def test_cli_live_estimated_smoke(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this runtime")
        result = CliRunner().invoke(app, [
            "live",
            "--race",
            "synthetic_2026_r01",
            "--source",
            "estimated",
            "--once",
            "--dry-run",
            "--n-iterations",
            "40",
            "--no-physical",
        ])

        self.assertEqual(0, result.exit_code, result.output)
        self.assertIn("live runner:", result.output)
        self.assertIn('"source_mode": "estimated"', result.output)

    def test_cli_live_replay_artifact_smoke(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this runtime")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "artifact.json"
            save_artifact_bundle(path, build_synthetic_artifact_bundle())
            result = CliRunner().invoke(app, [
                "live",
                "--race",
                "synthetic_2026_r01",
                "--source",
                "replay",
                "--artifact-path",
                str(path),
                "--once",
                "--dry-run",
                "--n-iterations",
                "40",
                "--no-physical",
            ])

        self.assertEqual(0, result.exit_code, result.output)
        payload = json.loads(result.output[result.output.index("{"):])
        self.assertEqual("trained_artifacts", payload["ml_input_source"])
        self.assertEqual("recorded", payload["source_mode"])

    def test_publish_failure_is_metadata_not_exception(self):
        class BrokenPublisher:
            def publish_batch(self, records):
                raise RuntimeError("redis down")

        payload = run_live_once(
            LiveRunnerConfig(race="SYN-2026-01", source="estimated", once=True, publish=True, n_iterations=40, physical=False),
            publisher=BrokenPublisher(),
        ).payload

        self.assertFalse(payload["publish_status"]["ok"])
        self.assertIn("publish_failed", payload["publish_status"]["reason"])

    def test_production_context_uses_race_truth_and_weekend_evidence(self):
        drivers = [
            Driver(id="max", number=1, code="VER", first_name="Max", last_name="Verstappen", nationality="Dutch", team="Red Bull Racing", points=25, position=1),
            Driver(id="charles", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari", points=18, position=2),
        ]
        constructors = [
            Constructor(id="red_bull", name="Red Bull Racing", nationality="Austrian", points=25, position=1),
            Constructor(id="ferrari", name="Ferrari", nationality="Italian", points=18, position=2),
        ]
        race = Race(round=1, name="Bahrain Grand Prix", circuit="Bahrain International Circuit", country="Bahrain", date="2026-03-01T14:00:00Z")
        features = {
            "completed_races": 1,
            "total_races": 24,
            "drivers": {
                "max": {"form_score": 0.82, "race_pace_score": 0.84, "qualifying_pace_score": 0.86, "reliability_score": 0.91},
                "charles": {"form_score": 0.74, "race_pace_score": 0.76, "qualifying_pace_score": 0.81, "reliability_score": 0.88},
            },
            "constructors": {
                "red_bull": {"team_score": 0.86},
                "ferrari": {"team_score": 0.78},
            },
        }
        weekend = {
            "ok": True,
            "source_mode": "recent",
            "confidence": 0.66,
            "session": "race",
            "drivers": {
                "max": {"practice": {"representative_lap": 92.1, "confidence": 0.7}, "grid": {"grid_position": 1, "confidence": 0.9}, "race_inputs": {"position": 1, "compound": "MEDIUM", "tyre_age": 5}},
                "charles": {"practice": {"representative_lap": 92.4, "confidence": 0.7}, "grid": {"grid_position": 2, "confidence": 0.9}, "race_inputs": {"position": 2, "compound": "MEDIUM", "tyre_age": 5}},
            },
            "missing_groups": ["race_control"],
        }
        truth = {
            "ok": True,
            "source_mode": "live",
            "confidence": 0.86,
            "status": "live",
            "by_driver_id": {
                "max": {"driver_id": "max", "driver_code": "VER", "position": 1, "lap": 12, "gap_to_leader": 0.0, "representative_lap": 92.0, "compound": "MEDIUM", "tyre_age": 8, "pit_stops": 0, "source_mode": "live", "confidence": 0.88},
                "charles": {"driver_id": "charles", "driver_code": "LEC", "position": 2, "lap": 12, "gap_to_leader": 4.2, "representative_lap": 92.6, "compound": "MEDIUM", "tyre_age": 8, "pit_stops": 0, "source_mode": "live", "confidence": 0.86},
            },
            "missing_groups": [],
        }

        payload = run_live_once(
            LiveRunnerConfig(
                race="2026-01-BAHRAIN",
                season=2026,
                round_num=1,
                source="auto",
                n_iterations=80,
                physical=False,
                dry_run=True,
                race_obj=race,
                drivers=drivers,
                constructors=constructors,
                features=features,
                race_truth=truth,
                weekend_evidence=weekend,
                live_state={"source_mode": "live", "confidence": 0.86},
            )
        ).payload

        self.assertTrue(payload["production_context"])
        self.assertTrue(payload["race_truth_used"])
        self.assertTrue(payload["weekend_evidence_used"])
        self.assertEqual("live", payload["source_mode"])
        self.assertGreater(payload["confidence"], 0.25)
        self.assertIn("race_truth", payload["evidence_groups_used"])
        self.assertEqual("2026-01-BAHRAIN", payload["race_id"])
        self.assertEqual({"LEC", "VER"}, {row["driver_code"] for row in payload["probabilities"]})

    def test_production_model_remains_default(self):
        registry = F1ModelRegistry()

        self.assertEqual("production_v1", registry.model_id)
        self.assertIsInstance(registry._model, BaselineRaceModel)


if __name__ == "__main__":
    unittest.main()
