import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    from typer.testing import CliRunner
    from sports.f1.ml.cli import app
except ModuleNotFoundError:
    CliRunner = None
    app = None

from common.ml.backtest.walk_forward import RaceData
from sports.f1.ml.artifacts.builder import build_synthetic_artifact_bundle
from sports.f1.ml.artifacts.loader import load_simulator_model_bundle
from sports.f1.ml.artifacts.store import validate_artifact_bundle
from sports.f1.ml.backtest.walk_forward import (
    MLBacktestConfig,
    evaluate_out_of_sample_gate,
    run_backtest,
    select_training_races,
)
from sports.f1.predictor.models.baseline import BaselineRaceModel
from sports.f1.predictor.models.registry import F1ModelRegistry


class F1MLBacktestTests(unittest.TestCase):
    def test_synthetic_model_all_returns_comparison_rows(self):
        with TemporaryDirectory() as tmp:
            result = run_backtest(MLBacktestConfig(
                season=2026,
                provider="synthetic",
                models=("uniform", "production_v1", "ml_simulator_v1"),
                training_seasons=(2025,),
                min_training_races=5,
                output_dir=str(Path(tmp) / "out"),
                artifact_dir=str(Path(tmp) / "artifacts"),
                n_iterations=80,
            ))

        self.assertFalse(result.per_race_metrics.empty)
        self.assertEqual({"uniform", "production_v1", "ml_simulator_v1"}, set(result.per_race_metrics["model_id"]))
        self.assertIn(result.model_comparison["best_model"], {"uniform", "production_v1", "ml_simulator_v1"})
        self.assertIn("ml_simulator_v1", result.aggregate_metrics["models"])

    def test_ml_metrics_differ_from_uniform_on_synthetic_fixture(self):
        with TemporaryDirectory() as tmp:
            result = run_backtest(MLBacktestConfig(
                season=2026,
                provider="synthetic",
                models=("uniform", "ml_simulator_v1"),
                training_seasons=(2025,),
                min_training_races=5,
                output_dir=str(Path(tmp) / "out"),
                artifact_dir=str(Path(tmp) / "artifacts"),
                n_iterations=120,
            ))

        metrics = result.aggregate_metrics["models"]
        self.assertNotEqual(metrics["uniform"]["winner_brier"], metrics["ml_simulator_v1"]["winner_brier"])

    def test_backtest_writes_json_and_csv_outputs(self):
        with TemporaryDirectory() as tmp:
            result = run_backtest(MLBacktestConfig(
                season=2026,
                provider="synthetic",
                models=("uniform",),
                training_seasons=(2025,),
                min_training_races=5,
                output_dir=str(Path(tmp) / "out"),
                artifact_dir=str(Path(tmp) / "artifacts"),
                n_iterations=40,
            ))
            paths = result.output_paths

            for path in paths.values():
                self.assertTrue(Path(path).exists(), path)
            comparison = json.loads(Path(paths["model_comparison"]).read_text(encoding="utf-8"))
            validation_gate = json.loads(Path(paths["validation_gate"]).read_text(encoding="utf-8"))
            calibration = json.loads(Path(paths["calibration"]).read_text(encoding="utf-8"))
            validation_report = Path(paths["validation_report"]).read_text(encoding="utf-8")

        self.assertEqual("uniform", comparison["best_model"])
        self.assertEqual("blocked", validation_gate["status"])
        self.assertIn("candidate_missing", validation_gate["reasons"])
        self.assertIn("calibration_gate", calibration["models"]["uniform"])
        self.assertIn(calibration["models"]["uniform"]["calibration_gate"]["status"], {"passed", "blocked", "unproven"})
        self.assertIn("F1 Out-of-Sample Validation Report", validation_report)
        self.assertIn("Calibration Gate", validation_report)
        self.assertIn("Live trading remains blocked", validation_report)

    def test_training_selection_excludes_future_target_and_anomalies(self):
        races = [
            _race("SYN-2025-01", 2025, 1),
            _race("SYN-2025-02", 2025, 2),
            _race("SYN-2026-01", 2026, 1),
            _race("SYN-2026-02", 2026, 2),
        ]

        selected = select_training_races(
            races,
            races[-1],
            (2025, 2026),
            excluded_race_ids={"SYN-2025-02"},
        )

        self.assertEqual(["SYN-2025-01", "SYN-2026-01"], [race.race_id for race in selected])

    def test_training_selection_supports_track_include_and_exclude_filters(self):
        races = [
            _race("2025-01-BAHRAIN", 2025, 1),
            _race("2025-02-MONZA", 2025, 2),
            _race("2025-03-SPA", 2025, 3),
            _race("2026-01-MONZA", 2026, 1),
        ]

        selected = select_training_races(
            races,
            races[-1],
            (2025, 2026),
            included_track_codes={"MONZA", "SPA"},
            excluded_track_codes={"SPA"},
        )

        self.assertEqual(["2025-02-MONZA"], [race.race_id for race in selected])

    def test_out_of_sample_gate_requires_candidate_to_beat_baseline(self):
        aggregate = {
            "models": {
                "uniform": {"race_count": 8, "winner_brier": 0.20, "winner_log_loss": 0.70, "winner_accuracy": 0.25},
                "ml_simulator_v1": {"race_count": 8, "winner_brier": 0.18, "winner_log_loss": 0.65, "winner_accuracy": 0.38},
            }
        }

        result = evaluate_out_of_sample_gate(aggregate, min_races=5)

        self.assertTrue(result["passed"])
        self.assertEqual("passed", result["status"])
        self.assertEqual([], result["reasons"])
        self.assertTrue(result["comparisons"]["winner_brier"]["passed"])

    def test_out_of_sample_gate_blocks_weaker_candidate(self):
        aggregate = {
            "models": {
                "uniform": {"race_count": 8, "winner_brier": 0.20, "winner_log_loss": 0.70, "winner_accuracy": 0.25},
                "ml_simulator_v1": {"race_count": 8, "winner_brier": 0.22, "winner_log_loss": 0.72, "winner_accuracy": 0.25},
            }
        }

        result = evaluate_out_of_sample_gate(aggregate, min_races=5)

        self.assertFalse(result["passed"])
        self.assertIn("winner_brier_not_improved", result["reasons"])
        self.assertIn("winner_log_loss_not_improved", result["reasons"])

    def test_out_of_sample_gate_blocks_when_candidate_calibration_fails(self):
        aggregate = {
            "models": {
                "uniform": {"race_count": 8, "winner_brier": 0.20, "winner_log_loss": 0.70, "winner_accuracy": 0.25},
                "ml_simulator_v1": {"race_count": 8, "winner_brier": 0.18, "winner_log_loss": 0.65, "winner_accuracy": 0.38},
            }
        }
        calibration = {
            "models": {
                "ml_simulator_v1": {
                    "calibration_gate": {
                        "status": "blocked",
                        "brier_improvement": -0.002,
                    }
                }
            }
        }

        result = evaluate_out_of_sample_gate(aggregate, calibration=calibration, min_races=5)

        self.assertFalse(result["passed"])
        self.assertIn("calibration_brier_not_improved", result["reasons"])
        self.assertEqual("blocked", result["calibration_gate"]["status"])

    def test_out_of_sample_gate_accepts_passing_candidate_calibration(self):
        aggregate = {
            "models": {
                "uniform": {"race_count": 8, "winner_brier": 0.20, "winner_log_loss": 0.70, "winner_accuracy": 0.25},
                "ml_simulator_v1": {"race_count": 8, "winner_brier": 0.18, "winner_log_loss": 0.65, "winner_accuracy": 0.38},
            }
        }
        calibration = {
            "models": {
                "ml_simulator_v1": {
                    "calibration_gate": {
                        "status": "passed",
                        "brier_improvement": 0.004,
                    }
                }
            }
        }

        result = evaluate_out_of_sample_gate(aggregate, calibration=calibration, min_races=5)

        self.assertTrue(result["passed"])
        self.assertEqual([], result["reasons"])
        self.assertEqual("passed", result["calibration_gate"]["status"])

    def test_cli_backtest_model_all_smoke(self):
        if CliRunner is None or app is None:
            self.skipTest("typer is not installed in this test runtime")
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            artifacts = Path(tmp) / "artifacts"
            result = CliRunner().invoke(app, [
                "backtest",
                "--season", "2026",
                "--provider", "synthetic",
                "--model", "all",
                "--training-seasons", "2025",
                "--min-training-races", "5",
                "--n-iterations", "40",
                "--output-dir", str(output),
                "--artifact-dir", str(artifacts),
            ])

            self.assertEqual(0, result.exit_code, result.output)
            self.assertIn("best model:", result.output)
            self.assertIn("validation gate:", result.output)
            self.assertTrue((output / "per_race_metrics_2026.csv").exists())
            self.assertTrue((output / "aggregate_metrics_2026.json").exists())
            self.assertTrue((output / "model_comparison_2026.json").exists())
            self.assertTrue((output / "calibration_2026.json").exists())
            self.assertTrue((output / "validation_gate_2026.json").exists())
            self.assertTrue((output / "validation_report_2026.md").exists())
            self.assertTrue((output / "artifact_manifest_2026.json").exists())

    def test_leakage_guard_rejects_target_race_in_training(self):
        bundle = build_synthetic_artifact_bundle(training_races=["SYN-2026-01"])
        validation = validate_artifact_bundle(bundle, target_race="SYN-2026-01")

        self.assertFalse(validation.ok)
        self.assertEqual("target_race_in_training", validation.reason)

    def test_missing_artifact_fallback_is_visible(self):
        bundle = load_simulator_model_bundle("missing-artifact.json", target_race="SYN-2026-01")
        summary = bundle.source_summary()

        self.assertFalse(summary["ml_model_contract_used"])
        self.assertEqual("artifact_load_failed", summary["ml_model_fallback_reason"])

    def test_backtest_artifact_manifest_reports_ml_artifacts(self):
        with TemporaryDirectory() as tmp:
            result = run_backtest(MLBacktestConfig(
                season=2026,
                provider="synthetic",
                models=("ml_simulator_v1",),
                training_seasons=(2025,),
                min_training_races=5,
                output_dir=str(Path(tmp) / "out"),
                artifact_dir=str(Path(tmp) / "artifacts"),
                n_iterations=40,
            ))
            manifest = json.loads(Path(result.output_paths["artifact_manifest"]).read_text(encoding="utf-8"))

        self.assertGreater(manifest["artifact_count"], 0)
        self.assertIn("fallback_rate", manifest)
        self.assertEqual("ok", manifest["artifacts"][0]["artifact_validation"])

    def test_production_model_remains_default(self):
        registry = F1ModelRegistry()

        self.assertEqual("production_v1", registry.model_id)
        self.assertIsInstance(registry._model, BaselineRaceModel)


def _race(race_id: str, season: int, round_num: int) -> RaceData:
    return RaceData(
        race_id=race_id,
        season=season,
        round=round_num,
        decision_time=datetime(season, 1, min(28, round_num), 12, tzinfo=timezone.utc),
        finish_order=["AAA", "BBB", "CCC"],
        dnf_drivers=set(),
    )


if __name__ == "__main__":
    unittest.main()
