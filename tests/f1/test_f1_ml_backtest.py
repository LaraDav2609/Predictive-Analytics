import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from typer.testing import CliRunner

from sports.f1.ml.artifacts.builder import build_synthetic_artifact_bundle
from sports.f1.ml.artifacts.loader import load_simulator_model_bundle
from sports.f1.ml.artifacts.store import validate_artifact_bundle
from sports.f1.ml.backtest.walk_forward import MLBacktestConfig, run_backtest
from sports.f1.ml.cli import app
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

        self.assertEqual("uniform", comparison["best_model"])

    def test_cli_backtest_model_all_smoke(self):
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
            self.assertTrue((output / "per_race_metrics_2026.csv").exists())
            self.assertTrue((output / "aggregate_metrics_2026.json").exists())
            self.assertTrue((output / "model_comparison_2026.json").exists())
            self.assertTrue((output / "calibration_2026.json").exists())
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


if __name__ == "__main__":
    unittest.main()
