"""Tests for the trained win model + its walk-forward evaluation
(``sports.f1.ml.training.win_model``)."""
import random
import unittest

import os
import tempfile

from sports.f1.ml.training.win_model import (
    FEATURES,
    ServeWinModel,
    TrainedWinModel,
    build_field,
    fit_serve_win_model,
    walk_forward_train_eval,
)


def _row(season, rnd, winner_feature=True, n=8, seed=0):
    """A synthetic full backtest row. When winner_feature is True, the winner's
    'form' is high and everyone else's is low, so the feature is learnable. The
    heuristic distribution is uniform (a weak baseline to beat)."""
    rng = random.Random(seed)
    drivers = [f"d{i}" for i in range(n)]
    winner = drivers[rng.randrange(n)]
    component_scores = {}
    dist = []
    for d in drivers:
        feats = {f: 0.0 for f in FEATURES}
        if winner_feature:
            feats["form"] = 0.9 if d == winner else rng.uniform(0.0, 0.35)
        else:
            feats["form"] = rng.uniform(0.0, 1.0)
        component_scores[d] = feats
        dist.append({"driver_id": d, "win_probability": 1.0 / n})
    return {
        "season": season,
        "round": rnd,
        "actual_winner": winner,
        "component_scores": component_scores,
        "probability_distribution": dist,
    }


def _season_rows(n_races, winner_feature=True, base_seed=0):
    rows = []
    idx = 0
    for season in (2023, 2024, 2025):
        for rnd in range(1, 20):
            if idx >= n_races:
                break
            rows.append(_row(season, rnd, winner_feature=winner_feature, seed=base_seed + idx))
            idx += 1
    return rows


class BuildFieldTests(unittest.TestCase):
    def test_extracts_features_and_winner_label(self):
        row = _row(2024, 1, seed=5)
        driver_ids, matrix, labels = build_field(row)
        self.assertEqual(len(driver_ids), 8)
        self.assertEqual(len(matrix[0]), len(FEATURES))
        self.assertEqual(sum(labels), 1.0)  # exactly one winner
        winner_idx = labels.index(1.0)
        self.assertEqual(driver_ids[winner_idx], row["actual_winner"])


class TrainedWinModelTests(unittest.TestCase):
    def test_fits_and_predicts_a_normalised_field(self):
        rows = _season_rows(30, winner_feature=True)
        model = TrainedWinModel("logistic").fit(rows)
        self.assertTrue(model.is_fitted())
        field = model.predict_field(rows[0])
        self.assertEqual(len(field), 8)
        self.assertAlmostEqual(sum(field.values()), 1.0, places=6)  # winner market sums to 1

    def test_unfitted_model_returns_uniform_field(self):
        # single-class data (no winners distinguishable) -> cannot fit -> uniform
        model = TrainedWinModel("logistic")
        field = model.predict_field(_row(2024, 1, seed=1))
        self.assertEqual(len(field), 8)
        self.assertAlmostEqual(sum(field.values()), 1.0, places=6)


class WalkForwardTrainEvalTests(unittest.TestCase):
    def test_learnable_signal_beats_baseline_and_heuristic(self):
        rows = _season_rows(40, winner_feature=True)
        report = walk_forward_train_eval(rows, model_kind="logistic", min_train_races=15, val_block=5)
        self.assertTrue(report["ok"])
        agg = report["aggregate"]
        self.assertLess(agg["trained"]["brier"], agg["baseline_uniform"]["brier"])
        # heuristic here is uniform, so trained should beat it too
        self.assertTrue(report["gate"]["beats_baseline"])
        self.assertTrue(report["gate"]["beats_heuristic"])

    def test_not_enough_races_reported(self):
        report = walk_forward_train_eval(_season_rows(8), min_train_races=20, val_block=5)
        self.assertFalse(report["ok"])
        self.assertEqual(report["reason"], "not_enough_races_for_walk_forward")

    def test_structure_has_folds_and_gate(self):
        report = walk_forward_train_eval(_season_rows(30, winner_feature=True), min_train_races=15, val_block=5)
        self.assertIn("folds", report)
        self.assertIn("verdict", report["gate"])
        self.assertGreaterEqual(report["fold_count"], 1)


class ServeWinModelTests(unittest.TestCase):
    def test_serve_applier_matches_sklearn(self):
        """The pure-Python serve applier must reproduce the fitted sklearn model."""
        rows = _season_rows(30, winner_feature=True)
        trained = TrainedWinModel("logistic").fit(rows)
        serve = ServeWinModel.from_dict(trained.linear_artifact())
        self.assertFalse(serve.is_identity())
        row = rows[0]
        sk_field = trained.predict_field(row)
        comps = {d: f for d, f in (row.get("component_scores") or {}).items()}
        serve_field = serve.apply_field(comps)
        for d in sk_field:
            self.assertAlmostEqual(sk_field[d], serve_field[d], places=4)

    def test_roundtrip_save_load(self):
        rows = _season_rows(30, winner_feature=True)
        serve = fit_serve_win_model(rows, source="test")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "win.json")
            serve.save(path)
            loaded = ServeWinModel.load(path)
        self.assertEqual(loaded.features, serve.features)
        self.assertEqual(len(loaded.weights), len(FEATURES))
        self.assertAlmostEqual(loaded.intercept, serve.intercept, places=6)

    def test_gbm_has_no_linear_artifact(self):
        rows = _season_rows(30, winner_feature=True)
        self.assertIsNone(TrainedWinModel("gbm").fit(rows).linear_artifact())

    def test_identity_returns_uniform_field(self):
        serve = ServeWinModel.identity()
        field = serve.apply_field({"a": {}, "b": {}, "c": {}})
        self.assertEqual(field, {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})


if __name__ == "__main__":
    unittest.main()
