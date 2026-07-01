"""Tests for the walk-forward out-of-sample validation harness
(``sports.f1.predictor.backtesting.validation``).

Covers race selection into time-ordered folds (no look-ahead), the raw /
calibrated / baseline metric blocks, and the A7 pass/fail gate.
"""
import random
import unittest

from sports.f1.predictor.backtesting.validation import (
    build_walk_forward_folds,
    evaluate_split,
    run_walk_forward_validation,
    sort_rows,
    _uniform_pairs,
    _winner_accuracy,
)


def _row(season, rnd, winner, field):
    """field: {driver_id: win_probability}."""
    return {
        "season": season,
        "round": rnd,
        "actual_winner": winner,
        "probability_distribution": [
            {"driver_id": d, "win_probability": p} for d, p in field.items()
        ],
    }


def _informative_rows(n_races=40, field_size=8, seed=1):
    """Oracle-ish informative model: the actual winner is given 0.40, the rest
    share 0.60. Beats a uniform baseline comfortably."""
    rng = random.Random(seed)
    drivers = [f"d{i}" for i in range(field_size)]
    rows = []
    idx = 0
    for season in (2023, 2024, 2025):
        for rnd in range(1, 20):
            if idx >= n_races:
                break
            winner = rng.choice(drivers)
            other = 0.60 / (field_size - 1)
            field = {d: (0.40 if d == winner else other) for d in drivers}
            rows.append(_row(season, rnd, winner, field))
            idx += 1
    return rows


def _uniform_rows(n_races=40, field_size=8, seed=2):
    """Uninformative model: every driver gets 1/field, winner random. Cannot
    beat the uniform baseline."""
    rng = random.Random(seed)
    drivers = [f"d{i}" for i in range(field_size)]
    rows = []
    idx = 0
    for season in (2023, 2024, 2025):
        for rnd in range(1, 20):
            if idx >= n_races:
                break
            winner = rng.choice(drivers)
            field = {d: 1.0 / field_size for d in drivers}
            rows.append(_row(season, rnd, winner, field))
            idx += 1
    return rows


def _overconfident_rows(n_races=120, seed=3):
    """Informative but systematically overconfident: the favourite always gets
    0.70 but only wins ~45% of the time — isotonic calibration should reduce
    out-of-sample Brier."""
    rng = random.Random(seed)
    drivers = [f"d{i}" for i in range(6)]
    fav = drivers[0]
    other = 0.30 / 5
    rows = []
    for i in range(n_races):
        winner = fav if rng.random() < 0.45 else rng.choice(drivers[1:])
        field = {d: (0.70 if d == fav else other) for d in drivers}
        rows.append(_row(2024, i + 1, winner, field))
    return rows


class WalkForwardFoldTests(unittest.TestCase):
    def test_folds_are_chronological_with_no_lookahead(self):
        rows = _informative_rows(40)
        folds = build_walk_forward_folds(rows, min_train_races=20, val_block=5)
        self.assertTrue(folds)
        for fold in folds:
            # training window ends strictly before the validation window starts
            self.assertLess(fold["train_span"][1], fold["val_span"][0])
            self.assertGreaterEqual(fold["train_size"], 20)
            self.assertLessEqual(fold["val_size"], 5)

    def test_validation_blocks_are_disjoint_and_cover_the_tail(self):
        rows = _informative_rows(40)
        folds = build_walk_forward_folds(rows, min_train_races=20, val_block=5, step=5)
        seen = []
        for fold in folds:
            for r in fold["val"]:
                seen.append((r["season"], r["round"]))
        self.assertEqual(len(seen), len(set(seen)), "validation races must not repeat")
        # 40 races, train starts at 20, blocks of 5 -> 20 validation races
        self.assertEqual(len(seen), 20)

    def test_insufficient_races_yields_no_folds(self):
        rows = _informative_rows(10)
        self.assertEqual(build_walk_forward_folds(rows, min_train_races=20, val_block=5), [])

    def test_sort_rows_orders_by_season_then_round(self):
        rows = [_row(2025, 2, "d0", {"d0": 1.0}), _row(2024, 9, "d0", {"d0": 1.0}), _row(2024, 1, "d0", {"d0": 1.0})]
        keys = [(r["season"], r["round"]) for r in sort_rows(rows)]
        self.assertEqual(keys, [(2024, 1), (2024, 9), (2025, 2)])


class MetricTests(unittest.TestCase):
    def test_uniform_baseline_pairs(self):
        rows = [_row(2024, 1, "d0", {"d0": 0.9, "d1": 0.1})]
        pairs = _uniform_pairs(rows)
        self.assertEqual(sorted(pairs), [(0.5, 0.0), (0.5, 1.0)])

    def test_winner_accuracy(self):
        rows = [
            _row(2024, 1, "d0", {"d0": 0.6, "d1": 0.4}),  # top pick wins
            _row(2024, 2, "d1", {"d0": 0.6, "d1": 0.4}),  # top pick loses
        ]
        self.assertEqual(_winner_accuracy(rows), 0.5)


class EvaluateSplitTests(unittest.TestCase):
    def test_calibration_improves_overconfident_model_out_of_sample(self):
        rows = _overconfident_rows(120)
        cut = 96
        report = evaluate_split(rows[:cut], rows[cut:], method="isotonic")
        self.assertFalse(report["calibrator"]["is_identity"], "should have enough data to fit")
        self.assertLessEqual(
            report["calibrated"]["brier"],
            report["raw"]["brier"] + 1e-6,
            "isotonic calibration should not worsen OOS Brier on overconfident data",
        )

    def test_tiny_training_set_returns_identity_calibrator(self):
        rows = _informative_rows(12)
        report = evaluate_split(rows[:6], rows[6:], method="isotonic")
        self.assertTrue(report["calibrator"]["is_identity"])


class GateTests(unittest.TestCase):
    def test_informative_model_passes_and_beats_baseline(self):
        result = run_walk_forward_validation(_informative_rows(40), min_train_races=20, val_block=5)
        self.assertTrue(result["ok"])
        self.assertTrue(result["gate"]["beats_baseline"])
        self.assertTrue(result["gate"]["passed"])
        agg = result["aggregate"]
        self.assertLess(agg["raw"]["brier"], agg["baseline_uniform"]["brier"])

    def test_uniform_model_fails_the_gate(self):
        result = run_walk_forward_validation(_uniform_rows(40), min_train_races=20, val_block=5)
        self.assertTrue(result["ok"])
        self.assertFalse(result["gate"]["beats_baseline"])
        self.assertFalse(result["gate"]["passed"])

    def test_not_enough_races_is_reported_not_crashed(self):
        result = run_walk_forward_validation(_informative_rows(10), min_train_races=20, val_block=5)
        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "not_enough_races_for_walk_forward")


if __name__ == "__main__":
    unittest.main()
