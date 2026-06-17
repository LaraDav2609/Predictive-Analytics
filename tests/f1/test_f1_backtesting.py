import unittest
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from sports.f1.predictor.backtesting.loader import HistoricalRaceLoader
from sports.f1.predictor.backtesting.evidence_cache import WeekendEvidenceCacheWriter
from sports.f1.predictor.backtesting.metrics import evaluate_race, summarize_races
from sports.f1.predictor.backtesting.replay import RaceReplayBuilder
from sports.f1.predictor.backtesting.service import F1BacktestService
from sports.f1.predictor.models.configs import get_model_config
from sports.f1.predictor.models.registry import F1ModelRegistry
from sports.f1.models.f1 import Driver, DriverRacePrediction, Race, RacePrediction


class F1BacktestingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.races = [_race(1, "Opening GP", "max", "charles"), _race(2, "Second GP", "charles", "max")]

    def test_replay_removes_future_races_from_features(self):
        replay = RaceReplayBuilder().build(2025, self.races, 2)

        self.assertEqual(1, replay.features["completed_races"])
        self.assertEqual(1, replay.features["drivers"]["max"]["starts"])
        self.assertEqual(1, replay.features["drivers"]["charles"]["starts"])
        self.assertEqual(25.0, replay.drivers[0].points)
        self.assertEqual("charles", replay.actual_winner)

    def test_replay_stage_controls_current_weekend_evidence(self):
        races = [
            self.races[0],
            _race(
                2,
                "Second GP",
                "charles",
                "max",
                qualifying_order=["max", "charles"],
                practice_laps={"max": 71.0, "charles": 72.5},
            ),
        ]

        pre_weekend = RaceReplayBuilder().build(2025, races, 2, stage="pre_weekend")
        post_quali = RaceReplayBuilder().build(2025, races, 2, stage="post_qualifying")
        practice = RaceReplayBuilder().build(2025, races, 2, stage="practice")

        self.assertNotIn("qualifying_source", pre_weekend.features["drivers"]["max"])
        self.assertNotIn("practice_pace_score", pre_weekend.features["drivers"]["max"])
        self.assertEqual("target_qualifying_results", post_quali.features["drivers"]["max"]["qualifying_source"])
        self.assertEqual(1, post_quali.features["drivers"]["max"]["grid_position"])
        self.assertTrue(post_quali.features["backtest"]["target_qualifying_used"])
        self.assertIn("practice_pace_score", practice.features["drivers"]["max"])
        self.assertTrue(practice.features["backtest"]["target_practice_used"])

    def test_replay_aggregates_multiple_practice_sessions_per_driver(self):
        target = _race(2, "Second GP", "charles", "max")
        target["PracticeResults"] = [
            {
                "session": "fp1",
                "driver_id": "max",
                "best_lap": 72.0,
                "representative_lap": 72.4,
                "long_run_lap": 73.1,
                "representative_sector_1": 23.4,
                "laps": 10,
            },
            {
                "session": "fp2",
                "driver_id": "max",
                "best_lap": 70.8,
                "representative_lap": 71.2,
                "long_run_lap": 72.0,
                "representative_sector_1": 22.9,
                "laps": 12,
            },
            {
                "session": "fp1",
                "driver_id": "charles",
                "best_lap": 71.2,
                "representative_lap": 71.7,
                "long_run_lap": 72.5,
                "representative_sector_1": 23.1,
                "laps": 11,
            },
        ]

        replay = RaceReplayBuilder().build(2025, [self.races[0], target], 2, stage="practice_available")
        max_number = str(next(driver.number for driver in replay.drivers if driver.id == "max"))
        max_laps = replay.features["openf1_session"]["laps"]["drivers"][max_number]

        self.assertEqual(22, max_laps["laps"])
        self.assertEqual(70.8, max_laps["best_lap"])
        self.assertEqual(22.9, max_laps["best_sector_1"])
        self.assertEqual(23.15, max_laps["representative_sector_1"])
        self.assertIn("practice_pace_score", replay.features["drivers"]["max"])

    def test_replay_preserves_practice_telemetry_quality(self):
        target = _race(2, "Second GP", "charles", "max")
        target["PracticeResults"] = [
            {
                "session": "fp1",
                "driver_id": "max",
                "best_lap": 70.8,
                "representative_lap": 71.2,
                "representative_sector_1": 22.9,
                "representative_sector_2": 23.8,
                "representative_sector_3": 24.5,
                "pace_stability": 0.82,
                "telemetry_quality": 0.91,
                "usable_laps": 13,
                "raw_laps": 18,
                "laps": 13,
                "lap_distribution": {
                    "sample_size": 13,
                    "p10": 70.9,
                    "median": 71.2,
                    "p90": 71.8,
                    "spread_p90_p10": 0.9,
                    "best": 70.8,
                },
            },
            {
                "session": "fp1",
                "driver_id": "charles",
                "best_lap": 71.2,
                "representative_lap": 71.7,
                "representative_sector_1": 23.1,
                "representative_sector_2": 24.0,
                "representative_sector_3": 24.6,
                "pace_stability": 0.78,
                "telemetry_quality": 0.88,
                "usable_laps": 12,
                "raw_laps": 16,
                "laps": 12,
            },
        ]

        replay = RaceReplayBuilder().build(2025, [self.races[0], target], 2, stage="practice_available")
        max_number = str(next(driver.number for driver in replay.drivers if driver.id == "max"))
        max_laps = replay.features["openf1_session"]["laps"]["drivers"][max_number]

        self.assertEqual(0.91, max_laps["telemetry_quality"])
        self.assertEqual(13, max_laps["lap_distribution"]["sample_size"])
        self.assertEqual(18, max_laps["raw_laps"])
        self.assertEqual(13, max_laps["usable_laps"])
        self.assertGreater(replay.features["drivers"]["max"]["practice_confidence"], 0.5)
        self.assertIn("practice_distribution_stability", replay.features["drivers"]["max"])

    def test_replay_uses_race_stint_inputs_only_for_live_stage(self):
        target = _race(2, "Second GP", "charles", "max")
        target["RaceInputs"] = {
            "source": "openf1_compact_race_inputs",
            "drivers": {
                "max": {
                    "driver_id": "max",
                    "driver_number": 1,
                    "position": 2,
                    "gap_to_leader": "+2.4",
                    "interval": "+2.4",
                    "lap": 38,
                    "laps": 58,
                    "compound": "HARD",
                    "tyre_age": 23,
                    "stints": 2,
                    "compound_sequence": ["SOFT", "HARD"],
                    "avg_stint_laps": 24.0,
                    "max_stint_laps": 31,
                    "final_stint_laps": 31,
                    "stint_lap_distribution": {"sample_size": 2, "median": 24.0, "p90": 30.3},
                    "pit_stops": 1,
                }
            },
        }

        post_quali = RaceReplayBuilder().build(2025, [self.races[0], target], 2, stage="post_qualifying")
        live = RaceReplayBuilder().build(2025, [self.races[0], target], 2, stage="live")

        self.assertFalse(post_quali.features["backtest"]["target_race_inputs_used"])
        self.assertTrue(live.features["backtest"]["target_race_inputs_used"])
        live_driver = live.features["weekend_evidence"]["drivers"]["max"]["race_inputs"]
        self.assertEqual(["SOFT", "HARD"], live_driver["compound_sequence"])
        self.assertEqual("stable", live_driver["tyre_phase"])
        self.assertEqual(2, live_driver["stint_lap_distribution"]["sample_size"])

    def test_metrics_calculate_accuracy_and_probability_quality(self):
        prediction = RacePrediction(
            driver_predictions={
                "max": DriverRacePrediction(
                    driver_id="max",
                    driver_name="Max Verstappen",
                    win_prob=0.70,
                    podium_prob=0.90,
                    top5_prob=0.95,
                    predicted_position=1,
                    expected_finish=1.3,
                ),
                "charles": DriverRacePrediction(
                    driver_id="charles",
                    driver_name="Charles Leclerc",
                    win_prob=0.30,
                    podium_prob=0.80,
                    top5_prob=0.90,
                    predicted_position=2,
                    expected_finish=2.1,
                ),
            }
        )
        metrics = evaluate_race(prediction, _parse_actual(self.races[0]))

        self.assertTrue(metrics["winner_hit"])
        self.assertEqual(2, metrics["podium_hits"])
        self.assertLess(metrics["log_loss"], 0.4)

    async def test_service_runs_without_sentiment_or_external_calls(self):
        service = F1BacktestService(_FakeClient(self.races))

        result = await service.backtest_season(2025, include_races=True)

        self.assertTrue(result["ok"])
        self.assertEqual(2, result["race_count"])
        self.assertIn("recommended_weights", result)
        self.assertEqual(2, len(result["races"]))

    async def test_model_compare_returns_best_model(self):
        service = F1BacktestService(_FakeClient(self.races))

        result = await service.compare_season(2025)

        self.assertTrue(result["ok"])
        self.assertIn("best_model_id", result)
        self.assertGreaterEqual(len(result["models"]), 3)
        self.assertTrue(all("model_id" in model for model in result["models"]))

    def test_model_config_falls_back_to_production(self):
        config = get_model_config("does_not_exist")

        self.assertEqual("production_v1", config.model_id)
        self.assertTrue(any(model["model_id"] == "conservative_v1" for model in F1ModelRegistry.list_models()))

    async def test_backtest_accepts_selected_model_id(self):
        service = F1BacktestService(_FakeClient(self.races))

        result = await service.backtest_season(2025, model_id="conservative_v1")

        self.assertTrue(result["ok"])
        self.assertEqual("conservative_v1", result["model_id"])

    async def test_backtest_accepts_ml_simulator_model_id(self):
        service = F1BacktestService(_FakeClient(self.races))

        result = await service.backtest_season(2025, model_id="ml_simulator_v1")

        self.assertTrue(result["ok"])
        self.assertEqual("ml_simulator_v1", result["model_id"])
        self.assertEqual(2, result["race_count"])
        self.assertGreaterEqual(result["fallback_coverage"]["fallback_count"], 1)
        self.assertEqual("guarded", result["leakage_status"]["status"])

    async def test_deep_backtest_loads_seasons_once_and_reports_matrices(self):
        fake = _FakeClient(self.races)
        service = F1BacktestService(fake)

        result = await service.deep_backtest(
            2025,
            2025,
            stages=["pre_weekend", "post_qualifying"],
            model_ids=["production_v1", "conservative_v1"],
            include_races=False,
        )

        self.assertTrue(result["ok"])
        self.assertEqual("deep_backtest", result["mode"])
        self.assertEqual(1, fake.race_loads)
        self.assertEqual(1, fake.qualifying_loads)
        self.assertEqual(["pre_weekend", "post_qualifying"], result["requested"]["stages"])
        self.assertEqual(4, len(result["runs"]))
        self.assertTrue(result["stage_matrix"])
        self.assertTrue(result["model_matrix"])
        self.assertTrue(result["track_segment_matrix"])
        self.assertTrue(all(run.get("track_segment_summary") for run in result["runs"]))
        segment_names = {row["segment"] for row in result["track_segment_matrix"]}
        self.assertIn("tyre_stress", segment_names)
        self.assertTrue(result["evidence_ablations"])
        self.assertTrue(result["track_segment_ablations"])
        tyre_ablation = next(
            item for item in result["track_segment_ablations"]
            if item["segment"] == "tyre_stress" and item["comparison_stage"] == "post_qualifying"
        )
        self.assertIn(tyre_ablation["interpretation"], {"evidence_helped", "evidence_hurt_or_overfit", "neutral"})
        self.assertEqual("very_thin", result["coverage"]["coverage_grade"])
        self.assertEqual(2, result["coverage"]["totals"]["completed_races"])
        self.assertTrue(result["recommendations"])
        self.assertIn("compact_report", result)
        self.assertIn("track_segment_count", result["compact_report"])
        self.assertTrue(result["load_strategy"]["upstream_calls_minimized"])

    async def test_post_qualifying_deep_backtest_does_not_require_practice_coverage(self):
        fake = _FakeClient(self.races)
        service = F1BacktestService(fake)

        result = await service.deep_backtest(
            2025,
            2025,
            stages=["post_qualifying"],
            model_ids=["production_v1"],
            include_ablations=False,
        )

        self.assertTrue(result["ok"])
        self.assertFalse(result["coverage"]["evidence_groups"]["practice"]["requested"])
        limitation_codes = {item["code"] for item in result["limitations"]}
        self.assertNotIn("practice_coverage_low", limitation_codes)
        self.assertNotIn("practice_telemetry_proxy", limitation_codes)
        self.assertIn("upstream_calls_minimized", limitation_codes)

    async def test_practice_deep_backtest_reports_practice_specific_limitations(self):
        fake = _FakeClient(self.races)
        service = F1BacktestService(fake)

        result = await service.deep_backtest(
            2025,
            2025,
            stages=["practice_available"],
            model_ids=["production_v1"],
            include_ablations=False,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(result["coverage"]["evidence_groups"]["practice"]["requested"])
        limitation_codes = {item["code"] for item in result["limitations"]}
        self.assertTrue({"practice_telemetry_proxy", "practice_compact_summary", "practice_compact_distribution"} & limitation_codes)

    async def test_deep_backtest_rejects_current_partial_without_flag(self):
        fake = _FakeClient(self.races, season=2025)
        service = F1BacktestService(fake)

        result = await service.deep_backtest(
            2025,
            2025,
            stages=["pre_weekend"],
            model_ids=["production_v1"],
            include_ablations=False,
        )

        self.assertTrue(result["ok"])
        self.assertEqual([], result["load_strategy"]["seasons_loaded"])
        self.assertEqual("season_load_errors", result["limitations"][0]["code"])

    async def test_historical_loader_disk_cache_avoids_second_upstream_load(self):
        with TemporaryDirectory() as tmp:
            first_client = _FakeClient(self.races, season=2026)
            first = HistoricalRaceLoader(first_client, cache_dir=tmp)
            first_rows = await first.load_season(2025)

            second_client = _FakeClient([], season=2026)
            second = HistoricalRaceLoader(second_client, cache_dir=tmp)
            second_rows = await second.load_season(2025)

        self.assertEqual(2, len(first_rows))
        self.assertEqual(2, len(second_rows))
        self.assertEqual(1, first_client.race_loads)
        self.assertEqual(1, first_client.qualifying_loads)
        self.assertEqual(0, second_client.race_loads)
        self.assertEqual(0, second_client.qualifying_loads)
        self.assertEqual(1, second.cache_stats["hits"])

    async def test_historical_loader_cache_only_uses_cache_without_upstream(self):
        with TemporaryDirectory() as tmp:
            first_client = _FakeClient(self.races, season=2026)
            first = HistoricalRaceLoader(first_client, cache_dir=tmp)
            await first.load_season(2025)

            second_client = _FakeClient([], season=2026)
            second = HistoricalRaceLoader(second_client, cache_dir=tmp)
            rows = await second.load_season(2025, cache_policy="cache_only")

        self.assertEqual(2, len(rows))
        self.assertEqual(0, second_client.race_loads)
        self.assertEqual(0, second_client.qualifying_loads)
        self.assertEqual("cache_only", second.cache_stats["policy"])
        self.assertEqual(1, second.cache_stats["hits"])

    async def test_historical_loader_cache_hit_overlays_new_weekend_evidence(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "season"
            evidence_dir = Path(tmp) / "evidence"
            evidence_dir.mkdir()
            first_client = _FakeClient(self.races, season=2026)
            first = HistoricalRaceLoader(first_client, cache_dir=cache_dir, evidence_cache_dir=evidence_dir)
            await first.load_season(2025)
            (evidence_dir / "season_2025_round_02.json").write_text(json.dumps({
                "source": "late_local_fp_summary",
                "practice_results": [
                    {"driver_id": "max", "best_lap": 70.8, "representative_lap": 71.1, "laps": 18},
                    {"driver_id": "charles", "best_lap": 71.2, "representative_lap": 71.4, "laps": 17},
                ],
            }), encoding="utf-8")

            second_client = _FakeClient([], season=2026)
            second = HistoricalRaceLoader(second_client, cache_dir=cache_dir, evidence_cache_dir=evidence_dir)
            rows = await second.load_season(2025, cache_policy="cache_only")

        race_two = next(row for row in rows if row["round"] == "2")
        self.assertEqual(2, len(race_two["PracticeResults"]))
        self.assertEqual(0, second_client.race_loads)
        self.assertEqual(1, second.cache_stats["hits"])
        self.assertEqual(1, second.cache_stats["evidence_cache"]["files_found"])
        self.assertEqual(1, second.cache_stats["evidence_cache"]["races_enriched"])

    async def test_historical_loader_cache_only_miss_avoids_upstream(self):
        with TemporaryDirectory() as tmp:
            client = _FakeClient(self.races, season=2026)
            loader = HistoricalRaceLoader(client, cache_dir=tmp)

            with self.assertRaises(ValueError):
                await loader.load_season(2025, cache_policy="cache_only")

        self.assertEqual(0, client.race_loads)
        self.assertEqual(0, client.qualifying_loads)
        self.assertEqual("cache_only", loader.cache_stats["policy"])
        self.assertEqual(1, loader.cache_stats["misses"])

    async def test_historical_loader_treats_empty_disk_cache_as_stale(self):
        with TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            cache_dir.mkdir(exist_ok=True)
            (cache_dir / "season_2025_merged.json").write_text(json.dumps({
                "season": 2025,
                "races": [],
            }), encoding="utf-8")
            client = _FakeClient(self.races, season=2026)
            loader = HistoricalRaceLoader(client, cache_dir=cache_dir)

            rows = await loader.load_season(2025, cache_policy="read_through")

        self.assertEqual(2, len(rows))
        self.assertEqual(1, client.race_loads)
        self.assertEqual(1, loader.cache_stats["misses"])
        self.assertEqual(1, loader.cache_stats["errors"])
        self.assertEqual(1, loader.cache_stats["writes"])

    async def test_historical_loader_refresh_bypasses_existing_cache(self):
        with TemporaryDirectory() as tmp:
            first_client = _FakeClient(self.races, season=2026)
            first = HistoricalRaceLoader(first_client, cache_dir=tmp)
            await first.load_season(2025)

            refreshed_races = [_race(1, "Refreshed GP", "charles", "max")]
            second_client = _FakeClient(refreshed_races, season=2026)
            second = HistoricalRaceLoader(second_client, cache_dir=tmp)
            rows = await second.load_season(2025, cache_policy="refresh")

        self.assertEqual(1, len(rows))
        self.assertEqual("Refreshed GP", rows[0]["raceName"])
        self.assertEqual(1, second_client.race_loads)
        self.assertEqual(1, second.cache_stats["refreshes"])
        self.assertEqual(1, second.cache_stats["writes"])

    async def test_historical_loader_merges_local_weekend_evidence_cache(self):
        with TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp) / "evidence"
            evidence_dir.mkdir()
            (evidence_dir / "season_2025_round_02.json").write_text(json.dumps({
                "source": "fastf1_cached_summary",
                "practice_results": [
                    {"driver_id": "max", "best_lap": 71.2, "representative_lap": 71.5, "laps": 12},
                    {"driver_id": "charles", "best_lap": 71.6, "representative_lap": 71.9, "laps": 11},
                ],
            }), encoding="utf-8")
            client = _FakeClient(self.races, season=2026)
            loader = HistoricalRaceLoader(client, cache_dir=Path(tmp) / "season", evidence_cache_dir=evidence_dir)

            rows = await loader.load_season(2025, cache_policy="refresh")

        race_two = next(row for row in rows if row["round"] == "2")
        self.assertEqual(2, len(race_two["PracticeResults"]))
        self.assertEqual(1, loader.cache_stats["evidence_cache"]["files_found"])
        self.assertEqual(1, loader.cache_stats["evidence_cache"]["races_enriched"])
        self.assertEqual(2, loader.cache_stats["evidence_cache"]["practice_rows"])

    async def test_deep_backtest_reports_practice_coverage_from_evidence_cache(self):
        with TemporaryDirectory() as tmp:
            evidence_dir = Path(tmp) / "evidence"
            evidence_dir.mkdir()
            (evidence_dir / "season_2025_round_02.json").write_text(json.dumps({
                "source": "fastf1_cached_summary",
                "practice_results": [
                    {"driver_id": "max", "best_lap": 71.0, "representative_lap": 71.2, "laps": 16},
                    {"driver_id": "charles", "best_lap": 72.0, "representative_lap": 72.4, "laps": 14},
                ],
            }), encoding="utf-8")
            service = F1BacktestService(
                _FakeClient(self.races),
                use_disk_cache=False,
                evidence_cache_dir=str(evidence_dir),
            )

            result = await service.deep_backtest(
                2025,
                2025,
                stages=["pre_weekend", "practice_available"],
                model_ids=["production_v1"],
                include_ablations=True,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(1, result["coverage"]["totals"]["practice_races"])
        self.assertEqual(2, result["coverage"]["totals"]["practice_driver_rows"])
        self.assertIn("avg_telemetry_quality", result["coverage"]["evidence_groups"]["practice"])
        self.assertEqual(1, result["load_strategy"]["historical_cache"]["evidence_cache"]["races_enriched"])
        practice_ablation = next(item for item in result["evidence_ablations"] if item["evidence_group"] == "practice_evidence")
        self.assertIn(practice_ablation["interpretation"], {"evidence_helped", "evidence_hurt_or_overfit", "neutral"})

    async def test_weekend_evidence_cache_writer_creates_compact_openf1_payload(self):
        with TemporaryDirectory() as tmp:
            race = Race(round=2, name="Second GP", circuit="Bahrain International Circuit", country="Bahrain", date="2025-03-02T14:00:00Z")
            drivers = _drivers_for_writer()
            writer = WeekendEvidenceCacheWriter(cache_dir=tmp)

            result = await writer.build_round(
                season=2025,
                race=race,
                drivers=drivers,
                profile={
                    "qualifying": [
                        {"position": 1, "driver_id": "max", "driver_code": "VER", "driver_name": "Max Verstappen", "team": "Red Bull Racing", "q3": "1:18.000"},
                    ],
                    "results": [],
                },
                openf1=_FakeOpenF1Evidence(),
                sessions=["fp1", "race"],
                write=True,
            )

            payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertTrue(result["available"])
        self.assertEqual(2, result["coverage"]["practice_rows"])
        self.assertEqual(1, result["coverage"]["qualifying_rows"])
        self.assertEqual(2, result["coverage"]["race_input_drivers"])
        self.assertEqual("f1-weekend-evidence-cache-v1", payload["schema_version"])
        self.assertEqual("max", payload["PracticeResults"][0]["driver_id"])
        self.assertEqual(23.1, payload["PracticeResults"][0]["representative_sector_1"])
        self.assertEqual(1.0, payload["PracticeResults"][0]["sector_coverage"])
        self.assertEqual(12, payload["PracticeResults"][0]["lap_distribution"]["sample_size"])
        self.assertEqual(["SOFT", "HARD"], payload["RaceInputs"]["drivers"]["max"]["compound_sequence"])
        self.assertEqual(2, payload["RaceInputs"]["drivers"]["max"]["stint_lap_distribution"]["sample_size"])

    async def test_weekend_evidence_cache_writer_uses_light_openf1_practice_profile(self):
        with TemporaryDirectory() as tmp:
            race = Race(round=2, name="Second GP", circuit="Bahrain International Circuit", country="Bahrain", date="2025-03-02T14:00:00Z")
            drivers = _drivers_for_writer()
            writer = WeekendEvidenceCacheWriter(cache_dir=tmp)
            openf1 = _RecordingOpenF1Evidence()

            result = await writer.build_round(
                season=2025,
                race=race,
                drivers=drivers,
                profile={},
                openf1=openf1,
                sessions=["fp1", "race"],
                write=False,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(("laps", "stints"), openf1.calls[0]["endpoints"])
        self.assertEqual(("laps", "positions", "intervals", "stints", "pits", "weather", "race_control"), openf1.calls[1]["endpoints"])

    async def test_weekend_evidence_cache_writer_warns_when_practice_laps_are_empty(self):
        with TemporaryDirectory() as tmp:
            race = Race(round=2, name="Second GP", circuit="Bahrain International Circuit", country="Bahrain", date="2025-03-02T14:00:00Z")
            drivers = _drivers_for_writer()
            writer = WeekendEvidenceCacheWriter(cache_dir=tmp)

            result = await writer.build_round(
                season=2025,
                race=race,
                drivers=drivers,
                profile={},
                openf1=_ZeroLapPracticeOpenF1Evidence(),
                sessions=["fp1"],
                write=True,
            )
            payload = json.loads(Path(result["path"]).read_text(encoding="utf-8"))

        self.assertTrue(result["ok"])
        self.assertEqual(0, result["coverage"]["practice_rows"])
        self.assertEqual("openf1_practice_lap_rows_empty", result["warnings"][0]["code"])
        self.assertEqual("openf1_practice_lap_rows_empty", payload["warnings"][0]["code"])

    async def test_weekend_evidence_cache_writer_preserves_existing_race_inputs_when_refresh_is_empty(self):
        with TemporaryDirectory() as tmp:
            race = Race(round=2, name="Second GP", circuit="Bahrain International Circuit", country="Bahrain", date="2025-03-02T14:00:00Z")
            drivers = _drivers_for_writer()
            writer = WeekendEvidenceCacheWriter(cache_dir=tmp)
            path = writer.path_for(2025, 2)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({
                "RaceInputs": {
                    "source": "openf1_compact_race_inputs",
                    "drivers": {
                        "max": {
                            "driver_id": "max",
                            "driver_number": 1,
                            "compound_sequence": ["SOFT", "HARD"],
                            "stint_lap_distribution": {"sample_size": 2},
                        }
                    },
                }
            }), encoding="utf-8")

            result = await writer.build_round(
                season=2025,
                race=race,
                drivers=drivers,
                profile={},
                openf1=None,
                sessions=["race"],
                write=False,
            )

        self.assertEqual(1, result["coverage"]["race_input_drivers"])
        self.assertEqual("preserved_existing_race_inputs", result["warnings"][-1]["code"])
        self.assertEqual(["SOFT", "HARD"], result["payload"]["RaceInputs"]["drivers"]["max"]["compound_sequence"])

    async def test_weekend_evidence_cache_writer_uses_fastf1_fallback_for_empty_openf1_practice_laps(self):
        with TemporaryDirectory() as tmp:
            race = Race(round=2, name="Second GP", circuit="Bahrain International Circuit", country="Bahrain", date="2025-03-02T14:00:00Z")
            drivers = _drivers_for_writer()
            writer = WeekendEvidenceCacheWriter(cache_dir=tmp)

            result = await writer.build_round(
                season=2025,
                race=race,
                drivers=drivers,
                profile={},
                openf1=_ZeroLapPracticeOpenF1Evidence(),
                fastf1_provider=_FakeFastF1PracticeProvider(),
                fastf1_mode="fallback",
                sessions=["fp1"],
                write=False,
            )

        self.assertTrue(result["ok"])
        self.assertEqual(2, result["coverage"]["practice_rows"])
        self.assertEqual("fastf1_cached_practice_laps", result["payload"]["PracticeResults"][0]["source"])
        self.assertIn("representative_sector_1", result["payload"]["PracticeResults"][0])
        self.assertIn("pace_stability", result["payload"]["PracticeResults"][0])
        self.assertIn("lap_distribution", result["payload"]["PracticeResults"][0])
        self.assertGreater(result["payload"]["PracticeResults"][0]["lap_distribution"]["sample_size"], 0)
        self.assertIn("telemetry_quality", result["payload"]["PracticeResults"][0])
        self.assertGreater(result["payload"]["PracticeResults"][0]["telemetry_quality"], 0.5)
        warning_codes = {item["code"] for item in result["warnings"]}
        self.assertNotIn("openf1_practice_lap_rows_empty", warning_codes)

    async def test_deep_backtest_can_export_full_and_compact_artifacts(self):
        with TemporaryDirectory() as tmp:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                service = F1BacktestService(_FakeClient(self.races), use_disk_cache=False)
                result = await service.deep_backtest(
                    2025,
                    2025,
                    stages=["pre_weekend"],
                    model_ids=["production_v1"],
                    include_races=False,
                    export_artifact=True,
                )
            finally:
                os.chdir(old_cwd)

            artifact = result.get("artifact") or {}
            self.assertTrue(artifact.get("ok"))
            self.assertTrue(Path(artifact["full_path"]).exists())
            self.assertTrue(Path(artifact["compact_path"]).exists())
            self.assertGreater(int(artifact["full_bytes"]), int(artifact["compact_bytes"]))

    async def test_backtest_accepts_stage_and_reports_replay_evidence(self):
        races = [
            self.races[0],
            _race(2, "Second GP", "charles", "max", qualifying_order=["max", "charles"]),
        ]
        service = F1BacktestService(_FakeClient(races))

        result = await service.backtest_race(2025, 2, stage="post_quali")

        self.assertTrue(result["ok"])
        self.assertEqual("post_qualifying", result["stage"])
        self.assertEqual("post_qualifying", result["replay_evidence"]["stage"])
        self.assertTrue(result["replay_evidence"]["backtest"]["target_qualifying_used"])

    async def test_backtest_scores_session_projection_probabilities(self):
        service = F1BacktestService(_FakeClient(self.races))

        result = await service.backtest_race(2025, 2, stage="pre_weekend")

        self.assertTrue(result["ok"])
        self.assertIn("probability_audit", result)
        self.assertIn("calibration_profile", result)
        self.assertIn("session-simulator", result["model_version"])
        self.assertTrue(result["probability_distribution"])
        self.assertIn("grid_position", next(iter(result["component_scores"].values())))

    async def test_current_partial_season_is_rejected_without_flag(self):
        service = F1BacktestService(_FakeClient(self.races, season=2025))

        result = await service.backtest_season(2025)

        self.assertFalse(result["ok"])
        self.assertTrue(result["partial"])

    def test_summary_aggregates_calibration_buckets(self):
        prediction = RacePrediction(
            driver_predictions={
                "max": DriverRacePrediction(driver_id="max", driver_name="Max Verstappen", win_prob=0.70, podium_prob=0.9, top5_prob=0.95),
                "charles": DriverRacePrediction(driver_id="charles", driver_name="Charles Leclerc", win_prob=0.30, podium_prob=0.8, top5_prob=0.9),
            }
        )
        metrics = evaluate_race(prediction, _parse_actual(self.races[0]))
        summary = summarize_races([{
            "metrics": metrics,
            "actual_winner": "max",
            "probability_distribution": [
                {"driver_id": "max", "win_probability": 0.70},
                {"driver_id": "charles", "win_probability": 0.30},
            ],
        }])

        self.assertEqual(1, summary["race_count"])
        self.assertTrue(summary["calibration_buckets"])
        self.assertTrue(summary["top_pick_calibration"])
        self.assertTrue(summary["stage_calibration"])
        self.assertEqual("unknown", summary["stage_calibration"][0]["stage"])

    def test_summary_reports_stage_specific_calibration(self):
        rows = [
            {
                "stage": "pre_weekend",
                "actual_winner": "max",
                "probability_distribution": [{"driver_id": "max", "win_probability": 0.40}],
                "metrics": {"winner_hit": True, "top_probability": 0.40, "winner_probability": 0.40, "log_loss": 0.9},
            },
            {
                "stage": "post_qualifying",
                "actual_winner": "charles",
                "probability_distribution": [{"driver_id": "max", "win_probability": 0.60}],
                "metrics": {"winner_hit": False, "top_probability": 0.60, "winner_probability": 0.20, "log_loss": 1.6},
            },
        ]

        summary = summarize_races(rows)
        by_stage = {item["stage"]: item for item in summary["stage_calibration"]}

        self.assertIn("pre_weekend", by_stage)
        self.assertIn("post_qualifying", by_stage)
        self.assertEqual(1.0, by_stage["pre_weekend"]["winner_accuracy"])
        self.assertEqual(0.0, by_stage["post_qualifying"]["winner_accuracy"])


class _FakeClient:
    backtest_disk_cache = False

    def __init__(self, races, season=2026):
        self.season = season
        self._races = races
        self.race_loads = 0
        self.qualifying_loads = 0

    async def get_historical_race_results(self, season):
        self.race_loads += 1
        return self._races

    async def get_historical_qualifying_results(self, season):
        self.qualifying_loads += 1
        return [
            {"round": race.get("round"), "QualifyingResults": race.get("QualifyingResults") or []}
            for race in self._races
            if race.get("QualifyingResults")
        ]


class _FakeOpenF1Evidence:
    async def get_session_features(self, race, session="race", drivers=None, live=False):
        if session == "fp1":
            return {
                "ok": True,
                "source": "openf1",
                "session": "practice1",
                "session_key": 11,
                "laps": {
                    "drivers": {
                        "1": {
                            "driver_number": 1,
                            "best_lap": 71.2,
                            "representative_lap": 71.5,
                            "median_lap": 71.6,
                            "laps": 12,
                            "compounds": ["SOFT"],
                            "lap_distribution": {
                                "sample_size": 12,
                                "p10": 71.2,
                                "median": 71.6,
                                "p90": 72.1,
                                "spread_p90_p10": 0.9,
                            },
                            "representative_sector_1": 23.1,
                            "representative_sector_2": 24.0,
                            "representative_sector_3": 24.4,
                            "sector_coverage": 1.0,
                        },
                        "16": {
                            "driver_number": 16,
                            "best_lap": 71.6,
                            "representative_lap": 71.9,
                            "median_lap": 72.0,
                            "laps": 10,
                            "compounds": ["MEDIUM"],
                            "lap_distribution": {
                                "sample_size": 10,
                                "p10": 71.6,
                                "median": 72.0,
                                "p90": 72.4,
                                "spread_p90_p10": 0.8,
                            },
                            "representative_sector_1": 23.3,
                            "representative_sector_2": 24.1,
                            "representative_sector_3": 24.5,
                            "sector_coverage": 1.0,
                        },
                    }
                },
                "positions": {"drivers": {}},
                "intervals": {"drivers": {}},
                "stints": {"drivers": {}},
                "pits": {"drivers": {}},
                "weather": {"missing_data": True},
                "race_control": {"missing_data": True},
                "raw_counts": {"laps": 22},
            }
        if session == "race":
            return {
                "ok": True,
                "source": "openf1",
                "session": "race",
                "session_key": 99,
                "laps": {"drivers": {"1": {"lap": 20, "laps": 20, "best_lap": 74.2}, "16": {"lap": 20, "laps": 20, "best_lap": 74.5}}},
                "positions": {"drivers": {"1": {"position": 1}, "16": {"position": 2}}},
                "intervals": {"drivers": {"1": {"gap_to_leader": 0.0, "interval": None}, "16": {"gap_to_leader": 2.4, "interval": 2.4}}},
                "stints": {
                    "drivers": {
                        "1": {
                            "compound": "HARD",
                            "tyre_age": 23,
                            "stints": 2,
                            "compounds": ["SOFT", "HARD"],
                            "compound_sequence": ["SOFT", "HARD"],
                            "avg_stint_laps": 24.0,
                            "max_stint_laps": 31,
                            "final_stint_laps": 31,
                            "stint_lap_distribution": {"sample_size": 2, "median": 24.0, "p90": 30.3},
                            "estimated_tyre_age": 23,
                        },
                        "16": {"compound": "HARD", "tyre_age": 3},
                    }
                },
                "pits": {"drivers": {"1": {"pit_stops": 1}, "16": {"pit_stops": 1}}},
                "weather": {"missing_data": False, "air_temperature": 22.0},
                "race_control": {"missing_data": False, "events": []},
                "raw_counts": {"laps": 40, "positions": 40, "intervals": 40, "stints": 2, "pits": 2},
            }
        return {"ok": False, "source": "openf1", "session": session, "reason": "not_available"}


class _RecordingOpenF1Evidence(_FakeOpenF1Evidence):
    def __init__(self):
        self.calls = []

    async def get_session_features(self, race, session="race", drivers=None, live=False, endpoints=None):
        self.calls.append({"session": session, "endpoints": tuple(endpoints or ())})
        return await super().get_session_features(race, session=session, drivers=drivers, live=live)


class _ZeroLapPracticeOpenF1Evidence:
    async def get_session_features(self, race, session="race", drivers=None, live=False, endpoints=None):
        return {
            "ok": True,
            "source": "openf1",
            "session": session,
            "session_key": 22,
            "laps": {"drivers": {}},
            "stints": {"drivers": {"1": {"compound": "SOFT", "stints": 1}}},
            "raw_counts": {"laps": 0, "stints": 1},
        }


class _FakeFastF1PracticeProvider:
    def laps(self, race, session):
        return [
            SimpleNamespace(driver_code="VER", lap_number=1, lap_time_s=71.2, sector1_s=23.1, sector2_s=24.0, sector3_s=24.1, compound=SimpleNamespace(value="SOFT")),
            SimpleNamespace(driver_code="VER", lap_number=2, lap_time_s=70.9, sector1_s=23.0, sector2_s=23.9, sector3_s=24.0, compound=SimpleNamespace(value="SOFT")),
            SimpleNamespace(driver_code="LEC", lap_number=1, lap_time_s=71.5, sector1_s=23.3, sector2_s=24.1, sector3_s=24.1, compound=SimpleNamespace(value="MEDIUM")),
            SimpleNamespace(driver_code="LEC", lap_number=2, lap_time_s=71.1, sector1_s=23.2, sector2_s=24.0, sector3_s=23.9, compound=SimpleNamespace(value="MEDIUM")),
        ]


def _drivers_for_writer():
    return [
        Driver(id="max", number=1, code="VER", first_name="Max", last_name="Verstappen", nationality="Dutch", team="Red Bull Racing"),
        Driver(id="charles", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari"),
    ]


def _race(round_num, race_name, winner_id, runner_up_id, qualifying_order=None, practice_laps=None):
    drivers = {
        "max": {
            "driverId": "max",
            "permanentNumber": "1",
            "code": "VER",
            "givenName": "Max",
            "familyName": "Verstappen",
            "nationality": "Dutch",
        },
        "charles": {
            "driverId": "charles",
            "permanentNumber": "16",
            "code": "LEC",
            "givenName": "Charles",
            "familyName": "Leclerc",
            "nationality": "Monegasque",
        },
    }
    constructors = {
        "max": {"constructorId": "red_bull", "name": "Red Bull Racing", "nationality": "Austrian"},
        "charles": {"constructorId": "ferrari", "name": "Ferrari", "nationality": "Italian"},
    }
    order = [winner_id, runner_up_id]
    raw = {
        "season": "2025",
        "round": str(round_num),
        "raceName": race_name,
        "date": f"2025-03-{round_num + 1:02d}",
        "time": "14:00:00Z",
        "Circuit": {
            "circuitId": "bahrain",
            "circuitName": "Bahrain International Circuit",
            "Location": {"country": "Bahrain", "locality": "Sakhir", "lat": "26.0325", "long": "50.5106"},
        },
        "Results": [
            {
                "position": str(index + 1),
                "grid": str(index + 1),
                "points": "25" if index == 0 else "18",
                "status": "Finished",
                "Driver": drivers[driver_id],
                "Constructor": constructors[driver_id],
            }
            for index, driver_id in enumerate(order)
        ],
    }
    if qualifying_order:
        raw["QualifyingResults"] = [
            {
                "position": str(index + 1),
                "Driver": drivers[driver_id],
                "Constructor": constructors[driver_id],
                "Q1": "1:20.000",
                "Q2": "1:19.500",
                "Q3": f"1:18.{index:03d}",
            }
            for index, driver_id in enumerate(qualifying_order)
        ]
    if practice_laps:
        raw["PracticeResults"] = [
            {
                "driver_id": driver_id,
                "best_lap": lap,
                "representative_lap": lap + 0.4,
                "laps": 18,
            }
            for driver_id, lap in practice_laps.items()
        ]
    return raw


def _parse_actual(raw):
    rows = []
    for item in raw["Results"]:
        driver = item["Driver"]
        rows.append({
            "driver_id": driver["driverId"],
            "position": int(item["position"]),
            "points": float(item["points"]),
        })
    return rows


if __name__ == "__main__":
    unittest.main()
