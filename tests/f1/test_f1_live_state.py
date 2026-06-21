import unittest
from datetime import datetime, timedelta, timezone

from sports.f1.predictor.live import F1LiveSessionEngine
from sports.f1.predictor.live.confidence import build_live_confidence_report
from sports.f1.predictor.live.dynamics import build_live_dynamics
from sports.f1.predictor.live.session_state import build_live_state
from sports.f1.predictor.simulation.session_projection import build_session_projection
from sports.f1.models.f1 import Constructor, Driver, Race


class F1LiveSessionStateTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.drivers = [
            Driver(id="max", number=1, code="VER", first_name="Max", last_name="Verstappen", nationality="Dutch", team="Red Bull Racing", points=25, wins=1, position=1),
            Driver(id="charles", number=16, code="LEC", first_name="Charles", last_name="Leclerc", nationality="Monegasque", team="Ferrari", points=18, wins=0, position=2),
        ]
        self.constructors = [
            Constructor(id="red_bull", name="Red Bull Racing", nationality="Austrian", points=25, wins=1, position=1),
            Constructor(id="ferrari", name="Ferrari", nationality="Italian", points=18, wins=0, position=2),
        ]
        self.race = Race(
            round=1,
            name="Bahrain Grand Prix",
            circuit="Bahrain International Circuit",
            country="Bahrain",
            date=datetime.now(timezone.utc),
        )

    async def test_live_engine_maps_driver_numbers_to_dashboard_ids(self):
        engine = F1LiveSessionEngine(_FakeOpenF1(), ttl_seconds=10)

        state = await engine.get_state(self.race, self.drivers, session="race", force=True)

        self.assertTrue(state["ok"])
        self.assertEqual("live", state["mode"])
        self.assertEqual("max", state["leader"])
        self.assertEqual(1, state["live_positions"]["max"]["position"])
        self.assertEqual(2, state["by_driver_id"]["charles"]["position"])
        self.assertEqual("MEDIUM", state["by_driver_id"]["max"]["compounds"][0])
        self.assertEqual(0.55, state["signals"]["chaos_score"])
        self.assertEqual(2, len(state["track"]["live_positions"]))

    async def test_missing_openf1_returns_safe_unavailable_state(self):
        engine = F1LiveSessionEngine(None)

        state = await engine.get_state(self.race, self.drivers, session="race", force=True)

        self.assertFalse(state["ok"])
        self.assertEqual("unavailable", state["mode"])
        self.assertEqual("openf1_client_unavailable", state["reason"])
        self.assertEqual({}, state["live_positions"])

    async def test_cached_state_marks_stale_after_ttl(self):
        engine = F1LiveSessionEngine(_FakeOpenF1(), ttl_seconds=5)
        state = await engine.get_state(self.race, self.drivers, session="race", force=True)
        key = (self.race.round, "race")
        engine._cache[key] = {
            **state,
            "refreshed_at": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(),
        }

        enriched = engine._with_age(engine._cache[key])

        self.assertTrue(enriched["stale"])
        self.assertEqual("stale", enriched["status"])
        self.assertGreater(enriched["age_seconds"], 5)

    def test_build_live_state_can_represent_historical_openf1_without_locations(self):
        state = build_live_state(
            race=self.race,
            drivers=self.drivers,
            session="race",
            openf1_session=_FakeOpenF1.session_features(),
            track={"source": "estimated", "mode": "estimated", "live_positions": []},
            ttl_seconds=10,
        )

        self.assertTrue(state["ok"])
        self.assertEqual("historical", state["mode"])
        self.assertEqual(1, state["live_positions"]["max"]["position"])

    def test_session_projection_uses_live_positions_instead_of_empty_anchors(self):
        live_state = build_live_state(
            race=self.race,
            drivers=self.drivers,
            session="race",
            openf1_session=_FakeOpenF1.session_features(),
            track=_FakeOpenF1.track_data(),
            ttl_seconds=10,
        )
        prediction = {
            "driver_predictions": {
                "max": {"win_prob": 0.30, "form_score": 0.80, "team_score": 0.80, "reliability_score": 0.90},
                "charles": {"win_prob": 0.30, "form_score": 0.80, "team_score": 0.80, "reliability_score": 0.90},
            }
        }
        features = {
            "completed_races": 0,
            "total_races": 22,
            "drivers": {
                "max": {"form_score": 0.80, "reliability_score": 0.90},
                "charles": {"form_score": 0.80, "reliability_score": 0.90},
            },
            "constructors": {
                "red bull racing": {"team_score": 0.80},
                "ferrari": {"team_score": 0.80},
            },
            "live_state": live_state,
        }

        simulation = build_session_projection(
            race=self.race,
            drivers=self.drivers,
            constructors=self.constructors,
            prediction=prediction,
            features=features,
            qualifying=[],
            sprint=[],
            results=[],
            session="race",
            live=True,
        )

        by_driver = {row["driver_id"]: row for row in simulation["simulations"]}
        self.assertEqual("live_openf1_state", simulation["status"])
        self.assertEqual("live", simulation["live_state"]["mode"])
        self.assertEqual(1, by_driver["max"]["components"]["live_position"])
        self.assertEqual(2, by_driver["charles"]["components"]["live_position"])
        self.assertLessEqual(by_driver["max"]["expected_finish"], by_driver["charles"]["expected_finish"])

    def test_live_dynamics_rewards_leader_and_close_chaser_without_overlocking(self):
        truth = {
            "source_mode": "live",
            "confidence": 0.92,
            "drivers": [
                {"driver_id": "max", "position": 1, "gap_to_leader": "0.000", "interval": "0.000", "representative_lap": 93.2, "compound": "MEDIUM", "tyre_age": 8, "pit_stops": 1},
                {"driver_id": "charles", "position": 2, "gap_to_leader": "+0.800", "interval": "+0.800", "representative_lap": 92.9, "compound": "HARD", "tyre_age": 6, "pit_stops": 1},
            ],
        }

        dynamics = build_live_dynamics(truth, track={"tire_stress": 0.55}, tires={"degradation_rate": 0.55})

        self.assertTrue(dynamics["ok"])
        self.assertGreater(dynamics["drivers"]["max"]["strength_multiplier"], 1.0)
        self.assertGreater(dynamics["drivers"]["charles"]["strength_multiplier"], 0.95)
        self.assertLess(dynamics["drivers"]["max"]["strength_multiplier"], 1.22)

    def test_live_dynamics_penalizes_old_soft_more_than_old_hard(self):
        truth = {
            "source_mode": "live",
            "confidence": 0.90,
            "drivers": [
                {"driver_id": "soft", "position": 1, "gap_to_leader": "0.000", "representative_lap": 93.0, "compound": "SOFT", "tyre_age": 30, "pit_stops": 1},
                {"driver_id": "hard", "position": 2, "gap_to_leader": "+2.000", "representative_lap": 93.0, "compound": "HARD", "tyre_age": 30, "pit_stops": 1},
            ],
        }

        dynamics = build_live_dynamics(truth, track={"tire_stress": 0.75}, tires={"degradation_rate": 0.75})

        self.assertLess(dynamics["drivers"]["soft"]["tyre_delta"], dynamics["drivers"]["hard"]["tyre_delta"])

    def test_estimated_source_discounts_live_dynamics(self):
        base = {
            "drivers": [
                {"driver_id": "max", "position": 1, "gap_to_leader": "0.000", "representative_lap": 93.0, "compound": "MEDIUM", "tyre_age": 8},
                {"driver_id": "charles", "position": 2, "gap_to_leader": "+12.000", "representative_lap": 94.0, "compound": "HARD", "tyre_age": 8},
            ],
        }
        live = build_live_dynamics({**base, "source_mode": "live", "confidence": 0.90})
        estimated = build_live_dynamics({**base, "source_mode": "estimated", "confidence": 0.90})

        self.assertGreater(abs(live["drivers"]["max"]["live_delta"]), abs(estimated["drivers"]["max"]["live_delta"]))

    def test_recording_pending_confidence_stays_low_and_explains_waiting(self):
        report = build_live_confidence_report(
            {
                "source_mode": "recording_pending",
                "confidence": 0.7,
                "drivers": [
                    {"driver_id": "max", "position": 1, "source_mode": "estimated"},
                    {"driver_id": "charles", "position": 2, "source_mode": "estimated"},
                ],
            },
            recorder_status={
                "running": True,
                "fastf1_available": True,
                "signalrcore_available": True,
                "recording_source_mode": "recording_pending",
                "parsed_driver_count": 0,
            },
        )

        self.assertLessEqual(report["confidence"], 0.25)
        self.assertIn("fastf1_timing_rows", report["missing_confidence_groups"])
        self.assertIn("waiting", report["next_best_action"].lower())

    def test_recorded_confident_confidence_can_exceed_recorded_floor(self):
        rows = [
            {
                "driver_id": f"driver_{idx}",
                "position": idx,
                "gap_to_leader": "0.000" if idx == 1 else f"+{idx}.000",
                "interval": "0.000" if idx == 1 else "+1.000",
                "lap": 20,
                "compound": "MEDIUM",
                "tyre_age": 8,
                "pit_stops": 1,
                "source_mode": "recorded_confident",
            }
            for idx in range(1, 21)
        ]

        report = build_live_confidence_report(
            {"source_mode": "recorded_confident", "confidence": 0.82, "drivers": rows},
            recorder_status={"running": True, "fastf1_available": True, "signalrcore_available": True},
        )

        self.assertGreaterEqual(report["confidence_ceiling"], 0.85)
        self.assertGreater(report["confidence"], 0.55)
        self.assertNotIn("usable_live_timing_rows", report["missing_confidence_groups"])


class _FakeOpenF1:
    async def get_session_features(self, race, session, drivers, live=False):
        return self.session_features()

    async def get_track_data(self, race, drivers, session="race", live=False, static_only=False):
        return self.track_data()

    @staticmethod
    def session_features():
        return {
            "ok": True,
            "source": "openf1",
            "session_key": 11,
            "meeting_key": 22,
            "positions": {
                "drivers": {
                    "1": {"driver_number": 1, "driver_code": "VER", "position": 1, "date": "2026-03-01T15:00:03Z"},
                    "16": {"driver_number": 16, "driver_code": "LEC", "position": 2, "date": "2026-03-01T15:00:02Z"},
                }
            },
            "intervals": {
                "drivers": {
                    "1": {"gap_to_leader": "0.000", "interval": "0.000"},
                    "16": {"gap_to_leader": "+2.100", "interval": "+2.100"},
                }
            },
            "laps": {
                "drivers": {
                    "1": {"best_lap": 92.1, "representative_lap": 93.0, "median_lap": 93.2, "laps": 12},
                    "16": {"best_lap": 92.8, "representative_lap": 93.4, "median_lap": 93.5, "laps": 12},
                }
            },
            "stints": {
                "drivers": {
                    "1": {"stints": 1, "compounds": ["MEDIUM"], "avg_stint_laps": 12},
                    "16": {"stints": 1, "compounds": ["HARD"], "avg_stint_laps": 12},
                }
            },
            "pits": {"drivers": {"16": {"pit_stops": 1, "avg_pit_duration": 2.8}}},
            "weather": {"chaos_score": 0.25, "rain_probability": 0.1},
            "race_control": {"chaos_score": 0.55, "events": [{"category": "SafetyCar"}]},
            "raw_counts": {"positions": 24, "laps": 24, "intervals": 24, "weather": 1, "race_control": 1},
        }

    @staticmethod
    def track_data():
        return {
            "ok": True,
            "source": "openf1",
            "mode": "live",
            "session_key": 11,
            "meeting_key": 22,
            "session_name": "Race",
            "path": "M 0 0 L 1 1",
            "points": [{"x": 0, "y": 0}, {"x": 1, "y": 1}],
            "live_positions": [
                {"driver_number": 1, "driver_id": "max", "driver_code": "VER", "x": 0.25, "y": 0.4, "date": "2026-03-01T15:00:03Z"},
                {"driver_number": 16, "driver_id": "charles", "driver_code": "LEC", "x": 0.20, "y": 0.3, "date": "2026-03-01T15:00:02Z"},
            ],
        }


if __name__ == "__main__":
    unittest.main()
