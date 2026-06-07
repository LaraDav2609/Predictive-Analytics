import unittest
from datetime import datetime, timezone

from sports.f1.data.openf1_client import (
    ESTIMATED_TRACK_PATHS,
    _estimated_trace,
    _trace_from_svg,
    _with_track_geometry,
    point_at_progress,
    progress_from_xy,
)
from sports.f1.models.f1 import Race


class F1TrackGeometryTests(unittest.TestCase):
    def test_curated_tracks_have_display_points_and_racing_line(self):
        for key in ESTIMATED_TRACK_PATHS:
            race = Race(
                round=1,
                name=f"{key} Grand Prix",
                circuit=key,
                country="Test",
                date=datetime(2026, 5, 3, tzinfo=timezone.utc),
            )
            track = _estimated_trace(race, "test")
            self.assertTrue(track["display_points"], key)
            self.assertGreaterEqual(len(track["racing_line"]), 120, key)
            self.assertIn("geometry_version", track)
            self.assertIn("markers", track)

    def test_progress_round_trip_stays_near_start_finish(self):
        race = Race(
            round=4,
            name="Miami Grand Prix",
            circuit="Miami International Autodrome",
            country="USA",
            date=datetime(2026, 5, 3, tzinfo=timezone.utc),
        )
        track = _estimated_trace(race, "test")
        start = point_at_progress(track, 0.0)
        finish = point_at_progress(track, 1.0)
        self.assertLess(abs(start["x"] - finish["x"]), 1.0)
        self.assertLess(abs(start["y"] - finish["y"]), 1.0)
        self.assertAlmostEqual(0.0, progress_from_xy(track, start["x"], start["y"]), delta=0.02)

    def test_svg_track_path_supports_relative_smooth_curves(self):
        svg = '<svg><path class="st0" d="M10,10c10,0 20,10 30,10s20,-10 30,0s20,10 30,10s-10,20 -30,20s-40,-10 -60,-20Z"/></svg>'

        track = _trace_from_svg(svg, "albertpark", "test", "local")

        self.assertTrue(track["ok"])
        self.assertEqual("wikimedia_svg", track["source"])
        self.assertEqual("wikimedia_svg", track["mapping_source"])
        self.assertGreaterEqual(len(track["display_points"]), 24)
        self.assertEqual(360, len(track["racing_line"]))
        self.assertEqual(0.88, track["geometry_confidence"])
        for point in track["display_points"]:
            self.assertGreaterEqual(point["x"], 0)
            self.assertLessEqual(point["x"], 1000)
            self.assertGreaterEqual(point["y"], 0)
            self.assertLessEqual(point["y"], 430)

    def test_svg_track_path_supports_lines_quadratics_and_arcs(self):
        svg = '<svg><path class="st0" d="M10,10h30v10q20,20 40,0t40,0a15,15 0 0 1 20,20l-40,20l-80,-20Z"/></svg>'

        track = _trace_from_svg(svg, "madring", "test", "local")

        self.assertTrue(track["ok"])
        self.assertEqual("wikimedia_svg", track["source"])
        self.assertEqual(360, len(track["racing_line"]))
        for point in track["display_points"]:
            self.assertGreaterEqual(point["x"], 0)
            self.assertLessEqual(point["x"], 1000)
            self.assertGreaterEqual(point["y"], 0)
            self.assertLessEqual(point["y"], 430)

    def test_las_vegas_curated_centerline_does_not_use_filled_svg_outline(self):
        track = _with_track_geometry({
            "ok": True,
            "source": "estimated",
            "track_key": "lasvegas",
            "path": ESTIMATED_TRACK_PATHS["lasvegas"],
            "sample_count": 0,
        }, None)

        self.assertEqual("curated_registry", track["mapping_source"])
        self.assertGreaterEqual(len(track["display_points"]), 30)
        self.assertEqual(360, len(track["racing_line"]))
        xs = [point["x"] for point in track["display_points"]]
        ys = [point["y"] for point in track["display_points"]]
        self.assertGreater(max(xs) - min(xs), 700)
        self.assertGreater(max(ys) - min(ys), 250)


if __name__ == "__main__":
    unittest.main()
