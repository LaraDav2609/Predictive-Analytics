import unittest

from sports.f1.api import f1_routes
from sports.f1.predictor.features.upgrade_impact import (
    build_upgrade_impacts, estimate_upgrade_impact, load_upgrade_calendar,
)


def test_estimate_positive_impact():
    est = estimate_upgrade_impact(pre_scores=[0.0, 0.0], post_scores=[0.05, 0.05, 0.05])
    assert est["applied"] is True
    assert est["impact"] > 0
    assert est["modifier"] > 1.0
    assert est["bounded_impact"] <= 0.06


def test_estimate_insufficient_post():
    est = estimate_upgrade_impact(pre_scores=[0.0, 0.0], post_scores=[0.05])
    assert est["applied"] is False
    assert est["reason"] == "insufficient_post_races"
    assert est["modifier"] == 1.0


def test_estimate_bounded_by_cap():
    est = estimate_upgrade_impact(pre_scores=[0.0], post_scores=[0.5, 0.5], cap=0.06)
    assert est["bounded_impact"] == 0.06   # huge raw impact clamped
    assert est["modifier"] == 1.06


def test_build_impacts():
    calendar = [{"constructor_id": "redbull", "round": 5, "component": "floor"}]
    pace = {"redbull": {1: 0.0, 2: 0.0, 5: 0.05, 6: 0.06, 7: 0.05}}
    impacts = build_upgrade_impacts(calendar, pace, current_round=8)
    assert len(impacts) == 1
    assert impacts[0]["constructor_id"] == "redbull"
    assert impacts[0]["applied"] is True
    assert impacts[0]["impact"] > 0


def test_default_calendar_is_empty_list():
    assert isinstance(load_upgrade_calendar(), list)


class UpgradeImpactRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_with_posted_calendar(self):
        body = {
            "calendar": [{"constructor_id": "redbull", "round": 5, "component": "floor"}],
            "pace_history": {"redbull": {"1": 0.0, "2": 0.0, "5": 0.05, "6": 0.06, "7": 0.05}},
        }
        res = await f1_routes.post_f1_upgrade_impact(8, body)
        self.assertTrue(res["ok"])
        self.assertEqual(res["applied_count"], 1)
        self.assertEqual(res["impacts"][0]["constructor_id"], "redbull")

    async def test_route_empty_calendar_is_honest(self):
        res = await f1_routes.post_f1_upgrade_impact(8, {})
        self.assertTrue(res["ok"])
        self.assertEqual(res["calendar_entries"], 0)
        self.assertEqual(res["applied_count"], 0)


if __name__ == "__main__":
    unittest.main()
