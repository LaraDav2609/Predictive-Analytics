from sports.f1.predictor.probability.uncertainty import (
    _position_quantiles, assess_conviction, driver_uncertainty,
)


def _dist(positions, podium=0.0, dnf=0.0):
    return {"positions": positions, "podium": podium, "dnf": dnf}


def test_quantiles_from_positions():
    p10, p50, p90 = _position_quantiles({"1": 0.5, "5": 0.5})
    assert p10 == 1
    assert p90 == 5


def test_quantiles_empty():
    assert _position_quantiles({}) == (None, None, None)


def test_band_widens_as_confidence_falls():
    dist = _dist({"1": 0.6, "2": 0.4}, podium=0.9)
    hi = driver_uncertainty(dist, 0.6, confidence=0.9, iterations=2400)
    lo = driver_uncertainty(dist, 0.6, confidence=0.2, iterations=2400)
    assert lo["band_width"] > hi["band_width"]
    assert hi["win_prob_low"] <= 0.6 <= hi["win_prob_high"]


def test_scenarios_and_quantiles_surfaced():
    u = driver_uncertainty(_dist({"1": 0.4, "3": 0.3, "8": 0.3}, podium=0.7, dnf=0.1), 0.4, 0.6, 2400)
    assert u["upside_podium"] == 0.7
    assert u["downside_dnf"] == 0.1
    assert u["finish_best"] is not None and u["finish_worst"] is not None
    assert u["finish_best"] <= u["finish_worst"]


def test_high_conviction_when_confident_and_tight():
    dist = _dist({"1": 0.9, "2": 0.1}, podium=0.99)
    u = driver_uncertainty(dist, 0.9, confidence=0.8, iterations=5000)
    c = assess_conviction(win_prob=0.9, confidence=0.8, uncertainty=u, governance_capped=False)
    assert c["conviction"] == "high"
    assert c["do_not_trade"] is False
    assert c["reasons"] == []


def test_do_not_trade_when_low_confidence_and_capped():
    dist = _dist({"1": 0.2, "5": 0.3, "12": 0.5}, podium=0.3, dnf=0.2)
    u = driver_uncertainty(dist, 0.2, confidence=0.22, iterations=800)
    c = assess_conviction(win_prob=0.2, confidence=0.22, uncertainty=u, governance_capped=True)
    assert c["conviction"] == "low"
    assert c["do_not_trade"] is True
    assert "low_confidence" in c["reasons"]
    assert "evidence_capped" in c["reasons"]
