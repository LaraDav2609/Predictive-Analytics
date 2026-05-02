"""Seed publisher — emits mock F1 probability snapshots to Redis on a schedule.

Used to exercise the dashboard's SignalR push path end-to-end without
depending on the full f1_predictor backtest harness or a live race feed.
The dashboard's F1ProbabilityRedisSubscriber listens on f1:prob:* and
forwards each message to the SignalR group keyed by race_id; this script
generates the messages the subscriber needs.

Usage::

    python -m f1_ml.bridge.seed_publisher --race-id 2026-01-BAHRAIN \\
        --interval 5 --jitter 0.02 --duration 600

Each tick perturbs the previous probabilities by a Gaussian jitter and
publishes the snapshot. Across tens of seconds the dashboard's Workspace
panel should animate as the probabilities drift.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone

import numpy as np

from f1_ml.bridge.redis_publisher import RedisPublisher
from f1_ml.common.types import RaceOutcomeProbability


# Default 20-driver grid mirrored from synthetic_provider so the demo is realistic.
DEFAULT_DRIVERS = [
    ("VER", 0.30), ("NOR", 0.18), ("LEC", 0.13), ("HAM", 0.10), ("RUS", 0.08),
    ("PIA", 0.06), ("SAI", 0.05), ("PER", 0.04), ("ALO", 0.02), ("STR", 0.01),
    ("GAS", 0.01), ("OCO", 0.005), ("HUL", 0.003), ("MAG", 0.002), ("RIC", 0.002),
    ("TSU", 0.001), ("ALB", 0.001), ("SAR", 0.0005), ("BOT", 0.0003), ("ZHO", 0.0002),
]


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def jitter_probabilities(
    current: dict[str, float],
    sigma: float,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Add multiplicative log-normal jitter, then renormalize to a proper distribution."""
    logits = {k: np.log(max(v, 1e-9)) + rng.normal(0, sigma) for k, v in current.items()}
    expd = {k: float(np.exp(v)) for k, v in logits.items()}
    return _normalize(expd)


def derive_podium(winner_probs: dict[str, float]) -> dict[str, float]:
    """Heuristic: P(podium) ≈ min(0.95, 3 × P(winner)). Real model would use joint sim."""
    return {k: min(0.95, v * 3.0) for k, v in winner_probs.items()}


def derive_fl(winner_probs: dict[str, float]) -> dict[str, float]:
    """Heuristic: fastest-lap probability concentrated on top of grid."""
    sorted_codes = sorted(winner_probs, key=lambda k: -winner_probs[k])
    out: dict[str, float] = {}
    for i, k in enumerate(sorted_codes):
        out[k] = max(0.0, 0.4 / (i + 1))
    return _normalize(out)


def derive_dnf(winner_probs: dict[str, float], rng: np.random.Generator) -> dict[str, float]:
    """Per-driver DNF probability ~ 4-8% with small noise; faster cars marginally lower."""
    sorted_codes = sorted(winner_probs, key=lambda k: -winner_probs[k])
    out: dict[str, float] = {}
    for i, k in enumerate(sorted_codes):
        base = 0.03 + 0.04 * (i / max(1, len(sorted_codes) - 1))
        out[k] = max(0.0, min(0.5, base + rng.normal(0, 0.005)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--race-id", default="DEMO-2026-01", help="Race id used in channel + payloads")
    parser.add_argument("--redis-host", default="localhost")
    parser.add_argument("--redis-port", type=int, default=6379)
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between ticks")
    parser.add_argument("--jitter", type=float, default=0.05, help="Per-tick log-normal sigma")
    parser.add_argument("--duration", type=float, default=600.0, help="Total run time in seconds")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-version", default="seed-publisher-v1")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    publisher = RedisPublisher(host=args.redis_host, port=args.redis_port)

    winner = _normalize(dict(DEFAULT_DRIVERS))
    end_time = time.time() + args.duration
    tick = 0
    print(f"[seed_publisher] race_id={args.race_id} interval={args.interval}s "
          f"redis={args.redis_host}:{args.redis_port}")
    while time.time() < end_time:
        winner = jitter_probabilities(winner, args.jitter, rng)
        podium = derive_podium(winner)
        fl = derive_fl(winner)
        dnf = derive_dnf(winner, rng)
        now = datetime.now(timezone.utc)
        records: list[RaceOutcomeProbability] = []
        for code in winner:
            for market, table in (("winner", winner), ("podium", podium),
                                   ("fastest_lap", fl), ("dnf", dnf)):
                records.append(RaceOutcomeProbability(
                    race_id=args.race_id,
                    driver_code=code,
                    market=market,
                    probability=float(table[code]),
                    knowable_as_of=now,
                    model_version=args.model_version,
                ))
        try:
            publisher.publish_batch(records)
        except Exception as e:  # pragma: no cover - opt-in tool
            print(f"[seed_publisher] publish failed (Redis down?): {e}")
            break
        tick += 1
        top = sorted(winner.items(), key=lambda kv: -kv[1])[:3]
        print(f"[seed_publisher] tick {tick:>3} top={', '.join(f'{c} {p*100:.1f}%' for c, p in top)}")
        time.sleep(args.interval)


if __name__ == "__main__":  # pragma: no cover - opt-in tool
    main()
