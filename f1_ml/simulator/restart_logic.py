"""Safety car / VSC restart — bunch the field, simulate the restart lap.

When SC triggers:
  - All non-DNF cars converge to the leader (gap → ~0.5 s minimum legal gap).
  - SC duration sampled from track-typical distribution.
  - Pit stops during SC are heavily discounted (~half the time penalty).
  - Restart lap has elevated overtake probability (cold tires, brake temps).

VSC behaves similarly but no bunching — just speed limits across the board.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RestartConfig:
    bunching_gap_s: float = 0.5
    pit_loss_under_sc_factor: float = 0.5
    restart_overtake_boost: float = 1.5  # multiplier on overtake probability


def bunch_field(cumulative_times: list[float], leader_idx: int, config: RestartConfig) -> list[float]:
    """Mutate cumulative race times so non-DNF cars sit `bunching_gap_s` behind
    the car ahead, in current order."""
    raise NotImplementedError


def restart_overtake_modifier(base_prob: float, config: RestartConfig) -> float:
    """Boost overtake probability for the lap after restart."""
    return min(1.0, base_prob * config.restart_overtake_boost)
