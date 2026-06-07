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


def bunch_field(
    cumulative_times: list[float],
    leader_idx: int,
    config: RestartConfig,
    dnf_mask: list[bool] | None = None,
) -> list[float]:
    """Bunch the field behind the leader: every non-DNF car sits
    `bunching_gap_s` behind the car ahead, in current order.

    Args:
        cumulative_times: race time in seconds per car.
        leader_idx: index of the car currently leading (smallest cum time among
            non-DNF cars). Provided explicitly so callers can avoid recomputing.
        config: restart-window configuration.
        dnf_mask: optional list[bool]; True ⇒ that car is out and not bunched.

    Returns:
        New cumulative_times list (does not mutate the input). DNF cars
        retain their pre-bunching values.
    """
    n = len(cumulative_times)
    if n == 0:
        return []
    if not (0 <= leader_idx < n):
        raise IndexError(f"leader_idx {leader_idx} out of range for {n} cars")

    if dnf_mask is None:
        dnf_mask = [False] * n
    if len(dnf_mask) != n:
        raise ValueError("dnf_mask length must match cumulative_times length")

    # Order non-DNF cars by their current cumulative time. The leader's time is
    # preserved; each subsequent non-DNF car is set to leader_time + k * gap.
    times = list(cumulative_times)
    leader_time = times[leader_idx]

    # Indices of non-DNF cars sorted by current cum_time.
    runners = [
        i for i in sorted(range(n), key=lambda j: times[j]) if not dnf_mask[i]
    ]
    if leader_idx not in runners:
        raise ValueError("leader_idx points at a DNF car")

    for k, idx in enumerate(runners):
        times[idx] = leader_time + k * config.bunching_gap_s
    return times


def restart_overtake_modifier(base_prob: float, config: RestartConfig) -> float:
    """Boost overtake probability for the lap after restart, capped at 1.0."""
    if base_prob < 0.0 or base_prob > 1.0:
        raise ValueError("base_prob must be in [0, 1]")
    return float(min(1.0, base_prob * config.restart_overtake_boost))


def pit_loss_under_sc(base_pit_loss_s: float, config: RestartConfig) -> float:
    """Cars pitting during a safety car lose ~half the usual time because the
    field is also slowed; the pit lane delta becomes the primary penalty."""
    if base_pit_loss_s < 0:
        raise ValueError("base_pit_loss_s must be >= 0")
    return float(base_pit_loss_s * config.pit_loss_under_sc_factor)
