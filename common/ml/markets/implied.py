"""Market-implied probability — the strongest external baseline for any market.

Prediction-market prices already encode the crowd's probability *plus* a margin
(the vig / overround / hold). To use price as a baseline you must strip that
margin. For a binary market the two outcomes' raw implied probabilities are
normalized to sum to 1 (the "no-vig" fair value).

Domain-agnostic and dependency-free, so f1 / baseball / csgo and the backtester
all share one definition of fair value, spread, and line movement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MarketImplied:
    valid: bool
    mid: float          # (yes_bid + yes_ask) / 2, 0..1
    spread: float       # yes_ask - yes_bid (prob units)
    implied_yes: float  # raw YES prob (ask side), includes vig
    implied_no: float   # raw NO prob (ask side), includes vig
    fair_yes: float     # no-vig YES probability
    fair_no: float      # no-vig NO probability
    overround: float    # implied_yes + implied_no - 1 (the hold / vig)


_INVALID = MarketImplied(False, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _ok(x: float | None) -> bool:
    return x is not None and 0.0 <= x <= 1.0


def implied_from_quote(
    yes_bid: float | None,
    yes_ask: float | None,
    no_bid: float | None = None,
    no_ask: float | None = None,
) -> MarketImplied:
    """Fair value + spread + overround from a YES quote (prob units, 0..1).

    If the NO side isn't quoted independently it's derived from the binary
    complement (no_ask = 1 - yes_bid). For a one-sided binary the overround
    equals the bid/ask spread — that *is* the vig.
    """
    if not (_ok(yes_bid) and _ok(yes_ask)):
        return _INVALID
    if yes_ask < yes_bid:  # crossed / ill-formed book
        return _INVALID

    implied_yes = yes_ask
    implied_no = no_ask if _ok(no_ask) else (1.0 - yes_bid)
    total = implied_yes + implied_no
    if total <= 0.0:
        return _INVALID

    fair_yes = implied_yes / total
    return MarketImplied(
        valid=True,
        mid=(yes_bid + yes_ask) / 2.0,
        spread=yes_ask - yes_bid,
        implied_yes=implied_yes,
        implied_no=implied_no,
        fair_yes=fair_yes,
        fair_no=1.0 - fair_yes,
        overround=total - 1.0,
    )


def implied_from_cents(
    yes_bid_cents: float | None,
    yes_ask_cents: float | None,
    no_bid_cents: float | None = None,
    no_ask_cents: float | None = None,
) -> MarketImplied:
    """Convenience for venues that quote in cents (0..100), e.g. the odds snapshots."""
    def c(v: float | None) -> float | None:
        return None if v is None else v / 100.0
    return implied_from_quote(c(yes_bid_cents), c(yes_ask_cents), c(no_bid_cents), c(no_ask_cents))


def no_vig_two_way(raw_yes: float, raw_no: float) -> tuple[float, float]:
    """Normalize two raw implied probabilities to a no-vig pair summing to 1."""
    total = raw_yes + raw_no
    if total <= 0.0:
        return (0.5, 0.5)
    return (raw_yes / total, raw_no / total)


def line_move(open_fair: float, current_fair: float) -> float:
    """Signed shift in fair YES probability since the opening line (+ = toward YES)."""
    return current_fair - open_fair
