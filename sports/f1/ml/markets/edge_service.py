"""F1 betting-edge service — the loop-closer that joins model probabilities to
venue market quotes and sizes a stake, reusing the shared betting core
(`common.ml.markets.{edge, kelly, implied}`). The F1 analogue of
`games.csgo.analytics.csgo_predictor.edge_and_stake`, batched over a race's quotes.

ANALYSIS-ONLY: produces edges + recommended fractional-Kelly stakes for review.
It does NOT place orders. The F1 model is still uncalibrated heuristic Monte-Carlo
(see `/api/f1/pipeline/health`), so keep this paper / gated until calibration is
proven.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from common.ml.markets.edge import MarketQuote, compute_edge
from common.ml.markets.implied import implied_from_quote
from common.ml.markets.kelly import size_position

# Model market types the F1 simulator publishes (see markets.mapper / the bridge).
SUPPORTED_MARKETS = ("winner", "podium", "top5", "points", "dnf")


class F1RiskPolicy(BaseModel):
    """Portfolio limits applied after Kelly sizing and before live tradeability."""

    max_stake_usd: float | None = Field(default=None, ge=0.0)
    max_stake_pct_bankroll: float = Field(default=0.05, ge=0.0, le=1.0)
    max_market_exposure_pct_bankroll: float = Field(default=0.10, ge=0.0, le=1.0)
    max_daily_exposure_pct_bankroll: float = Field(default=0.20, ge=0.0, le=1.0)
    max_venue_exposure_pct_bankroll: float = Field(default=0.15, ge=0.0, le=1.0)
    max_daily_loss_pct_bankroll: float = Field(default=0.05, ge=0.0, le=1.0)


class F1ExposureState(BaseModel):
    """Current paper/live exposure snapshot used by ``F1RiskPolicy``."""

    market_exposure_usd: dict[str, float] = Field(default_factory=dict)
    venue_exposure_usd: dict[str, float] = Field(default_factory=dict)
    daily_exposure_usd: float = 0.0
    daily_pnl_usd: float = 0.0


class F1TradeReadinessGate(BaseModel):
    """Live-trading guardrail attached to edge calculations.

    Edge math can still run for analysis, but a row should only be considered
    live-tradeable when validation, calibration, artifacts, and risk controls
    are all explicitly ready.
    """

    live_trading_enabled: bool = False
    risk_manager_active: bool = False
    validation_gate: dict[str, Any] = Field(default_factory=dict)
    artifact_readiness: dict[str, Any] = Field(default_factory=dict)
    calibration_status: dict[str, Any] = Field(default_factory=dict)
    risk_policy: F1RiskPolicy = Field(default_factory=F1RiskPolicy)
    exposure_state: F1ExposureState = Field(default_factory=F1ExposureState)


class F1EdgeQuoteIn(BaseModel):
    """One venue quote (YES side in probability units, 0..1) to price against the model."""
    market_id: str
    venue: str = "kalshi"                                  # "kalshi" | "polymarket"
    market: str                                            # one of SUPPORTED_MARKETS
    entity_code: str                                       # driver code, e.g. "VER"
    yes_bid: float = Field(ge=0.0, le=1.0)
    yes_ask: float = Field(ge=0.0, le=1.0)
    no_bid: float | None = Field(default=None, ge=0.0, le=1.0)
    no_ask: float | None = Field(default=None, ge=0.0, le=1.0)
    fee_bps: float = 0.0


class F1EdgeRequest(BaseModel):
    """POST body for /races/{round}/edges: a bankroll + a batch of venue quotes."""
    bankroll_usd: float = 1000.0
    min_edge_bps: float = 200.0
    shrinkage: float = 0.25
    max_per_market_pct: float = 0.05
    trade_readiness: F1TradeReadinessGate | None = None
    quotes: list[F1EdgeQuoteIn] = Field(default_factory=list)


class F1OrderIntent(BaseModel):
    """Venue-neutral order handoff generated only after edge/readiness/risk gates."""

    idempotency_key: str
    mode: str = "paper"
    venue: str
    market_id: str
    market: str
    entity_code: str
    direction: str
    limit_price: float
    stake_usd: float
    edge_bps: float
    model_prob: float
    created_at: str
    status: str = "ready_for_paper_submit"
    reason: str | None = None
    audit: dict[str, Any] = Field(default_factory=dict)


class F1OrderSubmission(BaseModel):
    """Audit record for a paper/live handoff attempt.

    This is deliberately not a venue execution client. It records what would be
    submitted, enforces idempotency, and blocks live handoff unless the caller
    explicitly enables it.
    """

    idempotency_key: str
    mode: str
    status: str
    venue: str
    market_id: str
    direction: str
    limit_price: float
    stake_usd: float
    submitted_at: str
    reason: str | None = None
    intent: dict[str, Any] = Field(default_factory=dict)


class F1OrderSettlement(BaseModel):
    """Deterministic paper/live settlement record for a submitted binary order."""

    idempotency_key: str
    mode: str
    status: str
    venue: str
    market_id: str
    market: str
    entity_code: str
    direction: str
    limit_price: float
    stake_usd: float
    outcome: bool
    won: bool
    pnl_usd: float
    settled_at: str
    submission: dict[str, Any] = Field(default_factory=dict)


@dataclass
class RaceQuote:
    market_id: str
    venue: str
    market: str
    entity_code: str
    yes_bid: float
    yes_ask: float
    no_bid: float | None
    no_ask: float | None
    fee_bps: float

    @classmethod
    def from_input(cls, q: F1EdgeQuoteIn) -> "RaceQuote":
        return cls(q.market_id, q.venue, q.market, q.entity_code.upper(),
                   q.yes_bid, q.yes_ask, q.no_bid, q.no_ask, q.fee_bps)


def probability_map_from_records(records) -> dict[tuple[str, str], float]:
    """Build {(entity_code, market): probability} from OutcomeProbability records."""
    return {(r.entity_code, r.market): r.probability for r in records}


def build_exposure_state(
    rows: list[dict[str, Any]],
    *,
    active_statuses: set[str] | None = None,
) -> F1ExposureState:
    """Aggregate open/paper positions into the risk-manager exposure snapshot."""

    active = active_statuses or {"open", "pending", "filled", "live", "paper_open", "paper_submitted", "live_handoff_ready"}
    market_exposure: dict[str, float] = {}
    venue_exposure: dict[str, float] = {}
    daily_exposure = 0.0
    daily_pnl = 0.0

    for row in rows or []:
        status = str(row.get("status") or row.get("state") or "open").strip().lower()
        stake = _first_number(row, "open_stake_usd", "stake_usd", "notional_usd", "exposure_usd") or 0.0
        pnl = _first_number(row, "pnl_usd", "realized_pnl", "daily_pnl_usd") or 0.0
        daily_pnl += pnl
        if status not in active or stake <= 0:
            continue
        market_key = _market_key(row)
        venue = str(row.get("venue") or "unknown").strip().lower() or "unknown"
        market_exposure[market_key] = market_exposure.get(market_key, 0.0) + stake
        venue_exposure[venue] = venue_exposure.get(venue, 0.0) + stake
        daily_exposure += stake

    return F1ExposureState(
        market_exposure_usd={key: round(value, 2) for key, value in sorted(market_exposure.items())},
        venue_exposure_usd={key: round(value, 2) for key, value in sorted(venue_exposure.items())},
        daily_exposure_usd=round(daily_exposure, 2),
        daily_pnl_usd=round(daily_pnl, 2),
    )


def build_order_intents(
    edge_rows: list[dict[str, Any]],
    *,
    race_id: str,
    mode: str = "paper",
    created_at: datetime | None = None,
) -> list[F1OrderIntent]:
    """Convert approved edge rows into deterministic paper/live order intents."""

    timestamp = (created_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    normalized_mode = str(mode or "paper").strip().lower()
    intents: list[F1OrderIntent] = []
    for row in edge_rows or []:
        if row.get("tradeable") is not True:
            continue
        direction = str(row.get("direction") or "").upper()
        if direction not in {"YES", "NO"}:
            continue
        stake = _first_number(row, "stake_usd") or 0.0
        if stake <= 0:
            continue
        price = _order_limit_price(row, direction)
        if price is None:
            continue
        key = _idempotency_key(race_id, normalized_mode, row, direction, stake, price)
        intents.append(F1OrderIntent(
            idempotency_key=key,
            mode=normalized_mode,
            venue=str(row.get("venue") or ""),
            market_id=str(row.get("market_id") or ""),
            market=str(row.get("market") or ""),
            entity_code=str(row.get("entity_code") or "").upper(),
            direction=direction,
            limit_price=round(price, 4),
            stake_usd=round(stake, 2),
            edge_bps=round(float(row.get("edge_bps") or 0.0), 1),
            model_prob=round(float(row.get("model_prob") or 0.0), 4),
            created_at=timestamp,
            status="ready_for_live_submit" if normalized_mode == "live" else "ready_for_paper_submit",
            audit={
                "race_id": race_id,
                "market_key": f"{row.get('market')}:{str(row.get('entity_code') or '').upper()}",
                "trade_readiness": row.get("trade_readiness") or {},
                "risk_check": row.get("risk_check") or {},
                "suggested_stake_usd": row.get("suggested_stake_usd"),
            },
        ))
    return intents


def submit_order_intents(
    intents: list[F1OrderIntent | dict[str, Any]],
    *,
    existing_submissions: list[F1OrderSubmission | dict[str, Any]] | None = None,
    allow_live: bool = False,
    submitted_at: datetime | None = None,
) -> list[F1OrderSubmission]:
    """Create idempotent paper/live submission audit records from order intents."""

    timestamp = (submitted_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    seen = {
        str(item.idempotency_key if isinstance(item, F1OrderSubmission) else item.get("idempotency_key"))
        for item in (existing_submissions or [])
        if (item.idempotency_key if isinstance(item, F1OrderSubmission) else item.get("idempotency_key"))
    }
    submissions: list[F1OrderSubmission] = []
    for raw in intents or []:
        intent = raw if isinstance(raw, F1OrderIntent) else F1OrderIntent.model_validate(raw)
        if intent.idempotency_key in seen:
            submissions.append(_submission_record(intent, timestamp, "duplicate_ignored", reason="idempotency_key_seen"))
            continue
        seen.add(intent.idempotency_key)
        if intent.mode == "live" and not allow_live:
            submissions.append(_submission_record(intent, timestamp, "blocked", reason="live_submission_disabled"))
            continue
        status = "live_handoff_ready" if intent.mode == "live" else "paper_submitted"
        submissions.append(_submission_record(intent, timestamp, status))
    return submissions


def load_order_submissions(path: str | Path) -> list[F1OrderSubmission]:
    """Load append-only paper/live submission audit records from JSONL."""

    source = Path(path)
    if not source.exists():
        return []
    records: list[F1OrderSubmission] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(F1OrderSubmission.model_validate(json.loads(line)))
    return records


def append_order_submissions(
    path: str | Path,
    submissions: list[F1OrderSubmission | dict[str, Any]],
) -> dict[str, Any]:
    """Append unique submission records to a JSONL ledger."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = load_order_submissions(target)
    seen = {record.idempotency_key for record in existing}
    parsed = [
        item if isinstance(item, F1OrderSubmission) else F1OrderSubmission.model_validate(item)
        for item in (submissions or [])
    ]
    to_write: list[F1OrderSubmission] = []
    for record in parsed:
        if record.idempotency_key in seen:
            continue
        seen.add(record.idempotency_key)
        to_write.append(record)
    if to_write:
        with target.open("a", encoding="utf-8") as handle:
            for record in to_write:
                handle.write(json.dumps(record.model_dump(), sort_keys=True) + "\n")
    return {
        "ok": True,
        "path": str(target),
        "existing_count": len(existing),
        "input_count": len(parsed),
        "appended_count": len(to_write),
        "duplicate_count": len(parsed) - len(to_write),
        "total_count": len(existing) + len(to_write),
    }


def settle_order_submissions(
    submissions: list[F1OrderSubmission | dict[str, Any]],
    outcomes: dict[str, Any],
    *,
    settled_at: datetime | None = None,
) -> dict[str, Any]:
    """Settle submitted binary orders against realized YES/NO outcomes.

    ``outcomes`` accepts flat keys like ``{"winner:VER": true}`` or nested
    market maps like ``{"winner": {"VER": true}}``. A YES order wins when the
    realized outcome is true; a NO order wins when it is false.
    """

    timestamp = (settled_at or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
    settlements: list[F1OrderSettlement] = []
    skipped: list[dict[str, Any]] = []
    total_staked = 0.0
    realized_pnl = 0.0
    wins = 0

    for raw in submissions or []:
        submission = raw if isinstance(raw, F1OrderSubmission) else F1OrderSubmission.model_validate(raw)
        if submission.status in {"blocked", "duplicate_ignored"}:
            skipped.append({"idempotency_key": submission.idempotency_key, "reason": submission.status})
            continue
        row = submission.model_dump()
        market_key = _market_key(row)
        outcome = _outcome_value(outcomes or {}, market_key, row)
        if outcome is None:
            skipped.append({"idempotency_key": submission.idempotency_key, "reason": "missing_outcome", "market_key": market_key})
            continue
        direction = str(submission.direction or "").upper()
        if direction not in {"YES", "NO"}:
            skipped.append({"idempotency_key": submission.idempotency_key, "reason": "unsupported_direction"})
            continue
        price = max(0.01, min(0.99, float(submission.limit_price)))
        stake = max(0.0, float(submission.stake_usd))
        if stake <= 0:
            skipped.append({"idempotency_key": submission.idempotency_key, "reason": "empty_stake"})
            continue

        yes_won = bool(outcome)
        won = yes_won if direction == "YES" else not yes_won
        pnl = _binary_contract_pnl(price, stake, won)
        intent = submission.intent if isinstance(submission.intent, dict) else {}
        market = str(intent.get("market") or market_key.split(":", 1)[0] or "unknown").lower()
        entity = str(intent.get("entity_code") or market_key.split(":", 1)[-1] or "UNKNOWN").upper()
        settlement = F1OrderSettlement(
            idempotency_key=submission.idempotency_key,
            mode=submission.mode,
            status="settled_win" if won else "settled_loss",
            venue=submission.venue,
            market_id=submission.market_id,
            market=market,
            entity_code=entity,
            direction=direction,
            limit_price=round(price, 4),
            stake_usd=round(stake, 2),
            outcome=yes_won,
            won=won,
            pnl_usd=round(pnl, 2),
            settled_at=timestamp,
            submission=submission.model_dump(),
        )
        settlements.append(settlement)
        total_staked += stake
        realized_pnl += pnl
        wins += 1 if won else 0

    return {
        "ok": True,
        "settlement_count": len(settlements),
        "skipped_count": len(skipped),
        "total_staked": round(total_staked, 2),
        "realized_pnl": round(realized_pnl, 2),
        "roi": round(realized_pnl / total_staked, 4) if total_staked else 0.0,
        "wins": wins,
        "losses": len(settlements) - wins,
        "hit_rate": round(wins / len(settlements), 4) if settlements else 0.0,
        "settlements": [settlement.model_dump() for settlement in settlements],
        "skipped": skipped,
    }


def compute_race_edges(
    prob_map: dict[tuple[str, str], float],
    quotes: list[RaceQuote],
    *,
    bankroll_usd: float = 1000.0,
    min_edge_bps: float = 200.0,
    shrinkage: float = 0.25,
    max_per_market_pct: float = 0.05,
    trade_readiness: F1TradeReadinessGate | dict[str, Any] | None = None,
) -> list[dict]:
    """For each quote, look up the model probability for (entity_code, market),
    de-vig the quote, compute the edge in bps and — if it clears `min_edge_bps` —
    a fractional-Kelly stake. Returns one row per quote, sorted by edge desc.

    Mirrors `compute_edge`'s framing: BUY YES edge = model_prob - best_ask;
    BUY NO edge = best_bid - model_prob (both minus the venue fee)."""
    readiness = evaluate_trade_readiness(trade_readiness)
    rows: list[dict] = []
    for q in quotes:
        model_prob = prob_map.get((q.entity_code, q.market))
        base = {
            "market_id": q.market_id, "venue": q.venue, "market": q.market,
            "entity_code": q.entity_code, "yes_bid": q.yes_bid, "yes_ask": q.yes_ask,
            "fee_bps": q.fee_bps,
        }
        if model_prob is None:
            rows.append({**base, "ok": False, "reason": "no_model_probability",
                         "model_prob": None, "edge_bps": None, "tradeable": False,
                         "direction": None, "stake_fraction": 0.0, "stake_usd": 0.0})
            continue

        implied = implied_from_quote(q.yes_bid, q.yes_ask, q.no_bid, q.no_ask)
        opp = compute_edge(
            model_prob,
            MarketQuote(venue=q.venue, market_id=q.market_id,
                        best_bid=q.yes_bid, best_ask=q.yes_ask, fee_bps=q.fee_bps),
            min_edge_bps=min_edge_bps,
        )
        # Raw (pre-threshold) edges, reported for transparency even when not tradeable.
        yes_edge_bps = (model_prob - q.yes_ask) * 10_000 - q.fee_bps
        no_edge_bps = (q.yes_bid - model_prob) * 10_000 - q.fee_bps
        best_edge = max(yes_edge_bps, no_edge_bps)

        would_trade = opp is not None
        readiness_ready = bool(readiness.get("live_trading_ready", True))
        tradeable = would_trade and readiness_ready
        row = {
            **base, "ok": True,
            "model_prob": round(model_prob, 4),
            "fair_yes": round(implied.fair_yes, 4) if implied.valid else None,
            "overround_bps": round(implied.overround * 10_000, 1) if implied.valid else None,
            "spread_bps": round(implied.spread * 10_000, 1) if implied.valid else None,
            "yes_edge_bps": round(yes_edge_bps, 1),
            "no_edge_bps": round(no_edge_bps, 1),
            "best_direction": "YES" if yes_edge_bps >= no_edge_bps else "NO",
            "best_edge_bps": round(best_edge, 1),
            "would_trade": would_trade,
            "tradeable": tradeable,
            "trade_readiness": readiness,
        }
        if opp is not None and readiness_ready:
            # Size against the chosen side's price (NO = the complement quote).
            if opp.direction == "YES":
                stake = size_position(model_prob, q.yes_ask, bankroll_usd, shrinkage, max_per_market_pct)
            else:
                stake = size_position(1.0 - model_prob, 1.0 - q.yes_bid, bankroll_usd, shrinkage, max_per_market_pct)
            risk = evaluate_order_risk(
                venue=q.venue,
                market_key=f"{q.market}:{q.entity_code}",
                stake_usd=stake.notional_usd,
                bankroll_usd=bankroll_usd,
                trade_readiness=trade_readiness,
            )
            tradeable = bool(risk.get("ok"))
            row["tradeable"] = tradeable
            row["risk_check"] = risk
            if not tradeable:
                row["trade_blocked"] = True
                row["trade_block_reasons"] = risk.get("reasons") or []
            row.update({
                "direction": opp.direction,
                "edge_bps": round(opp.edge_bps, 1),
                "market_implied_prob": round(opp.market_implied_prob, 4),
                "suggested_stake_usd": round(stake.notional_usd, 2),
                "stake_fraction": round(stake.fraction_of_bankroll, 4) if tradeable else 0.0,
                "stake_usd": round(stake.notional_usd, 2) if tradeable else 0.0,
            })
        elif opp is not None:
            row.update({
                "direction": opp.direction,
                "edge_bps": round(opp.edge_bps, 1),
                "market_implied_prob": round(opp.market_implied_prob, 4),
                "trade_blocked": True,
                "trade_block_reasons": readiness.get("reasons") or [],
                "stake_fraction": 0.0,
                "stake_usd": 0.0,
            })
        else:
            row.update({"direction": None, "edge_bps": round(best_edge, 1),
                        "stake_fraction": 0.0, "stake_usd": 0.0})
        rows.append(row)

    rows.sort(key=lambda r: (r.get("edge_bps") if r.get("edge_bps") is not None else -1e9), reverse=True)
    return rows


def evaluate_trade_readiness(readiness: F1TradeReadinessGate | dict[str, Any] | None = None) -> dict[str, Any]:
    """Return a normalized live-trading readiness verdict for market edges."""

    if readiness is None:
        return {
            "mode": "analysis_only",
            "live_trading_ready": True,
            "reasons": [],
        }
    gate = readiness if isinstance(readiness, F1TradeReadinessGate) else F1TradeReadinessGate.model_validate(readiness)
    reasons: list[str] = []
    validation = gate.validation_gate or {}
    artifact = gate.artifact_readiness or {}
    calibration = gate.calibration_status or {}

    if not gate.live_trading_enabled:
        reasons.append("live_trading_disabled")
    if validation:
        validation_passed = bool(validation.get("passed") is True or str(validation.get("status") or "").lower() == "passed")
        if not validation_passed:
            reasons.append("validation_gate_blocked")
    else:
        reasons.append("validation_gate_missing")
    if artifact:
        if artifact.get("live_trading_ready") is not True:
            reasons.append("artifact_not_live_ready")
    else:
        reasons.append("artifact_readiness_missing")
    if calibration:
        if str(calibration.get("status") or "").lower() != "passed":
            reasons.append("calibration_gate_not_passed")
    else:
        reasons.append("calibration_status_missing")
    if not gate.risk_manager_active:
        reasons.append("risk_manager_inactive")

    return {
        "mode": "live" if gate.live_trading_enabled else "paper",
        "live_trading_ready": not reasons,
        "reasons": reasons,
        "risk_manager_active": gate.risk_manager_active,
        "validation_status": validation.get("status"),
        "artifact_live_trading_ready": artifact.get("live_trading_ready"),
        "calibration_status": calibration.get("status"),
    }


def evaluate_order_risk(
    *,
    venue: str,
    market_key: str,
    stake_usd: float,
    bankroll_usd: float,
    trade_readiness: F1TradeReadinessGate | dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Check a proposed order against deterministic F1 exposure limits."""

    if trade_readiness is None:
        return {"ok": True, "reasons": [], "mode": "analysis_only"}
    gate = trade_readiness if isinstance(trade_readiness, F1TradeReadinessGate) else F1TradeReadinessGate.model_validate(trade_readiness)
    if not gate.risk_manager_active:
        return {"ok": False, "reasons": ["risk_manager_inactive"], "mode": "paper"}

    policy = gate.risk_policy
    exposure = gate.exposure_state
    bankroll = max(0.0, float(bankroll_usd))
    stake = max(0.0, float(stake_usd))
    market_existing = float(exposure.market_exposure_usd.get(market_key, 0.0) or 0.0)
    venue_existing = float(exposure.venue_exposure_usd.get(str(venue), 0.0) or 0.0)
    daily_exposure = float(exposure.daily_exposure_usd or 0.0)
    daily_pnl = float(exposure.daily_pnl_usd or 0.0)

    reasons: list[str] = []
    if bankroll <= 0:
        reasons.append("bankroll_invalid")
    if policy.max_stake_usd is not None and stake > policy.max_stake_usd:
        reasons.append("stake_usd_limit_exceeded")
    if bankroll > 0 and stake > bankroll * policy.max_stake_pct_bankroll:
        reasons.append("stake_pct_limit_exceeded")
    if bankroll > 0 and market_existing + stake > bankroll * policy.max_market_exposure_pct_bankroll:
        reasons.append("market_exposure_limit_exceeded")
    if bankroll > 0 and venue_existing + stake > bankroll * policy.max_venue_exposure_pct_bankroll:
        reasons.append("venue_exposure_limit_exceeded")
    if bankroll > 0 and daily_exposure + stake > bankroll * policy.max_daily_exposure_pct_bankroll:
        reasons.append("daily_exposure_limit_exceeded")
    if bankroll > 0 and daily_pnl < 0 and abs(daily_pnl) >= bankroll * policy.max_daily_loss_pct_bankroll:
        reasons.append("daily_loss_limit_exceeded")

    return {
        "ok": not reasons,
        "reasons": reasons,
        "stake_usd": round(stake, 2),
        "bankroll_usd": round(bankroll, 2),
        "market_key": market_key,
        "market_exposure_after_usd": round(market_existing + stake, 2),
        "venue_exposure_after_usd": round(venue_existing + stake, 2),
        "daily_exposure_after_usd": round(daily_exposure + stake, 2),
        "limits": {
            "max_stake_usd": policy.max_stake_usd,
            "max_stake_pct_bankroll": policy.max_stake_pct_bankroll,
            "max_market_exposure_pct_bankroll": policy.max_market_exposure_pct_bankroll,
            "max_daily_exposure_pct_bankroll": policy.max_daily_exposure_pct_bankroll,
            "max_venue_exposure_pct_bankroll": policy.max_venue_exposure_pct_bankroll,
            "max_daily_loss_pct_bankroll": policy.max_daily_loss_pct_bankroll,
        },
    }


def _market_key(row: dict[str, Any]) -> str:
    if row.get("market_key"):
        return str(row["market_key"])
    intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
    market = str(row.get("market") or intent.get("market") or "unknown").strip().lower() or "unknown"
    entity = str(
        row.get("entity_code")
        or row.get("driver_code")
        or row.get("entity")
        or intent.get("entity_code")
        or "unknown"
    ).strip().upper() or "UNKNOWN"
    return f"{market}:{entity}"


def _first_number(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = row.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _binary_contract_pnl(price: float, stake: float, won: bool) -> float:
    price = max(0.01, min(0.99, float(price)))
    stake = max(0.0, float(stake))
    return stake * (1.0 - price) / price if won else -stake


def _outcome_value(outcomes: dict[str, Any], market_key: str, row: dict[str, Any]) -> bool | None:
    keys = {
        market_key,
        market_key.lower(),
        market_key.upper(),
    }
    for key in keys:
        if key in outcomes:
            return _coerce_bool(outcomes.get(key))
    intent = row.get("intent") if isinstance(row.get("intent"), dict) else {}
    market = str(row.get("market") or intent.get("market") or market_key.split(":", 1)[0] or "").strip().lower()
    entity = str(
        row.get("entity_code")
        or row.get("driver_code")
        or row.get("entity")
        or intent.get("entity_code")
        or market_key.split(":", 1)[-1]
        or ""
    ).strip().upper()
    nested = outcomes.get(market)
    if isinstance(nested, dict):
        for key in (entity, entity.lower(), entity.upper()):
            if key in nested:
                return _coerce_bool(nested.get(key))
    return None


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "win", "won"}
    return bool(value)


def _order_limit_price(row: dict[str, Any], direction: str) -> float | None:
    if direction == "YES":
        return _first_number(row, "yes_ask", "market_implied_prob")
    yes_bid = _first_number(row, "yes_bid", "market_implied_prob")
    if yes_bid is None:
        return None
    return max(0.0, min(1.0, 1.0 - yes_bid))


def _idempotency_key(
    race_id: str,
    mode: str,
    row: dict[str, Any],
    direction: str,
    stake_usd: float,
    limit_price: float,
) -> str:
    raw = "|".join([
        "f1-order-intent-v1",
        str(race_id),
        mode,
        str(row.get("venue") or ""),
        str(row.get("market_id") or ""),
        str(row.get("market") or ""),
        str(row.get("entity_code") or "").upper(),
        direction,
        f"{stake_usd:.2f}",
        f"{limit_price:.4f}",
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _submission_record(
    intent: F1OrderIntent,
    submitted_at: str,
    status: str,
    *,
    reason: str | None = None,
) -> F1OrderSubmission:
    return F1OrderSubmission(
        idempotency_key=intent.idempotency_key,
        mode=intent.mode,
        status=status,
        venue=intent.venue,
        market_id=intent.market_id,
        direction=intent.direction,
        limit_price=intent.limit_price,
        stake_usd=intent.stake_usd,
        submitted_at=submitted_at,
        reason=reason,
        intent=intent.model_dump(),
    )
