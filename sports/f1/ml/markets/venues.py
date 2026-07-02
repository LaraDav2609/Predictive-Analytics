"""Venue quote normalization for F1 prediction-market edges.

The edge engine works with ``RaceQuote`` objects in probability units. Venue
payloads are messier: Kalshi commonly quotes cents, while Polymarket snapshots
often use decimal strings. Keep that translation here so pricing logic stays
venue-agnostic.
"""

from __future__ import annotations

import re
from typing import Any

from sports.f1.ml.markets.edge_service import RaceQuote


MARKET_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("dnf", ("dnf", "did not finish", "not finish", "retire", "retirement")),
    ("podium", ("podium", "top 3", "top three")),
    ("top5", ("top 5", "top five", "finish in the top five", "finish top five")),
    ("points", ("points", "score a point", "score points", "top 10", "top ten")),
    ("winner", ("winner", "race winner", "grand prix winner", "to win", " win the ", " wins ")),
)


def normalize_venue_quote(
    raw: dict[str, Any],
    *,
    venue: str,
    market: str,
    entity_code: str,
    market_id: str | None = None,
    fee_bps: float | None = None,
) -> RaceQuote:
    normalized = str(venue or "").strip().lower()
    if normalized == "kalshi":
        return normalize_kalshi_quote(raw, market=market, entity_code=entity_code, market_id=market_id, fee_bps=fee_bps)
    if normalized == "polymarket":
        return normalize_polymarket_quote(raw, market=market, entity_code=entity_code, market_id=market_id, fee_bps=fee_bps)
    raise ValueError(f"unsupported venue '{venue}'")


def normalize_matched_venue_quote(
    raw: dict[str, Any],
    *,
    venue: str,
    driver_aliases: dict[str, Any] | None = None,
    market: str | None = None,
    entity_code: str | None = None,
    market_id: str | None = None,
    fee_bps: float | None = None,
) -> RaceQuote:
    """Infer canonical F1 market/entity fields and normalize the quote."""

    match = match_f1_market(raw, driver_aliases=driver_aliases, market=market, entity_code=entity_code)
    return normalize_venue_quote(
        raw,
        venue=venue,
        market=match["market"],
        entity_code=match["entity_code"],
        market_id=market_id,
        fee_bps=fee_bps,
    )


def match_f1_market(
    raw: dict[str, Any],
    *,
    driver_aliases: dict[str, Any] | None = None,
    market: str | None = None,
    entity_code: str | None = None,
) -> dict[str, Any]:
    """Return canonical ``market`` and ``entity_code`` from venue metadata."""

    text = _market_text(raw)
    canonical_market = _canonical_market(market or str(raw.get("market") or ""), text)
    code = _canonical_entity_code(entity_code or raw.get("entity_code") or raw.get("driver_code"))
    if not code:
        code = _driver_code_from_aliases(text, driver_aliases or {})
    if not canonical_market:
        raise ValueError("unable to infer F1 market type")
    if not code:
        raise ValueError("unable to infer F1 driver code")
    return {
        "market": canonical_market,
        "entity_code": code,
        "matched_text": text,
    }


def normalize_kalshi_quote(
    raw: dict[str, Any],
    *,
    market: str,
    entity_code: str,
    market_id: str | None = None,
    fee_bps: float | None = None,
) -> RaceQuote:
    """Normalize a Kalshi market/orderbook snapshot into probability units."""

    return RaceQuote(
        market_id=_first_text(raw, market_id, "ticker", "market_id", "id"),
        venue="kalshi",
        market=str(market),
        entity_code=str(entity_code).upper(),
        yes_bid=_probability(raw, "yes_bid", "yes_bid_cents", "best_yes_bid", "bid", "bid_cents"),
        yes_ask=_probability(raw, "yes_ask", "yes_ask_cents", "best_yes_ask", "ask", "ask_cents"),
        no_bid=_optional_probability(raw, "no_bid", "no_bid_cents", "best_no_bid"),
        no_ask=_optional_probability(raw, "no_ask", "no_ask_cents", "best_no_ask"),
        fee_bps=_fee_bps(raw, fee_bps),
    )


def normalize_polymarket_quote(
    raw: dict[str, Any],
    *,
    market: str,
    entity_code: str,
    market_id: str | None = None,
    fee_bps: float | None = None,
) -> RaceQuote:
    """Normalize a Polymarket CLOB/market snapshot into probability units."""

    return RaceQuote(
        market_id=_first_text(raw, market_id, "condition_id", "question_id", "market_id", "id", "slug"),
        venue="polymarket",
        market=str(market),
        entity_code=str(entity_code).upper(),
        yes_bid=_probability(raw, "yes_bid", "best_bid", "bestBid", "bid"),
        yes_ask=_probability(raw, "yes_ask", "best_ask", "bestAsk", "ask"),
        no_bid=_optional_probability(raw, "no_bid", "noBid", "best_no_bid"),
        no_ask=_optional_probability(raw, "no_ask", "noAsk", "best_no_ask"),
        fee_bps=_fee_bps(raw, fee_bps),
    )


def normalize_venue_quotes(
    rows: list[dict[str, Any]],
    *,
    venue: str,
    market: str,
    entity_code_key: str = "entity_code",
    market_id_key: str | None = None,
    fee_bps: float | None = None,
) -> list[RaceQuote]:
    quotes: list[RaceQuote] = []
    for row in rows:
        entity = str(row.get(entity_code_key) or row.get("driver_code") or row.get("outcome") or "").strip()
        if not entity:
            continue
        quotes.append(
            normalize_venue_quote(
                row,
                venue=venue,
                market=str(row.get("market") or market),
                entity_code=entity,
                market_id=str(row.get(market_id_key)) if market_id_key and row.get(market_id_key) else None,
                fee_bps=fee_bps,
            )
        )
    return quotes


def normalize_matched_venue_quotes(
    rows: list[dict[str, Any]],
    *,
    venue: str,
    driver_aliases: dict[str, Any] | None = None,
    default_market: str | None = None,
    fee_bps: float | None = None,
) -> list[RaceQuote]:
    quotes: list[RaceQuote] = []
    for row in rows:
        try:
            quotes.append(
                normalize_matched_venue_quote(
                    row,
                    venue=venue,
                    driver_aliases=driver_aliases,
                    market=str(row.get("market") or default_market or "") or None,
                    fee_bps=fee_bps,
                )
            )
        except ValueError:
            continue
    return quotes


def _probability(raw: dict[str, Any], *keys: str) -> float:
    value = _optional_probability(raw, *keys)
    if value is None:
        raise ValueError(f"missing probability field; expected one of {keys}")
    return value


def _optional_probability(raw: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key not in raw or raw.get(key) is None:
            continue
        return _normalize_probability(raw.get(key))
    return None


def _normalize_probability(value: Any) -> float:
    if isinstance(value, str):
        text = value.strip().rstrip("%")
        number = float(text)
        if value.strip().endswith("%"):
            number /= 100.0
    else:
        number = float(value)
    if number > 1.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


def _first_text(raw: dict[str, Any], explicit: str | None, *keys: str) -> str:
    if explicit:
        return str(explicit)
    for key in keys:
        value = raw.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"missing market id; expected one of {keys}")


def _fee_bps(raw: dict[str, Any], explicit: float | None) -> float:
    if explicit is not None:
        return float(explicit)
    for key in ("fee_bps", "taker_fee_bps", "maker_fee_bps"):
        value = raw.get(key)
        if value is not None:
            return float(value)
    return 0.0


def _market_text(raw: dict[str, Any]) -> str:
    fields = (
        "title",
        "question",
        "name",
        "slug",
        "market_slug",
        "ticker",
        "market_id",
        "condition_id",
        "outcome",
        "event_title",
    )
    pieces = [str(raw.get(key) or "") for key in fields]
    return " ".join(piece for piece in pieces if piece).strip().lower().replace("-", " ")


def _canonical_market(explicit: str, text: str) -> str | None:
    normalized = str(explicit or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "win": "winner",
        "winner": "winner",
        "race_winner": "winner",
        "podium": "podium",
        "top_3": "podium",
        "top5": "top5",
        "top_5": "top5",
        "points": "points",
        "top_10": "points",
        "dnf": "dnf",
        "not_finish": "dnf",
    }
    if normalized in aliases:
        return aliases[normalized]
    padded = f" {text} "
    for market, patterns in MARKET_PATTERNS:
        if any(pattern in padded for pattern in patterns):
            return market
    return None


def _canonical_entity_code(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip().upper()
    return text if re.fullmatch(r"[A-Z]{2,4}", text) else None


def _driver_code_from_aliases(text: str, aliases: dict[str, Any]) -> str | None:
    lookup: dict[str, str] = {}
    for key, value in aliases.items():
        code = _canonical_entity_code(key)
        values = value if isinstance(value, (list, tuple, set)) else [value]
        if code:
            lookup[_normalize_alias(key)] = code
            for alias in values:
                lookup[_normalize_alias(alias)] = code
        else:
            mapped = _canonical_entity_code(value)
            if mapped:
                lookup[_normalize_alias(key)] = mapped
    normalized_text = f" {_normalize_alias(text)} "
    matches = [
        (alias, code)
        for alias, code in lookup.items()
        if alias and f" {alias} " in normalized_text
    ]
    if not matches:
        return None
    return sorted(matches, key=lambda item: len(item[0]), reverse=True)[0][1]


def _normalize_alias(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return re.sub(r"\s+", " ", text).strip()
