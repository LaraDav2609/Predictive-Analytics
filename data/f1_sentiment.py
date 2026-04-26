"""Live F1 context collection and lightweight sentiment scoring."""

from __future__ import annotations

import hashlib
import html
import json
import logging
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx
import redis

from models.f1 import Constructor, Driver

logger = logging.getLogger(__name__)

JOLPICA_BASE_URL = "https://api.jolpi.ca/ergast/f1"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_RSS_FEEDS = [
    "https://www.formula1.com/en/latest/all.xml",
    "https://www.fia.com/rss/news",
    "https://www.fia.com/rss/press-release",
]

POSITIVE_TERMS = {
    "win": 0.18,
    "wins": 0.18,
    "pole": 0.16,
    "podium": 0.14,
    "fastest": 0.10,
    "upgrade": 0.12,
    "improvement": 0.10,
    "extends": 0.08,
    "leads": 0.10,
    "strong": 0.10,
    "breakthrough": 0.12,
    "clean": 0.08,
}

NEGATIVE_TERMS = {
    "penalty": -0.18,
    "grid drop": -0.24,
    "crash": -0.18,
    "dnf": -0.20,
    "retired": -0.16,
    "failure": -0.18,
    "engine": -0.08,
    "investigation": -0.15,
    "stewards": -0.12,
    "reprimand": -0.12,
    "disqualified": -0.30,
    "breach": -0.18,
    "injury": -0.25,
}


async def refresh_f1_sentiment(
    drivers: list[Driver],
    constructors: list[Constructor],
    season: int,
    redis_url: str = DEFAULT_REDIS_URL,
    max_rss_items: int = 20,
) -> dict:
    """Fetch live F1/FIA context, score it, publish it to Redis, and return aggregates."""
    items = await _collect_live_items(season, max_rss_items=max_rss_items)
    if not items:
        items = [_standings_item(drivers, constructors, season)]

    scored = [_score_item(item) for item in items]
    aggregates = aggregate_f1_sentiment(scored, drivers, constructors)
    published = _publish_scored_items(scored, aggregates, redis_url)
    aggregates["published_items"] = published
    aggregates["source_items"] = len(scored)
    return aggregates


def read_f1_sentiment(
    drivers: list[Driver],
    constructors: list[Constructor],
    redis_url: str = DEFAULT_REDIS_URL,
    limit: int = 60,
) -> dict:
    """Read recent F1 sentiment from Redis and aggregate it by driver/team."""
    try:
        r = redis.from_url(redis_url, decode_responses=True)
        composite_raw = r.get("sentiment:composite:latest:f1")
        composite = json.loads(composite_raw) if composite_raw else None
        keys = r.zrevrange("feeditems:category:f1", 0, max(limit - 1, 0))
        items = []
        for key in keys:
            raw = r.get(key)
            if not raw:
                continue
            payload = json.loads(raw)
            if payload.get("SentimentScore") is None:
                payload.update(_score_payload(payload))
            items.append(payload)
        aggregates = aggregate_f1_sentiment(items, drivers, constructors)
        if composite:
            aggregates["composite"] = composite
        return aggregates
    except Exception as ex:
        logger.warning("Failed to read F1 sentiment from Redis: %s", ex)
        return {
            "composite": None,
            "drivers": {},
            "teams": {},
            "items": [],
            "reason": str(ex),
        }


def aggregate_f1_sentiment(items: list[dict], drivers: list[Driver], constructors: list[Constructor]) -> dict:
    driver_aliases = {
        driver.id: _driver_aliases(driver)
        for driver in drivers
    }
    team_aliases = {
        constructor.name.lower(): _team_aliases(constructor)
        for constructor in constructors
    }

    driver_hits: dict[str, list[tuple[float, float]]] = {driver.id: [] for driver in drivers}
    team_hits: dict[str, list[tuple[float, float]]] = {constructor.name.lower(): [] for constructor in constructors}
    global_scores: list[tuple[float, float]] = []

    for item in items:
        text = f"{item.get('Title', '')} {item.get('Content', '')}".lower()
        score = float(item.get("SentimentScore") or 0.0)
        confidence = float(item.get("SentimentConfidence") or 0.45)
        global_scores.append((score, confidence))
        for driver_id, aliases in driver_aliases.items():
            if any(alias and alias in text for alias in aliases):
                driver_hits[driver_id].append((score, confidence))
        for team_key, aliases in team_aliases.items():
            if any(alias and alias in text for alias in aliases):
                team_hits[team_key].append((score, confidence))

    driver_summary = {
        driver_id: _entity_summary(scores)
        for driver_id, scores in driver_hits.items()
        if scores
    }
    team_summary = {
        team_key: _entity_summary(scores)
        for team_key, scores in team_hits.items()
        if scores
    }
    composite_score = _weighted_average(global_scores)
    composite = {
        "MarketCategory": "f1",
        "Composite": round(composite_score, 4),
        "News": round(composite_score, 4),
        "Social": 0.0,
        "OnChain": 0.0,
        "Label": _label(composite_score),
        "Timestamp": int(datetime.now(timezone.utc).timestamp()),
        "TotalItems": len(items),
        "NewsCount": len(items),
        "SocialCount": 0,
        "OnChainCount": 0,
    }
    return {
        "composite": composite,
        "drivers": driver_summary,
        "teams": team_summary,
        "items": items[:12],
    }


async def _collect_live_items(season: int, max_rss_items: int) -> list[dict]:
    items: list[dict] = []
    async with httpx.AsyncClient(headers={"User-Agent": "F1DashboardLiveSentiment/1.0"}, timeout=20.0) as http:
        for path, builder in [
            (f"{season}/driverstandings.json", _build_driver_standings_items),
            (f"{season}/constructorstandings.json", _build_constructor_standings_items),
            (f"{season}.json", _build_calendar_items),
            (f"{season}/results.json?limit=1000", _build_result_items),
        ]:
            try:
                response = await http.get(f"{JOLPICA_BASE_URL}/{path}")
                response.raise_for_status()
                items.extend(builder(response.json(), season))
            except Exception as ex:
                logger.warning("Failed to collect F1 context %s: %s", path, ex)
        for feed in DEFAULT_RSS_FEEDS:
            try:
                response = await http.get(feed)
                response.raise_for_status()
                items.extend(_parse_rss_items(response.text, feed, max_rss_items))
            except Exception as ex:
                logger.warning("Failed to collect F1 RSS %s: %s", feed, ex)
    return items


def _build_driver_standings_items(data: dict, season: int) -> list[dict]:
    standings = _standings_list(data).get("DriverStandings") or []
    if not standings:
        return []
    lines = []
    for entry in standings[:12]:
        driver = _driver_name(entry.get("Driver") or {})
        team = _constructor_name(entry)
        lines.append(f"P{entry.get('position')}: {driver} ({team}) {entry.get('points', 0)} pts, {entry.get('wins', 0)} wins")
    return [_payload(
        f"{season} F1 driver standings live update",
        "Current pilot standings and momentum context. " + "; ".join(lines),
        "jolpica",
        "driver_standings",
    )]


def _build_constructor_standings_items(data: dict, season: int) -> list[dict]:
    standings = _standings_list(data).get("ConstructorStandings") or []
    if not standings:
        return []
    lines = []
    for entry in standings[:10]:
        constructor = (entry.get("Constructor") or {}).get("name") or "Unknown team"
        lines.append(f"P{entry.get('position')}: {constructor} {entry.get('points', 0)} pts, {entry.get('wins', 0)} wins")
    return [_payload(
        f"{season} F1 constructor standings live update",
        "Current team pace, upgrade, and reliability context. " + "; ".join(lines),
        "jolpica",
        "constructor_standings",
    )]


def _build_calendar_items(data: dict, season: int) -> list[dict]:
    now = datetime.now(timezone.utc)
    upcoming = []
    for race in _races(data):
        dt = _race_datetime(race)
        if dt >= now:
            upcoming.append((dt, race))
    if not upcoming:
        return []
    dt, race = sorted(upcoming, key=lambda item: item[0])[0]
    circuit = race.get("Circuit") or {}
    location = circuit.get("Location") or {}
    return [_payload(
        f"Upcoming F1 GP: {race.get('raceName')} at {circuit.get('circuitName')}",
        (
            f"Anticipation context for round {race.get('round')} of the {season} F1 season. "
            f"The next Grand Prix is scheduled for {dt:%Y-%m-%d %H:%M UTC} in "
            f"{location.get('locality')}, {location.get('country')}. Monitor practice, qualifying, sprint, penalties, "
            "weather, upgrades, FIA regulations, and race pace before final prediction."
        ),
        "jolpica",
        "calendar",
        url=race.get("url"),
    )]


def _build_result_items(data: dict, season: int) -> list[dict]:
    items = []
    for race in sorted(_races(data), key=_race_datetime, reverse=True)[:6]:
        results = race.get("Results") or []
        if not results:
            continue
        winner = _driver_name((results[0] or {}).get("Driver") or {})
        lines = []
        incidents = []
        for result in results[:10]:
            driver = _driver_name(result.get("Driver") or {})
            team = (result.get("Constructor") or {}).get("name") or ""
            status = result.get("status") or ""
            lines.append(f"P{result.get('position')}: {driver} ({team}), {result.get('points', 0)} pts, {status}")
            if status and not _finished(status):
                incidents.append(f"{driver} {status}")
        items.append(_payload(
            f"{season} {race.get('raceName')} result: {winner} wins",
            "Recent race performance context. " + "; ".join(lines) + (f". Reliability notes: {'; '.join(incidents)}." if incidents else "."),
            "jolpica",
            "race_results",
            url=race.get("url"),
            published=_race_datetime(race),
        ))
    return items


def _parse_rss_items(xml_text: str, feed_url: str, max_items: int) -> list[dict]:
    root = ElementTree.fromstring(xml_text)
    nodes = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    items = []
    for node in nodes[:max_items]:
        title = _clean(node.findtext("title") or node.findtext("{http://www.w3.org/2005/Atom}title"))
        content = _clean(
            node.findtext("description")
            or node.findtext("{http://www.w3.org/2005/Atom}summary")
            or node.findtext("{http://www.w3.org/2005/Atom}content")
        )
        link = _clean(node.findtext("link")) or None
        published_raw = node.findtext("pubDate") or node.findtext("{http://www.w3.org/2005/Atom}published")
        if not title or not _is_f1_related(title, content):
            continue
        try:
            published = parsedate_to_datetime(published_raw).astimezone(timezone.utc) if published_raw else None
        except Exception:
            published = None
        items.append(_payload(title, content or title, _source_from_feed(feed_url), feed_url, url=link, published=published))
    return items


def _score_item(item: dict) -> dict:
    item.update(_score_payload(item))
    return item


def _score_payload(item: dict) -> dict:
    text = f"{item.get('Title', '')} {item.get('Content', '')}".lower()
    score = 0.0
    hits = 0
    for term, weight in POSITIVE_TERMS.items():
        if term in text:
            score += weight
            hits += 1
    for term, weight in NEGATIVE_TERMS.items():
        if term in text:
            score += weight
            hits += 1
    if "fia" in text and ("regulation" in text or "technical directive" in text):
        hits += 1
        score -= 0.04
    score = max(-1.0, min(1.0, score))
    confidence = min(0.92, 0.42 + hits * 0.08)
    return {
        "SentimentScore": round(score, 4),
        "SentimentLabel": _label(score),
        "SentimentConfidence": round(confidence, 4),
        "SentimentAnalyzer": "f1-live-rules",
        "SentimentAnalyzerUsed": "f1-live-rules",
        "SentimentModel": "f1-news-rules-v1",
    }


def _publish_scored_items(items: list[dict], aggregates: dict, redis_url: str) -> int:
    try:
        r = redis.from_url(redis_url, decode_responses=True)
        index_key = "feeditems:category:f1"
        count = 0
        for item in items:
            key = f"feeditem:f1:{item.get('Source', 'live')}:{_hash(item.get('NativeId') or item.get('Title'))}"
            payload = json.dumps(item, ensure_ascii=False)
            score = _published_timestamp(item)
            r.set(key, payload, ex=30 * 24 * 3600)
            r.zadd(index_key, {key: score})
            count += 1
        r.expire(index_key, 30 * 24 * 3600)
        composite_json = json.dumps(aggregates["composite"], ensure_ascii=False)
        r.set("sentiment:composite:latest:f1", composite_json, ex=3600)
        r.publish("feeds:sentiment:f1", composite_json)
        return count
    except Exception as ex:
        logger.warning("Failed to publish F1 sentiment to Redis: %s", ex)
        return 0


def _standings_item(drivers: list[Driver], constructors: list[Constructor], season: int) -> dict:
    driver_lines = [f"P{d.position}: {d.first_name} {d.last_name} ({d.team}) {d.points} pts" for d in drivers[:10]]
    constructor_lines = [f"P{c.position}: {c.name} {c.points} pts" for c in constructors[:10]]
    return _score_item(_payload(
        f"{season} F1 live standings fallback",
        "Driver standings: " + "; ".join(driver_lines) + ". Constructor standings: " + "; ".join(constructor_lines),
        "dashboard",
        "fallback_standings",
    ))


def _payload(title: str, content: str, source: str, feed: str, url: str | None = None, published: datetime | None = None) -> dict:
    now = datetime.now(timezone.utc)
    published = published or now
    return {
        "Title": title,
        "Content": content,
        "Url": url,
        "Source": source,
        "Feed": feed,
        "MarketCategory": "f1",
        "PublishedUtc": published.isoformat().replace("+00:00", "Z"),
        "Timestamp": now.isoformat().replace("+00:00", "Z"),
        "Keywords": ["f1", "formula 1", "prediction context"],
        "Language": "en",
        "HasFullContent": True,
        "NativeId": url or title,
    }


def _standings_list(data: dict) -> dict:
    lists = (((data.get("MRData") or {}).get("StandingsTable") or {}).get("StandingsLists")) or []
    return lists[0] if lists else {}


def _races(data: dict) -> list[dict]:
    return (((data.get("MRData") or {}).get("RaceTable") or {}).get("Races")) or []


def _race_datetime(race: dict) -> datetime:
    date_raw = race.get("date") or "1970-01-01"
    time_raw = (race.get("time") or "00:00:00Z").replace("Z", "+00:00")
    return datetime.fromisoformat(f"{date_raw}T{time_raw}").astimezone(timezone.utc)


def _driver_name(driver: dict) -> str:
    return f"{driver.get('givenName', '')} {driver.get('familyName', '')}".strip() or driver.get("driverId", "")


def _constructor_name(entry: dict) -> str:
    constructors = entry.get("Constructors") or []
    if constructors:
        return constructors[0].get("name") or ""
    return (entry.get("Constructor") or {}).get("name") or ""


def _driver_aliases(driver: Driver) -> list[str]:
    full_name = f"{driver.first_name} {driver.last_name}".strip().lower()
    aliases = {full_name, driver.last_name.lower(), driver.id.lower(), driver.code.lower()}
    return [alias for alias in aliases if alias]


def _team_aliases(constructor: Constructor) -> list[str]:
    name = constructor.name.lower()
    aliases = {name, constructor.id.lower()}
    if "red bull" in name:
        aliases.add("red bull")
    if "rb" == constructor.id.lower() or "racing bulls" in name:
        aliases.add("racing bulls")
    return [alias for alias in aliases if alias]


def _entity_summary(scores: list[tuple[float, float]]) -> dict:
    score = _weighted_average(scores)
    confidence = sum(conf for _, conf in scores) / max(1, len(scores))
    return {
        "score": round(score, 4),
        "label": _label(score),
        "confidence": round(confidence, 4),
        "mentions": len(scores),
    }


def _weighted_average(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    weight = sum(max(conf, 0.01) for _, conf in values)
    return sum(score * max(conf, 0.01) for score, conf in values) / weight


def _label(score: float) -> str:
    if score > 0.08:
        return "Bullish"
    if score < -0.08:
        return "Bearish"
    return "Neutral"


def _published_timestamp(item: dict) -> int:
    raw = item.get("PublishedUtc") or item.get("Timestamp")
    try:
        return int(datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp())
    except Exception:
        return int(datetime.now(timezone.utc).timestamp())


def _clean(value: str | None) -> str:
    if not value:
        return ""
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _source_from_feed(feed_url: str) -> str:
    host = urlparse(feed_url).netloc.lower()
    if "fia.com" in host:
        return "fia"
    if "formula1.com" in host:
        return "formula1"
    return host or "rss"


def _is_f1_related(title: str, content: str) -> bool:
    text = f"{title} {content}".lower()
    return any(token in text for token in ["f1", "formula 1", "formula one", "grand prix", "fia", "driver", "constructor"])


def _finished(status: str) -> bool:
    normalized = status.lower()
    return normalized == "finished" or normalized.startswith("+") or "lap" in normalized


def _hash(value: Any) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8", "ignore")).hexdigest()
