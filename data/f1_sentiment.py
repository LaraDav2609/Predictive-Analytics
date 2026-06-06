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
    "https://www.motorsport.com/rss/f1/news/",
    "https://www.autosport.com/rss/f1/news/",
    "https://www.racefans.net/feed/",
    "https://racer.com/f1/feed",
    "https://www.the-race.com/category/formula-1/feed/",
    "https://feeds.bbci.co.uk/sport/formula1/rss.xml",
    "https://www.skysports.com/rss/12040",
    "https://www.espn.com/espn/rss/f1/news",
    "https://www.crash.net/rss/f1",
    "https://www.motorsportweek.com/series/single-seater/formula-1/feed/",
    "https://www.planetf1.com/rss",
    "https://racingnews365.com/feed/news.xml",
    "https://www.theguardian.com/sport/formulaone/rss",
    "https://www.independent.co.uk/sport/motor-racing/rss",
    "https://www.mirror.co.uk/sport/formula-1/?service=rss",
    "https://www.f1technical.net/rss/news.xml",
    # Public social/community RSS. These are useful for early paddock chatter,
    # but scoring below deliberately treats them as noisy, capped signals.
    "https://www.reddit.com/r/formula1/.rss",
    "https://www.reddit.com/r/F1Technical/.rss",
    "https://www.reddit.com/r/F1Strategy/.rss",
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
    "title charge": 0.16,
    "front-runner": 0.12,
    "fastest lap": 0.12,
    "new floor": 0.10,
    "upgrade package": 0.12,
    "contract extension": 0.08,
    "recovery": 0.08,
    "confident": 0.08,
    "boost": 0.08,
    "optimistic": 0.07,
    "solves": 0.08,
    "improved": 0.08,
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
    "setback": -0.12,
    "struggles": -0.12,
    "problem": -0.08,
    "power unit": -0.10,
    "gearbox": -0.10,
    "under pressure": -0.10,
    "collision": -0.14,
    "unsafe release": -0.12,
    "warning": -0.08,
    "concern": -0.08,
    "uncertain": -0.07,
    "set to miss": -0.18,
    "struggling": -0.11,
    "banned": -0.12,
}

TOPIC_TERMS = {
    "performance": ["pace", "fastest", "long run", "qualifying", "race pace", "front-running", "podium", "win", "lap time"],
    "reliability": ["engine", "power unit", "gearbox", "hydraulic", "brake", "failure", "reliability"],
    "strategy": ["strategy", "tyre", "tire", "pit stop", "undercut", "overcut", "degradation"],
    "regulation": ["fia", "regulation", "technical directive", "rule", "stewards", "penalty", "scrutineering", "banned"],
    "team": ["upgrade", "package", "floor", "wing", "development", "factory", "team principal", "aero", "aerodynamic"],
    "driver": ["driver", "contract", "confidence", "mistake", "crash", "fitness", "injury", "future"],
    "market": ["seat", "silly season", "driver market", "contract", "switch", "extension"],
    "technical": ["aero", "floor", "wing", "suspension", "power unit", "cooling", "brake", "setup", "technical"],
}

RACE_AWARE_DIMENSIONS = [
    "performance",
    "qualifying_pace",
    "race_pace",
    "track_fit",
    "reliability",
    "strategy",
    "tires",
    "weather",
    "regulation",
    "driver_personal",
    "team_operations",
    "market_noise",
]

DIMENSION_TERMS = {
    "performance": ["pace", "lap time", "fastest", "upgrade", "package", "performance", "front-running"],
    "qualifying_pace": ["qualifying", "quali", "pole", "front row", "grid position", "one-lap", "one lap"],
    "race_pace": ["race pace", "long run", "stint", "race trim", "degradation", "podium", "win"],
    "track_fit": ["track fit", "street circuit", "low speed", "high speed", "overtaking", "monaco", "setup"],
    "reliability": ["reliability", "engine", "power unit", "gearbox", "hydraulic", "brake", "failure", "dnf"],
    "strategy": ["strategy", "pit stop", "undercut", "overcut", "track position", "safety car"],
    "tires": ["tyre", "tire", "compound", "soft", "medium", "hard", "degradation", "stint"],
    "weather": ["weather", "rain", "wet", "wind", "temperature", "forecast", "storm"],
    "regulation": ["fia", "stewards", "penalty", "grid drop", "regulation", "technical directive", "scrutineering"],
    "driver_personal": ["driver", "fitness", "injury", "confidence", "mistake", "contract", "future"],
    "team_operations": ["team principal", "factory", "pit crew", "operations", "strategy", "upgrade", "package"],
    "market_noise": ["rumour", "rumor", "silly season", "driver market", "seat", "switch", "contract"],
}

RACE_KEYWORDS = {
    1: ["australian", "albert park", "melbourne", "australia"],
    2: ["chinese", "shanghai", "china"],
    3: ["japanese", "suzuka", "japan"],
    4: ["miami", "miami international"],
    5: ["canadian", "gilles villeneuve", "montreal", "canada"],
    6: ["monaco", "monte carlo", "circuit de monaco"],
    7: ["spanish", "barcelona", "catalunya", "spain"],
    8: ["austrian", "red bull ring", "spielberg", "austria"],
    9: ["british", "silverstone", "great britain", "uk"],
    10: ["belgian", "spa", "spa-francorchamps", "belgium"],
    11: ["hungarian", "hungaroring", "hungary"],
    12: ["dutch", "zandvoort", "netherlands"],
    13: ["italian", "monza", "italy"],
    14: ["madrid", "spanish grand prix"],
    15: ["azerbaijan", "baku"],
    16: ["singapore", "marina bay"],
    17: ["united states", "cota", "austin", "americas"],
    18: ["mexico", "mexican", "mexico city"],
    19: ["brazil", "brazilian", "interlagos", "sao paulo"],
    20: ["las vegas", "vegas"],
    21: ["qatar", "losail", "lusail"],
    22: ["abu dhabi", "yas marina"],
}

SOURCE_WEIGHTS = {
    "formula1": 1.05,
    "fia": 1.10,
    "motorsport.com": 0.95,
    "autosport.com": 0.95,
    "racefans.net": 0.90,
    "racer.com": 0.88,
    "the-race.com": 0.92,
    "bbc-sport": 0.90,
    "sky-sports": 0.90,
    "espn": 0.86,
    "crash.net": 0.86,
    "motorsportweek.com": 0.88,
    "planetf1.com": 0.84,
    "racingnews365.com": 0.88,
    "theguardian.com": 0.84,
    "independent.co.uk": 0.82,
    "mirror.co.uk": 0.78,
    "f1technical.net": 0.92,
    "reddit-formula1": 0.48,
    "reddit-f1technical": 0.56,
    "reddit-f1strategy": 0.52,
    "jolpica": 1.0,
    "dashboard": 0.75,
}

SOURCE_KIND_WEIGHTS = {
    "official": 1.08,
    "technical": 0.96,
    "news": 0.88,
    "tabloid": 0.62,
    "social": 0.46,
    "data": 0.98,
    "internal": 0.70,
}

SOURCE_CONFIDENCE_CEILINGS = {
    "official": 0.94,
    "technical": 0.84,
    "news": 0.78,
    "tabloid": 0.58,
    "social": 0.50,
    "data": 0.86,
    "internal": 0.62,
}

SOURCE_IMPACT_CAPS = {
    "official": 0.42,
    "technical": 0.34,
    "news": 0.28,
    "tabloid": 0.14,
    "social": 0.10,
    "data": 0.18,
    "internal": 0.12,
}

SOURCE_WEIGHT_CAPS = {
    "official": 2.60,
    "technical": 1.80,
    "news": 1.60,
    "tabloid": 0.70,
    "social": 0.62,
    "data": 1.40,
    "internal": 0.80,
}

RUMOR_TERMS = [
    "rumour",
    "rumor",
    "leak",
    "leaked",
    "insider",
    "unverified",
    "speculation",
    "paddock whisper",
    "reportedly",
    "could switch",
    "set to replace",
]

SIGNIFICANT_TITLE_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "after",
    "before",
    "formula",
    "one",
    "grand",
    "prix",
    "f1",
    "live",
    "latest",
    "news",
    "update",
    "updates",
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

    scored = [_enrich_scored_item(_score_item(item), drivers, constructors, season) for item in _dedupe_items(items)]
    aggregates = aggregate_f1_sentiment(scored, drivers, constructors)
    published = _publish_scored_items(scored, aggregates, redis_url)
    aggregates["published_items"] = published
    aggregates["source_items"] = len(scored)
    aggregates["configured_sources"] = len(DEFAULT_RSS_FEEDS)
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
        raw_items = []
        for key in keys:
            raw = r.get(key)
            if not raw:
                continue
            payload = json.loads(raw)
            if payload.get("SentimentScore") is None or payload.get("SourceKind") is None or payload.get("SentimentModel") != "f1-news-social-governed-rules-v2":
                payload.update(_score_payload(payload))
            if _source_kind(payload) == "social" and not _is_social_prediction_signal(payload.get("Title"), payload.get("Content")):
                continue
            raw_items.append(payload)
        items = [
            _enrich_scored_item(item, drivers, constructors)
            for item in _dedupe_items(raw_items)
        ]
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

    driver_hits: dict[str, list[dict]] = {driver.id: [] for driver in drivers}
    team_hits: dict[str, list[dict]] = {constructor.name.lower(): [] for constructor in constructors}
    global_items: list[dict] = []

    for item in items:
        text = f"{item.get('Title', '')} {item.get('Content', '')}".lower()
        global_items.append(item)
        for driver_id, aliases in driver_aliases.items():
            if any(alias and alias in text for alias in aliases):
                driver_hits[driver_id].append(item)
        for team_key, aliases in team_aliases.items():
            if any(alias and alias in text for alias in aliases):
                team_hits[team_key].append(item)

    global_summary = _entity_summary(global_items)
    team_summary = {
        team_key: _entity_summary(team_items)
        for team_key, team_items in team_hits.items()
    }
    driver_summary = {}
    for driver in drivers:
        team_key = (driver.team or "").lower()
        personal = _entity_summary(driver_hits.get(driver.id) or [])
        team = team_summary.get(team_key) or _entity_summary([])
        personal_score = personal["score"]
        team_score = team["score"]
        overall_score = global_summary["score"]
        blended = 0.55 * personal_score + 0.30 * team_score + 0.15 * overall_score
        mentions = int(personal["mentions"]) + int(team["mentions"])
        confidence = (
            0.55 * float(personal["confidence"])
            + 0.30 * float(team["confidence"])
            + 0.15 * float(global_summary["confidence"])
        )
        driver_summary[driver.id] = {
            "score": round(blended, 4),
            "label": _label(blended),
            "confidence": round(confidence, 4),
            "mentions": mentions,
            "personal_score": personal_score,
            "personal_label": personal["label"],
            "personal_mentions": personal["mentions"],
            "team_score": team_score,
            "team_label": team["label"],
            "team_mentions": team["mentions"],
            "overall_score": overall_score,
            "overall_label": global_summary["label"],
            "source_breakdown": _merge_breakdowns(personal["source_breakdown"], team["source_breakdown"]),
            "topic_scores": _merge_topic_scores(personal["topic_scores"], team["topic_scores"], global_summary["topic_scores"], [0.55, 0.30, 0.15]),
            "latest_items": _latest_items((driver_hits.get(driver.id) or []) + (team_hits.get(team_key) or []), limit=5),
            "win_probability_modifier": round(1.0 + max(-0.10, min(0.10, blended * 0.10)), 4),
            "wdc_probability_modifier": round(1.0 + max(-0.14, min(0.14, blended * 0.14)), 4),
        }

    composite_score = global_summary["score"]
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
        "Sources": global_summary["source_breakdown"],
        "Topics": global_summary["topic_scores"],
    }
    return {
        "composite": composite,
        "drivers": driver_summary,
        "teams": team_summary,
        "items": items[:12],
    }


async def _collect_live_items(season: int, max_rss_items: int) -> list[dict]:
    items: list[dict] = []
    async with httpx.AsyncClient(headers={"User-Agent": "F1DashboardLiveSentiment/1.0"}, timeout=20.0, follow_redirects=True) as http:
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
            or node.findtext("{http://purl.org/rss/1.0/modules/content/}encoded")
            or node.findtext("{http://www.w3.org/2005/Atom}summary")
            or node.findtext("{http://www.w3.org/2005/Atom}content")
        )
        link = _clean(node.findtext("link")) or None
        if not link:
            link_node = node.find("{http://www.w3.org/2005/Atom}link")
            link = _clean(link_node.get("href") if link_node is not None else None) or None
        published_raw = node.findtext("pubDate") or node.findtext("{http://www.w3.org/2005/Atom}published")
        if not title or not _is_f1_related(title, content):
            continue
        try:
            published = parsedate_to_datetime(published_raw).astimezone(timezone.utc) if published_raw else None
        except Exception:
            published = None
        source = _source_from_feed(feed_url)
        if source.startswith("reddit-") and not _is_social_prediction_signal(title, content):
            continue
        payload = _payload(title, content or title, source, feed_url, url=link, published=published)
        payload["Topics"] = _topics_for_text(title, content)
        items.append(payload)
    return items


def _dedupe_items(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in items:
        title_key = re.sub(r"[^a-z0-9]+", " ", str(item.get("Title") or "").lower()).strip()
        title_signature = _title_signature(item.get("Title"))
        url_key = _canonical_url(item.get("Url"))
        native_key = str(item.get("NativeId") or "").strip().lower()
        fingerprint = url_key or native_key or title_signature or title_key or _hash(item)
        if fingerprint in seen or (title_signature and title_signature in seen) or (title_key and title_key in seen):
            continue
        seen.add(fingerprint)
        if title_signature:
            seen.add(title_signature)
        if title_key:
            seen.add(title_key)
        deduped.append(item)
    return deduped


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
    if "wdc" in text or "championship" in text or "title" in text:
        hits += 1
        score += 0.03 if score >= 0 else -0.01
    topics = item.get("Topics") or _topics_for_text(item.get("Title", ""), item.get("Content", ""))
    source_kind = _source_kind(item)
    bias_flags = _bias_flags(text, source_kind)
    rumor_penalty = 0.72 if "rumor_or_unverified" in bias_flags else 1.0
    if "social_source" in bias_flags and abs(score) < 0.08 and hits <= 1:
        score *= 0.50
    score = max(-1.0, min(1.0, score))
    confidence = (0.42 + hits * 0.08) * _source_weight(item) * SOURCE_KIND_WEIGHTS.get(source_kind, 0.86) * rumor_penalty
    confidence = min(SOURCE_CONFIDENCE_CEILINGS.get(source_kind, 0.78), confidence)
    return {
        "SentimentScore": round(score, 4),
        "SentimentLabel": _label(score),
        "SentimentConfidence": round(confidence, 4),
        "SentimentAnalyzer": "f1-live-rules",
        "SentimentAnalyzerUsed": "f1-live-rules",
        "SentimentModel": "f1-news-social-governed-rules-v2",
        "Topics": topics,
        "SourceWeight": round(_source_weight(item), 4),
        "SourceKind": source_kind,
        "BiasFlags": bias_flags,
    }


def _enrich_scored_item(
    item: dict,
    drivers: list[Driver],
    constructors: list[Constructor],
    season: int | None = None,
) -> dict:
    text = f"{item.get('Title', '')} {item.get('Content', '')}".lower()
    driver_ids = [
        driver.id
        for driver in drivers
        if any(alias and alias in text for alias in _driver_aliases(driver))
    ]
    constructor_ids = [
        constructor.id
        for constructor in constructors
        if any(alias and alias in text for alias in _team_aliases(constructor))
    ]
    race_round = _race_round_for_text(text)
    dimensions = _impact_dimensions_for_text(text, item)
    session_scope = _session_scope_for_text(text, dimensions)
    time_decay = _time_decay(item)
    sentiment_score = float(item.get("SentimentScore") or 0.0)
    source_weight = float(item.get("SourceWeight") or _source_weight(item))
    confidence = float(item.get("SentimentConfidence") or 0.45)
    prediction_impact = _prediction_impact_score(sentiment_score, dimensions, source_weight, confidence, time_decay)
    prediction_cap = _prediction_impact_cap(item)
    prediction_impact = round(max(-prediction_cap, min(prediction_cap, prediction_impact)), 4)
    reason = _prediction_reason(item, dimensions, session_scope, prediction_impact)

    enriched = {
        "DriverIds": driver_ids,
        "ConstructorIds": constructor_ids,
        "RaceRound": race_round,
        "SessionScope": session_scope,
        "ImpactDimensions": dimensions,
        "PredictionImpactScore": prediction_impact,
        "PredictionImpactCap": prediction_cap,
        "TimeDecay": time_decay,
        "PredictionImpactReason": reason,
        "RaceAwareSeason": season,
        # JSON-friendly aliases for downstream adapters and diagnostics.
        "driver_ids": driver_ids,
        "constructor_ids": constructor_ids,
        "race_round": race_round,
        "session_scope": session_scope,
        "topics": item.get("Topics") or [],
        "sentiment_score": sentiment_score,
        "prediction_impact_score": prediction_impact,
        "prediction_impact_cap": prediction_cap,
        "source_weight": round(source_weight, 4),
        "source_kind": item.get("SourceKind") or _source_kind(item),
        "bias_flags": item.get("BiasFlags") or _bias_flags(text, _source_kind(item)),
        "confidence_ceiling": SOURCE_CONFIDENCE_CEILINGS.get(_source_kind(item), 0.78),
        "confidence": round(confidence, 4),
        "time_decay": time_decay,
        "reason": reason,
    }
    item.update(enriched)
    return item


def _impact_dimensions_for_text(text: str, item: dict) -> dict:
    sentiment_score = float(item.get("SentimentScore") or 0.0)
    score_sign = 1.0 if sentiment_score >= 0 else -1.0
    dimensions: dict[str, float] = {}
    for dimension, terms in DIMENSION_TERMS.items():
        hits = sum(1 for term in terms if term in text)
        if hits <= 0:
            continue
        local_sign = score_sign
        magnitude = min(1.0, abs(sentiment_score) + 0.08 + hits * 0.04)
        if dimension == "regulation" and any(term in text for term in ["penalty", "grid drop", "stewards", "breach", "disqualified"]):
            local_sign = -1.0
        if dimension == "reliability" and any(term in text for term in ["failure", "dnf", "engine", "gearbox", "hydraulic"]):
            local_sign = -1.0 if sentiment_score <= 0.05 else local_sign
        dimensions[dimension] = round(max(-1.0, min(1.0, local_sign * magnitude)), 4)
    return dimensions


def _session_scope_for_text(text: str, dimensions: dict) -> str:
    if any(term in text for term in ["sprint", "sprint shootout"]):
        return "sprint"
    if "qualifying_pace" in dimensions or any(term in text for term in ["qualifying", "quali", "pole", "front row"]):
        return "qualifying"
    if any(term in text for term in ["race pace", "long run", "stint", "strategy", "pit stop", "safety car"]):
        return "race"
    if any(term in text for term in ["practice", "fp1", "fp2", "fp3"]):
        return "practice"
    return "weekend"


def _prediction_impact_score(
    sentiment_score: float,
    dimensions: dict,
    source_weight: float,
    confidence: float,
    time_decay: float,
) -> float:
    dimensional_strength = sum(abs(float(value or 0.0)) for value in dimensions.values()) / max(1, len(dimensions))
    if not dimensions:
        dimensional_strength = 0.30
    signed_dimension = sum(float(value or 0.0) for value in dimensions.values()) / max(1, len(dimensions))
    raw = (0.55 * sentiment_score + 0.45 * signed_dimension) * dimensional_strength
    weighted = raw * max(0.55, min(1.15, source_weight)) * max(0.20, min(1.0, confidence)) * time_decay
    return round(max(-0.45, min(0.45, weighted)), 4)


def _race_round_for_text(text: str) -> int | None:
    for round_num, keywords in RACE_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return round_num
    return None


def _time_decay(item: dict) -> float:
    raw = item.get("PublishedUtc") or item.get("Timestamp")
    try:
        published = datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return 0.55
    age_hours = max(0.0, (datetime.now(timezone.utc) - published).total_seconds() / 3600.0)
    if age_hours <= 24:
        return 1.0
    if age_hours <= 72:
        return 0.75
    if age_hours <= 168:
        return 0.45
    return 0.20


def _prediction_reason(item: dict, dimensions: dict, session_scope: str, impact: float) -> str:
    title = str(item.get("Title") or "F1 news")
    direction = "raises" if impact > 0.015 else "reduces" if impact < -0.015 else "barely moves"
    topics = ", ".join(list(dimensions.keys())[:3]) or "general context"
    return f"{title} {direction} {session_scope} prediction impact via {topics}."


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
        "Topics": _topics_for_text(title, content),
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
        aliases.add("visa cash app rb")
    if "mercedes" in name:
        aliases.add("silver arrows")
    if "mclaren" in name:
        aliases.add("papaya")
    if "aston" in name:
        aliases.add("aston")
    if "cadillac" in name:
        aliases.add("cadillac f1")
    if "audi" in name or "sauber" in name:
        aliases.update({"audi", "sauber", "kick sauber"})
    return [alias for alias in aliases if alias]


def _entity_summary(items: list[dict]) -> dict:
    scores = _source_capped_scores(items)
    score = _weighted_average(scores)
    confidence = _entity_confidence(items, scores)
    return {
        "score": round(score, 4),
        "label": _label(score),
        "confidence": round(confidence, 4),
        "mentions": len(items),
        "source_breakdown": _source_breakdown(items),
        "topic_scores": _topic_scores(items),
        "latest_items": _latest_items(items),
    }


def _weighted_average(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    weight = sum(max(conf, 0.01) for _, conf in values)
    return sum(score * max(conf, 0.01) for score, conf in values) / weight


def _item_score(item: dict) -> tuple[float, float]:
    score = float(item.get("SentimentScore") or 0.0)
    confidence = float(item.get("SentimentConfidence") or 0.45) * _source_weight(item) * SOURCE_KIND_WEIGHTS.get(_source_kind(item), 0.86)
    return score, max(0.01, min(1.0, confidence))


def _source_capped_scores(items: list[dict]) -> list[tuple[float, float]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for item in items:
        grouped.setdefault(str(item.get("Source") or "unknown"), []).append(_item_score(item))
    capped: list[tuple[float, float]] = []
    for source, values in grouped.items():
        source_kind = _source_kind({"Source": source, "Feed": source})
        cap = SOURCE_WEIGHT_CAPS.get(source_kind, 1.20)
        total = sum(weight for _, weight in values)
        scale = min(1.0, cap / total) if total > 0 else 1.0
        capped.extend((score, weight * scale) for score, weight in values)
    return capped


def _entity_confidence(items: list[dict], scores: list[tuple[float, float]]) -> float:
    if not items:
        return 0.12
    avg_confidence = sum(conf for _, conf in scores) / max(1, len(scores))
    source_count = len({str(item.get("Source") or "unknown") for item in items})
    official_count = sum(1 for item in items if _source_kind(item) in {"official", "technical", "data"})
    social_count = sum(1 for item in items if _source_kind(item) == "social")
    rumor_count = sum(1 for item in items if "rumor_or_unverified" in (item.get("BiasFlags") or item.get("bias_flags") or []))
    diversity = min(0.20, source_count / 5.0 * 0.20)
    official_bonus = min(0.16, official_count / max(1, len(items)) * 0.16)
    social_penalty = min(0.18, social_count / max(1, len(items)) * 0.18)
    rumor_penalty = min(0.16, rumor_count / max(1, len(items)) * 0.16)
    return max(0.08, min(0.92, avg_confidence + diversity + official_bonus - social_penalty - rumor_penalty))


def _source_weight(item: dict) -> float:
    source = str(item.get("Source") or "").lower()
    host = urlparse(str(item.get("Feed") or item.get("Url") or "")).netloc.lower()
    return SOURCE_WEIGHTS.get(source) or SOURCE_WEIGHTS.get(host) or 0.86


def _source_kind(item: dict) -> str:
    source = str(item.get("Source") or "").lower()
    host = urlparse(str(item.get("Feed") or item.get("Url") or "")).netloc.lower()
    text = f"{source} {host}"
    if source in {"formula1", "fia"} or "fia.com" in host or "formula1.com" in host:
        return "official"
    if source == "jolpica" or "ergast" in host:
        return "data"
    if "reddit" in text:
        return "social"
    if source in {"f1technical.net"} or "f1technical" in host:
        return "technical"
    if source in {"mirror.co.uk", "planetf1.com", "crash.net"} or any(token in host for token in ["mirror.co.uk", "planetf1.com", "crash.net"]):
        return "tabloid"
    if source == "dashboard":
        return "internal"
    return "news"


def _bias_flags(text: str, source_kind: str) -> list[str]:
    flags: list[str] = []
    if source_kind == "social":
        flags.append("social_source")
    if source_kind == "tabloid":
        flags.append("low_reliability_source")
    if any(term in text for term in RUMOR_TERMS):
        flags.append("rumor_or_unverified")
    if "confirmed" in text or "official" in text or "fia" in text:
        flags.append("confirmation_language")
    return flags


def _prediction_impact_cap(item: dict) -> float:
    source_kind = _source_kind(item)
    cap = SOURCE_IMPACT_CAPS.get(source_kind, 0.24)
    flags = item.get("BiasFlags") or item.get("bias_flags") or []
    if "rumor_or_unverified" in flags:
        cap *= 0.62
    if "confirmation_language" in flags and source_kind in {"official", "technical", "news"}:
        cap *= 1.08
    return round(max(0.04, min(0.45, cap)), 4)


def _source_breakdown(items: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for item in items:
        source = str(item.get("Source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def _topic_scores(items: list[dict]) -> dict:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for item in items:
        for topic in item.get("Topics") or ["general"]:
            grouped.setdefault(topic, []).append(_item_score(item))
    return {topic: round(_weighted_average(values), 4) for topic, values in grouped.items()}


def _merge_breakdowns(*breakdowns: dict) -> dict:
    merged: dict[str, int] = {}
    for breakdown in breakdowns:
        for key, value in (breakdown or {}).items():
            merged[key] = merged.get(key, 0) + int(value or 0)
    return merged


def _merge_topic_scores(first: dict, second: dict, third: dict, weights: list[float]) -> dict:
    keys = set((first or {}).keys()) | set((second or {}).keys()) | set((third or {}).keys())
    merged = {}
    parts = [first or {}, second or {}, third or {}]
    for key in keys:
        total_weight = sum(weight for part, weight in zip(parts, weights) if key in part)
        if total_weight <= 0:
            continue
        merged[key] = round(sum(float(part.get(key, 0.0)) * weight for part, weight in zip(parts, weights) if key in part) / total_weight, 4)
    return merged


def _latest_items(items: list[dict], limit: int = 4) -> list[dict]:
    ordered = sorted(items, key=_published_timestamp, reverse=True)
    return [
        {
            "title": item.get("Title"),
            "source": item.get("Source"),
            "url": item.get("Url"),
            "published_utc": item.get("PublishedUtc"),
            "score": item.get("SentimentScore"),
            "label": item.get("SentimentLabel"),
            "confidence": item.get("SentimentConfidence") or item.get("confidence"),
            "source_weight": item.get("SourceWeight") or item.get("source_weight"),
            "prediction_impact_score": item.get("PredictionImpactScore") or item.get("prediction_impact_score"),
            "time_decay": item.get("TimeDecay") or item.get("time_decay"),
            "race_round": item.get("RaceRound") or item.get("race_round"),
            "session_scope": item.get("SessionScope") or item.get("session_scope"),
            "driver_ids": item.get("DriverIds") or item.get("driver_ids") or [],
            "constructor_ids": item.get("ConstructorIds") or item.get("constructor_ids") or [],
            "dimensions": item.get("ImpactDimensions") or {},
            "reason": item.get("PredictionImpactReason") or item.get("reason"),
            "topics": item.get("Topics") or [],
        }
        for item in ordered[:limit]
    ]


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
    path = urlparse(feed_url).path.lower()
    if "fia.com" in host:
        return "fia"
    if "formula1.com" in host:
        return "formula1"
    if "bbc.co.uk" in host or "bbci.co.uk" in host:
        return "bbc-sport"
    if "skysports.com" in host:
        return "sky-sports"
    if "espn.com" in host:
        return "espn"
    if "reddit.com" in host:
        if "/r/f1technical" in path:
            return "reddit-f1technical"
        if "/r/f1strategy" in path:
            return "reddit-f1strategy"
        if "/r/formula1" in path:
            return "reddit-formula1"
        return "reddit"
    return host.replace("www.", "") or "rss"


def _is_f1_related(title: str, content: str) -> bool:
    text = f"{title} {content}".lower()
    return any(token in text for token in ["f1", "formula 1", "formula one", "grand prix", "fia", "driver", "constructor"])


def _is_social_prediction_signal(title: str | None, content: str | None) -> bool:
    text = f"{title or ''} {content or ''}".lower()
    if any(token in text for token in [
        "lounge",
        "where can i watch",
        "new to f1",
        "wallpaper",
        "meme",
        "posted my",
        "months ago",
        "cad data",
        "car design",
        "battery deployment",
        "control over the deployment",
        "boost and overtake",
        "new driving4answers video",
    ]):
        return False
    signal_terms = [
        "grand prix",
        "practice",
        "fp1",
        "fp2",
        "fp3",
        "qualifying",
        "quali",
        "sprint",
        "race strategy",
        "race pace",
        "long run",
        "stint",
        "tyre",
        "tire",
        "degradation",
        "pit stop",
        "safety car",
        "fia",
        "regulation",
        "technical directive",
        "penalty",
        "grid drop",
        "upgrade",
        "floor",
        "wing",
        "engine",
        "power unit",
        "reliability",
        "weather",
        "rain",
        "setup",
    ]
    race_terms = [keyword for keywords in RACE_KEYWORDS.values() for keyword in keywords]
    if any(term in text for term in signal_terms + race_terms):
        return True
    team_terms = ["ferrari", "mercedes", "mclaren", "red bull", "aston martin", "williams", "alpine", "haas", "sauber", "racing bulls"]
    team_context_terms = [
        "pace",
        "upgrade",
        "package",
        "practice",
        "qualifying",
        "race",
        "strategy",
        "penalty",
        "regulation",
        "reliability",
        "tyre",
        "tire",
        "setup",
        "floor",
        "wing",
        "engine",
        "power unit",
        "weather",
    ]
    return any(term in text for term in team_terms) and any(term in text for term in team_context_terms)


def _topics_for_text(title: str | None, content: str | None) -> list[str]:
    text = f"{title or ''} {content or ''}".lower()
    topics = [topic for topic, terms in TOPIC_TERMS.items() if any(term in text for term in terms)]
    return topics or ["general"]


def _finished(status: str) -> bool:
    normalized = status.lower()
    return normalized == "finished" or normalized.startswith("+") or "lap" in normalized


def _hash(value: Any) -> str:
    return hashlib.sha1(str(value or "").encode("utf-8", "ignore")).hexdigest()


def _canonical_url(value: str | None) -> str:
    if not value:
        return ""
    parsed = urlparse(str(value).strip().lower())
    if not parsed.netloc:
        return str(value).strip().lower()
    path = parsed.path.rstrip("/")
    return f"{parsed.netloc}{path}"


def _title_signature(value: str | None) -> str:
    tokens = [
        token
        for token in re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).split()
        if len(token) > 2 and token not in SIGNIFICANT_TITLE_STOPWORDS
    ]
    if not tokens:
        return ""
    return " ".join(tokens[:10])
