"""Optional adapters over completed F1 sentiment outputs."""

from __future__ import annotations

from typing import Any

from sports.f1.models.f1 import Constructor, Driver, Race
from sports.f1.predictor.scoring.normalization import label_from_score, num

RACE_SESSION_DIMENSIONS = {
    "qualifying": ["qualifying_pace", "track_fit", "regulation", "driver_personal", "team_operations"],
    "sprint": ["qualifying_pace", "race_pace", "track_fit", "tires", "reliability", "regulation"],
    "race": ["race_pace", "strategy", "tires", "weather", "reliability", "regulation", "team_operations"],
    "practice": ["performance", "track_fit", "tires", "weather", "team_operations"],
}

TRACK_QUALI_HEAVY_TERMS = ["monaco", "singapore", "las vegas", "baku", "street"]


class SentimentFeatureProvider:
    def __init__(self, sentiment: dict | None):
        self._sentiment = sentiment or {}

    def get_features(self) -> dict:
        drivers = self._sentiment.get("drivers") or {}
        teams = self._sentiment.get("teams") or {}
        source_items = int(self._sentiment.get("source_items") or self._sentiment.get("published_items") or 0)
        source_count = int(self._sentiment.get("configured_sources") or 0)
        return {
            "drivers": {
                driver_id: {
                    **(item or {}),
                    "coverage_score": _coverage_score(item),
                    "source": "f1_sentiment_driver_adapter",
                    "missing_data": int((item or {}).get("mentions") or 0) == 0,
                }
                for driver_id, item in drivers.items()
            },
            "teams": {
                team_id: {
                    **(item or {}),
                    "coverage_score": _coverage_score(item),
                    "source": "f1_sentiment_constructor_adapter",
                    "missing_data": int((item or {}).get("mentions") or 0) == 0,
                }
                for team_id, item in teams.items()
            },
            "composite": self._sentiment.get("composite") or {},
            "source_items": source_items,
            "configured_sources": source_count,
            "confidence": round(min(0.82, 0.25 + min(1.0, source_items / 80.0) * 0.40 + min(1.0, source_count / 10.0) * 0.17), 4),
            "sentiment_available": bool(drivers or teams),
        }


def driver_sentiment_component(driver_id: str, team: str, sentiment: dict | None) -> dict:
    sentiment = sentiment or {}
    sentiment_drivers = sentiment.get("drivers") or {}
    sentiment_teams = sentiment.get("teams") or {}
    race_impact = ((sentiment.get("race_sentiment_impact") or {}).get("drivers") or {}).get(driver_id) or {}
    global_sentiment = float((sentiment.get("composite") or {}).get("Composite") or 0.0)
    driver_sentiment_data = sentiment_drivers.get(driver_id) or {}
    team_sentiment_data = sentiment_teams.get((team or "").lower()) or {}

    personal_news_score = num(
        driver_sentiment_data.get("personal_news_score", driver_sentiment_data.get("personal_score")),
        driver_sentiment_data.get("score", 0.0),
    )
    team_news_score = num(
        driver_sentiment_data.get("team_news_score", driver_sentiment_data.get("team_score")),
        team_sentiment_data.get("score", 0.0),
    )
    overall_news_score = num(
        driver_sentiment_data.get("overall_news_score", driver_sentiment_data.get("overall_score")),
        global_sentiment,
    )
    sentiment_score = max(
        -1.0,
        min(
            1.0,
            num(
                driver_sentiment_data.get("score"),
                0.55 * personal_news_score + 0.30 * team_news_score + 0.15 * overall_news_score,
            ),
        ),
    )
    news_win_modifier = max(
        0.90,
        min(1.10, num(driver_sentiment_data.get("win_probability_modifier"), 1.0 + sentiment_score * 0.10)),
    )
    race_delta = num(race_impact.get("prediction_delta"), 0.0)
    if race_impact:
        news_win_modifier = max(0.88, min(1.12, news_win_modifier * (1.0 + race_delta)))
    wdc_modifier = max(
        0.86,
        min(1.14, num(driver_sentiment_data.get("wdc_probability_modifier"), 1.0 + sentiment_score * 0.14)),
    )
    return {
        "sentiment_score": sentiment_score,
        "sentiment_label": driver_sentiment_data.get("label") or label_from_score(sentiment_score),
        "sentiment_mentions": int(driver_sentiment_data.get("mentions") or 0),
        "personal_news_score": personal_news_score,
        "personal_news_mentions": int(driver_sentiment_data.get("personal_mentions") or 0),
        "team_news_score": team_news_score,
        "team_news_mentions": int(driver_sentiment_data.get("team_mentions") or 0),
        "overall_news_score": overall_news_score,
        "news_win_modifier": news_win_modifier,
        "wdc_modifier": wdc_modifier,
        "source_breakdown": driver_sentiment_data.get("source_breakdown") or {},
        "topic_scores": driver_sentiment_data.get("topic_scores") or {},
        "coverage_score": _coverage_score(driver_sentiment_data),
        "race_sentiment_impact_score": num(race_impact.get("driver_impact_score"), 0.0),
        "race_sentiment_team_score": num(race_impact.get("team_impact_score"), 0.0),
        "race_sentiment_delta": race_delta,
        "race_sentiment_confidence": num(race_impact.get("confidence"), 0.0),
        "race_sentiment_articles": int(race_impact.get("article_count") or 0),
        "race_sentiment_explanations": race_impact.get("explanations") or [],
    }


def build_race_sentiment_impact(
    race: Race | dict | None,
    drivers: list[Driver],
    constructors: list[Constructor],
    sentiment: dict | None,
    session: str = "race",
) -> dict:
    """Convert broad news sentiment into bounded race/session-specific prediction signals."""
    sentiment = sentiment or {}
    items = list(sentiment.get("items") or [])
    session_key = (session or "race").lower()
    race_round = int(_get(race, "round", 0) or 0)
    race_text = " ".join(
        str(part or "").lower()
        for part in [
            _get(race, "name"),
            _get(race, "circuit"),
            _get(race, "country"),
        ]
    )
    driver_aliases = {driver.id: _driver_aliases(driver) for driver in drivers}
    constructor_aliases = {constructor.id: _team_aliases(constructor) for constructor in constructors}
    constructor_by_name = {(constructor.name or "").lower(): constructor for constructor in constructors}
    team_to_drivers: dict[str, list[Driver]] = {}
    for driver in drivers:
        team_to_drivers.setdefault((driver.team or "").lower(), []).append(driver)

    driver_buckets: dict[str, list[dict]] = {driver.id: [] for driver in drivers}
    constructor_buckets: dict[str, list[dict]] = {constructor.id: [] for constructor in constructors}
    relevant_items = []

    for item in items:
        text = f"{item.get('Title', '')} {item.get('Content', '')}".lower()
        explicit_drivers = set(item.get("DriverIds") or item.get("driver_ids") or [])
        explicit_teams = set(item.get("ConstructorIds") or item.get("constructor_ids") or [])
        if not explicit_drivers:
            explicit_drivers = {
                driver_id
                for driver_id, aliases in driver_aliases.items()
                if any(alias and alias in text for alias in aliases)
            }
        if not explicit_teams:
            explicit_teams = {
                constructor_id
                for constructor_id, aliases in constructor_aliases.items()
                if any(alias and alias in text for alias in aliases)
            }
        if not explicit_drivers and not explicit_teams and not _is_race_relevant(item, race_round, race_text):
            continue

        relevant_items.append(item)
        for driver_id in explicit_drivers:
            if driver_id in driver_buckets:
                driver_buckets[driver_id].append({**item, "_entity_weight": 1.85, "_entity_kind": "driver"})
        for constructor_id in explicit_teams:
            if constructor_id in constructor_buckets:
                constructor_buckets[constructor_id].append({**item, "_entity_weight": 0.82, "_entity_kind": "constructor"})
            constructor = next((c for c in constructors if c.id == constructor_id), None)
            if constructor:
                for driver in team_to_drivers.get((constructor.name or "").lower(), []):
                    driver_buckets[driver.id].append({**item, "_entity_weight": 0.42, "_entity_kind": "team"})

    driver_impacts = {
        driver.id: _entity_impact(
            driver_buckets.get(driver.id) or [],
            session_key=session_key,
            race_round=race_round,
            race_text=race_text,
            label=f"{driver.code or driver.last_name}",
        )
        for driver in drivers
    }
    constructor_impacts = {
        constructor.id: _entity_impact(
            constructor_buckets.get(constructor.id) or [],
            session_key=session_key,
            race_round=race_round,
            race_text=race_text,
            label=constructor.name,
        )
        for constructor in constructors
    }
    for driver in drivers:
        team_key = (driver.team or "").lower()
        constructor = constructor_by_name.get(team_key)
        team_impact = constructor_impacts.get(constructor.id if constructor else "") or {}
        if team_impact:
            driver_impacts[driver.id]["team_impact_score"] = team_impact.get("driver_impact_score", 0.0)
            blended = driver_impacts[driver.id]["driver_impact_score"] + float(team_impact.get("driver_impact_score") or 0.0) * 0.15
            driver_impacts[driver.id]["driver_impact_score"] = round(max(-1.0, min(1.0, blended)), 4)
            driver_impacts[driver.id]["prediction_delta"] = _bounded_delta(driver_impacts[driver.id]["driver_impact_score"], driver_impacts[driver.id]["confidence"])

    non_empty = [item for item in driver_impacts.values() if item.get("article_count")]
    confidence = sum(float(item.get("confidence") or 0.0) for item in non_empty) / max(1, len(non_empty))
    top_impacts = sorted(
        [
            {
                "driver_id": driver.id,
                "driver_code": driver.code,
                "driver_name": f"{driver.first_name} {driver.last_name}".strip(),
                "team": driver.team,
                **driver_impacts[driver.id],
            }
            for driver in drivers
        ],
        key=lambda item: abs(float(item.get("prediction_delta") or 0.0)),
        reverse=True,
    )[:10]
    return {
        "ok": True,
        "sentiment_available": bool(relevant_items),
        "race_round": race_round,
        "session": session_key,
        "source": "f1_race_sentiment_adapter_v1",
        "source_count": len({str(item.get("Source") or "unknown") for item in relevant_items}),
        "article_count": len(relevant_items),
        "confidence": round(confidence, 4) if relevant_items else 0.0,
        "missing_groups": [] if relevant_items else ["race_specific_sentiment"],
        "drivers": driver_impacts,
        "constructors": constructor_impacts,
        "top_impacts": top_impacts,
        "explanations": _top_explanations(top_impacts),
    }


def _entity_impact(items: list[dict], session_key: str, race_round: int, race_text: str, label: str) -> dict:
    if not items:
        return _neutral_entity_impact()
    weighted_scores: list[tuple[float, float]] = []
    topic_scores: dict[str, list[tuple[float, float]]] = {}
    source_breakdown: dict[str, int] = {}
    articles = []
    for item in items:
        dimensions = item.get("ImpactDimensions") or item.get("dimensions") or {}
        session_weight = _session_weight(item, session_key)
        race_weight = _race_weight(item, race_round, race_text)
        entity_weight = float(item.get("_entity_weight") or 1.0)
        source_weight = float(item.get("SourceWeight") or item.get("source_weight") or 0.86)
        source_kind = str(item.get("SourceKind") or item.get("source_kind") or _source_kind(item)).lower()
        confidence = float(item.get("SentimentConfidence") or item.get("confidence") or 0.45)
        time_decay = float(item.get("TimeDecay") or item.get("time_decay") or 0.45)
        impact = float(item.get("PredictionImpactScore") or item.get("prediction_impact_score") or item.get("SentimentScore") or 0.0)
        impact_cap = float(item.get("PredictionImpactCap") or item.get("prediction_impact_cap") or _impact_cap(source_kind, item))
        impact = max(-impact_cap, min(impact_cap, impact))
        dimension_weight = _dimension_weight(dimensions, session_key, race_text)
        source_kind_weight = _source_kind_weight(source_kind, item)
        weight = max(0.02, entity_weight * session_weight * race_weight * source_weight * source_kind_weight * confidence * time_decay * dimension_weight)
        weighted_scores.append((impact, weight))
        for topic, value in dimensions.items():
            topic_scores.setdefault(topic, []).append((float(value or 0.0), weight))
        source = str(item.get("Source") or "unknown")
        source_breakdown[source] = source_breakdown.get(source, 0) + 1
        articles.append(_article_summary(item, impact, weight))
    score = _weighted_average(weighted_scores)
    social_ratio = sum(1 for item in items if _source_kind(item) == "social") / max(1, len(items))
    rumor_ratio = sum(1 for item in items if "rumor_or_unverified" in (item.get("BiasFlags") or item.get("bias_flags") or [])) / max(1, len(items))
    confidence = min(
        0.92,
        0.22
        + min(1.0, len(items) / 5.0) * 0.22
        + min(1.0, len(source_breakdown) / 4.0) * 0.22
        + sum(weight for _, weight in weighted_scores) / max(1, len(items)) * 0.28
        - social_ratio * 0.14
        - rumor_ratio * 0.12,
    )
    confidence = max(0.05, confidence)
    topic_summary = {topic: round(_weighted_average(values), 4) for topic, values in topic_scores.items()}
    return {
        "driver_impact_score": round(score, 4),
        "team_impact_score": 0.0,
        "qualifying_impact": _topic_blend(topic_summary, ["qualifying_pace", "track_fit", "regulation"]),
        "race_impact": _topic_blend(topic_summary, ["race_pace", "strategy", "tires", "weather"]),
        "reliability_impact": _topic_blend(topic_summary, ["reliability"]),
        "regulation_impact": _topic_blend(topic_summary, ["regulation"]),
        "confidence": round(confidence, 4),
        "source_coverage": round(min(1.0, len(source_breakdown) / 4.0), 4),
        "source_breakdown": source_breakdown,
        "article_count": len(items),
        "prediction_delta": _bounded_delta(score, confidence),
        "topic_scores": topic_summary,
        "articles": sorted(articles, key=lambda item: abs(item.get("impact", 0.0) * item.get("weight", 0.0)), reverse=True)[:4],
        "explanations": _impact_explanations(label, score, topic_summary, articles),
        "missing_data": False,
    }


def _neutral_entity_impact() -> dict:
    return {
        "driver_impact_score": 0.0,
        "team_impact_score": 0.0,
        "qualifying_impact": 0.0,
        "race_impact": 0.0,
        "reliability_impact": 0.0,
        "regulation_impact": 0.0,
        "confidence": 0.0,
        "source_coverage": 0.0,
        "source_breakdown": {},
        "article_count": 0,
        "prediction_delta": 0.0,
        "topic_scores": {},
        "articles": [],
        "explanations": [],
        "missing_data": True,
    }


def _session_weight(item: dict, session_key: str) -> float:
    scope = str(item.get("SessionScope") or item.get("session_scope") or "weekend").lower()
    if scope == session_key:
        return 1.18
    if scope == "weekend":
        return 1.0
    if session_key == "race" and scope == "qualifying":
        return 0.78
    if session_key == "qualifying" and scope == "race":
        return 0.70
    return 0.82


def _race_weight(item: dict, race_round: int, race_text: str) -> float:
    item_round = item.get("RaceRound") or item.get("race_round")
    if item_round and race_round and int(item_round) == race_round:
        return 1.25
    text = f"{item.get('Title', '')} {item.get('Content', '')}".lower()
    if race_text and any(part and part in text for part in race_text.split()):
        return 1.10
    return 0.72 if item_round else 0.86


def _dimension_weight(dimensions: dict, session_key: str, race_text: str) -> float:
    if not dimensions:
        return 0.78
    important = set(RACE_SESSION_DIMENSIONS.get(session_key) or RACE_SESSION_DIMENSIONS["race"])
    matched = sum(1 for key in dimensions if key in important)
    weight = 0.85 + min(0.35, matched * 0.08)
    if any(term in race_text for term in TRACK_QUALI_HEAVY_TERMS):
        if "qualifying_pace" in dimensions or "track_fit" in dimensions:
            weight += 0.18
        if session_key == "race" and "strategy" in dimensions:
            weight += 0.05
    return max(0.55, min(1.35, weight))


def _bounded_delta(score: float, confidence: float) -> float:
    return round(max(-0.035, min(0.035, float(score or 0.0) * float(confidence or 0.0) * 0.045)), 4)


def _topic_blend(topic_summary: dict, keys: list[str]) -> float:
    values = [float(topic_summary.get(key) or 0.0) for key in keys if key in topic_summary]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _article_summary(item: dict, impact: float, weight: float) -> dict:
    return {
        "title": item.get("Title") or item.get("title"),
        "source": item.get("Source") or item.get("source"),
        "url": item.get("Url") or item.get("url"),
        "published_utc": item.get("PublishedUtc") or item.get("published_utc"),
        "topics": item.get("Topics") or item.get("topics") or [],
        "dimensions": item.get("ImpactDimensions") or item.get("dimensions") or {},
        "impact": round(float(impact or 0.0), 4),
        "weight": round(float(weight or 0.0), 4),
        "confidence": item.get("SentimentConfidence") or item.get("confidence"),
        "source_kind": item.get("SourceKind") or item.get("source_kind") or _source_kind(item),
        "bias_flags": item.get("BiasFlags") or item.get("bias_flags") or [],
        "prediction_impact_cap": item.get("PredictionImpactCap") or item.get("prediction_impact_cap"),
        "reason": item.get("PredictionImpactReason") or item.get("reason"),
    }


def _impact_explanations(label: str, score: float, topic_summary: dict, articles: list[dict]) -> list[str]:
    if not articles:
        return []
    direction = "improves" if score > 0.02 else "hurts" if score < -0.02 else "keeps neutral"
    top_topics = sorted(topic_summary, key=lambda key: abs(float(topic_summary.get(key) or 0.0)), reverse=True)[:2]
    topic_text = " and ".join(topic.replace("_", " ") for topic in top_topics) or "general news"
    first = articles[0].get("title") or "Latest F1 news"
    return [f"{label} sentiment {direction} race model via {topic_text}.", str(first)]


def _top_explanations(top_impacts: list[dict]) -> list[str]:
    return [
        (item.get("explanations") or [])[0]
        for item in top_impacts[:5]
        if item.get("explanations")
    ]


def _is_race_relevant(item: dict, race_round: int, race_text: str) -> bool:
    if item.get("RaceRound") and race_round and int(item.get("RaceRound")) == race_round:
        return True
    text = f"{item.get('Title', '')} {item.get('Content', '')}".lower()
    return bool(race_text and any(part and len(part) > 3 and part in text for part in race_text.split()))


def _weighted_average(values: list[tuple[float, float]]) -> float:
    if not values:
        return 0.0
    total_weight = sum(max(0.001, weight) for _, weight in values)
    return sum(float(value) * max(0.001, weight) for value, weight in values) / total_weight


def _driver_aliases(driver: Driver) -> list[str]:
    full_name = f"{driver.first_name} {driver.last_name}".strip().lower()
    return [alias for alias in {full_name, driver.last_name.lower(), driver.id.lower(), driver.code.lower()} if alias]


def _team_aliases(constructor: Constructor) -> list[str]:
    name = (constructor.name or "").lower()
    aliases = {name, (constructor.id or "").lower()}
    if "red bull" in name:
        aliases.add("red bull")
    if "mercedes" in name:
        aliases.add("silver arrows")
    if "mclaren" in name:
        aliases.add("papaya")
    if "aston" in name:
        aliases.add("aston")
    if "sauber" in name or "audi" in name:
        aliases.update({"sauber", "audi", "kick sauber"})
    if "racing bulls" in name or constructor.id == "rb":
        aliases.update({"racing bulls", "visa cash app rb"})
    return [alias for alias in aliases if alias]


def _source_kind(item: dict) -> str:
    source = str(item.get("Source") or item.get("source") or "").lower()
    if source in {"formula1", "fia"}:
        return "official"
    if source == "jolpica":
        return "data"
    if "reddit" in source:
        return "social"
    if source in {"f1technical.net"}:
        return "technical"
    if source in {"mirror.co.uk", "planetf1.com", "crash.net"}:
        return "tabloid"
    return "news"


def _source_kind_weight(source_kind: str, item: dict) -> float:
    weight = {
        "official": 1.08,
        "technical": 0.96,
        "news": 0.88,
        "tabloid": 0.58,
        "social": 0.42,
        "data": 0.80,
    }.get(source_kind, 0.82)
    flags = item.get("BiasFlags") or item.get("bias_flags") or []
    if "rumor_or_unverified" in flags:
        weight *= 0.62
    return max(0.20, min(1.12, weight))


def _impact_cap(source_kind: str, item: dict) -> float:
    cap = {
        "official": 0.42,
        "technical": 0.34,
        "news": 0.28,
        "tabloid": 0.14,
        "social": 0.10,
        "data": 0.16,
    }.get(source_kind, 0.22)
    flags = item.get("BiasFlags") or item.get("bias_flags") or []
    if "rumor_or_unverified" in flags:
        cap *= 0.62
    return max(0.04, min(0.45, cap))


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _coverage_score(item: dict | None) -> float:
    item = item or {}
    mentions = int(item.get("mentions") or item.get("personal_mentions") or 0)
    sources = len(item.get("source_breakdown") or {})
    return round(min(1.0, mentions / 18.0 * 0.70 + sources / 6.0 * 0.30), 4)
