"""Formula 1 API routes."""

import json

from fastapi import APIRouter
import redis

from data.f1_client import F1Client
from data.openf1_client import OpenF1Client
from data.f1_sentiment import read_f1_sentiment, refresh_f1_sentiment
from analytics.f1_predictor import F1Predictor
from analytics.f1_simulator import build_session_simulation

router = APIRouter(prefix="/f1", tags=["f1"])

# Shared instances (initialized in server.py lifespan)
client: F1Client | None = None
predictor: F1Predictor | None = None
openf1: OpenF1Client | None = None


def init(fc: F1Client, fp: F1Predictor):
    global client, predictor, openf1
    client, predictor = fc, fp
    openf1 = OpenF1Client()


async def close():
    if openf1:
        await openf1.close()


def _sentiment_snapshot() -> dict:
    if predictor:
        return predictor.get_sentiment()
    return {}


async def _ensure_sentiment() -> dict:
    sentiment = _sentiment_snapshot()
    if sentiment.get("drivers") and sentiment.get("teams"):
        return sentiment

    sentiment = read_f1_sentiment(client.get_drivers(), client.get_constructors(), limit=80)
    if sentiment.get("drivers") and sentiment.get("teams") and predictor:
        features = await client.get_prediction_features()
        predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, sentiment)
        return sentiment

    refreshed = await refresh_f1_sentiment(client.get_drivers(), client.get_constructors(), client.season)
    if predictor:
        features = await client.get_prediction_features()
        predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, refreshed)
    return refreshed


def _driver_sentiment(driver_id: str) -> dict:
    sentiment = _sentiment_snapshot()
    driver = (sentiment.get("drivers") or {}).get(driver_id) or {}
    composite = sentiment.get("composite") or {}
    personal_score = driver.get("personal_score", driver.get("score", 0.0))
    team_score = driver.get("team_score", 0.0)
    overall_score = driver.get("overall_score", composite.get("Composite", 0.0))
    return {
        "score": driver.get("score", 0.0),
        "label": driver.get("label", "Neutral"),
        "confidence": driver.get("confidence", 0.0),
        "mentions": driver.get("mentions", 0),
        "personal_score": personal_score,
        "personal_label": driver.get("personal_label", "Neutral"),
        "personal_mentions": driver.get("personal_mentions", 0),
        "team_score": team_score,
        "team_label": driver.get("team_label", "Neutral"),
        "team_mentions": driver.get("team_mentions", 0),
        "overall_score": overall_score,
        "overall_label": driver.get("overall_label", composite.get("Label", "Neutral")),
        "win_probability_modifier": driver.get("win_probability_modifier", 1.0),
        "wdc_probability_modifier": driver.get("wdc_probability_modifier", 1.0),
        "source_breakdown": driver.get("source_breakdown", {}),
        "topic_scores": driver.get("topic_scores", {}),
        "dimensions": [
            {"key": "personal", "label": "Personal", "score": personal_score, "mentions": driver.get("personal_mentions", 0), "sentiment": driver.get("personal_label", "Neutral")},
            {"key": "team", "label": "Team", "score": team_score, "mentions": driver.get("team_mentions", 0), "sentiment": driver.get("team_label", "Neutral")},
            {"key": "overall", "label": "Overall", "score": overall_score, "mentions": (composite.get("TotalItems") or 0), "sentiment": driver.get("overall_label", composite.get("Label", "Neutral"))},
        ],
        "latest_items": driver.get("latest_items", []),
    }


def _constructor_sentiment(constructor_name: str) -> dict:
    sentiment = _sentiment_snapshot()
    team = (sentiment.get("teams") or {}).get((constructor_name or "").lower()) or {}
    return {
        "score": team.get("score", 0.0),
        "label": team.get("label", "Neutral"),
        "confidence": team.get("confidence", 0.0),
        "mentions": team.get("mentions", 0),
        "source_breakdown": team.get("source_breakdown", {}),
        "topic_scores": team.get("topic_scores", {}),
        "dimensions": [
            {"key": key, "label": key.replace("_", " ").title(), "score": value}
            for key, value in sorted((team.get("topic_scores") or {}).items())
        ],
        "latest_items": team.get("latest_items", []),
    }


@router.get("/drivers")
async def get_drivers():
    drivers = client.get_drivers()
    return {"ok": True, "drivers": [d.model_dump() for d in drivers]}


@router.get("/drivers/{driver_id}")
async def get_driver(driver_id: str, season: int | None = None):
    profile = await client.get_driver_profile(driver_id, season)
    if profile.get("ok"):
        await _ensure_sentiment()
        driver = profile.get("driver") or {}
        profile["sentiment"] = _driver_sentiment(driver.get("id") or driver_id)
    return profile


@router.get("/constructors")
async def get_constructors():
    constructors = client.get_constructors()
    return {"ok": True, "constructors": [c.model_dump() for c in constructors]}


@router.get("/constructors/{constructor_id}")
async def get_constructor(constructor_id: str, season: int | None = None):
    profile = await client.get_constructor_profile(constructor_id, season)
    if profile.get("ok"):
        await _ensure_sentiment()
        constructor = profile.get("constructor") or {}
        profile["sentiment"] = _constructor_sentiment(constructor.get("name") or constructor_id)
    return profile


@router.get("/calendar")
async def get_calendar():
    races = client.get_races()
    if predictor:
        races = predictor.predict_races(races)
    return {"ok": True, "races": [r.model_dump(mode="json") for r in races]}


@router.get("/sentiment")
async def get_sentiment(limit: int = 12):
    try:
        r = redis.from_url("redis://localhost:6379/0", decode_responses=True)
        composite_raw = r.get("sentiment:composite:latest:f1")
        composite = json.loads(composite_raw) if composite_raw else None
        keys = r.zrevrange("feeditems:category:f1", 0, max(limit - 1, 0))
        items = []
        for key in keys:
            raw = r.get(key)
            if not raw:
                continue
            item = json.loads(raw)
            items.append({
                "title": item.get("Title"),
                "content": item.get("Content"),
                "url": item.get("Url"),
                "source": item.get("Source"),
                "feed": item.get("Feed"),
                "published_utc": item.get("PublishedUtc"),
                "score": item.get("SentimentScore"),
                "label": item.get("SentimentLabel"),
                "confidence": item.get("SentimentConfidence"),
                "analyzer": item.get("SentimentAnalyzerUsed") or item.get("SentimentAnalyzer"),
            })
        entities = read_f1_sentiment(client.get_drivers(), client.get_constructors(), limit=60)
        return {
            "ok": True,
            "composite": composite or entities.get("composite"),
            "items": items,
            "drivers": entities.get("drivers", {}),
            "teams": entities.get("teams", {}),
        }
    except Exception as ex:
        try:
            features = await client.get_prediction_features()
            entities = await refresh_f1_sentiment(client.get_drivers(), client.get_constructors(), client.season)
            if predictor:
                predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, entities)
            return {
                "ok": True,
                "reason": f"Redis unavailable, refreshed live news directly: {ex}",
                "composite": entities.get("composite"),
                "items": [
                    {
                        "title": item.get("Title"),
                        "content": item.get("Content"),
                        "url": item.get("Url"),
                        "source": item.get("Source"),
                        "feed": item.get("Feed"),
                        "published_utc": item.get("PublishedUtc"),
                        "score": item.get("SentimentScore"),
                        "label": item.get("SentimentLabel"),
                        "confidence": item.get("SentimentConfidence"),
                        "topics": item.get("Topics") or [],
                        "analyzer": item.get("SentimentAnalyzerUsed") or item.get("SentimentAnalyzer"),
                    }
                    for item in (entities.get("items") or [])[:limit]
                ],
                "drivers": entities.get("drivers", {}),
                "teams": entities.get("teams", {}),
            }
        except Exception as refresh_ex:
            return {"ok": False, "reason": f"{ex}; live refresh failed: {refresh_ex}", "composite": None, "items": []}


@router.get("/performance")
async def get_performance():
    if predictor:
        if not (predictor._features or {}).get("drivers"):
            features = await client.get_prediction_features()
            predictor.load_drivers(client.get_drivers(), client.get_constructors(), features, await _ensure_sentiment())
        return predictor.get_performance_intelligence()
    features = await client.get_prediction_features()
    fallback = F1Predictor()
    fallback.load_drivers(client.get_drivers(), client.get_constructors(), features, await _ensure_sentiment())
    return fallback.get_performance_intelligence()


@router.get("/races/{round_num}")
async def get_race(round_num: int):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found"}
    if race.status == "SCHEDULED" and predictor:
        race.prediction = predictor.predict_race(race)
    return {"ok": True, "race": race.model_dump(mode="json")}


@router.get("/races/{round_num}/profile")
async def get_race_profile(round_num: int):
    return await client.get_race_profile(round_num, predictor)


@router.get("/races/{round_num}/simulation")
async def get_race_simulation(round_num: int, session: str = "race", live: bool = False):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found"}

    if race.status == "SCHEDULED" and predictor:
        race.prediction = predictor.predict_race(race)

    features = await client.get_prediction_features()
    profile = await client.get_race_profile(round_num, predictor)
    if not profile.get("ok"):
        return profile
    context = profile.get("context") or {}
    if (session or "").lower() == "sprint" and not context.get("has_sprint"):
        return {
            "ok": False,
            "reason": "Sprint session is not scheduled for this Grand Prix",
            "code": "sprint_unavailable",
            "race": race.model_dump(mode="json"),
            "context": context,
        }

    simulation = build_session_simulation(
        race=race,
        drivers=client.get_drivers(),
        constructors=client.get_constructors(),
        prediction=(race.prediction.model_dump(mode="json") if race.prediction else {}),
        features=features,
        qualifying=profile.get("qualifying") or [],
        sprint=profile.get("sprint") or [],
        results=profile.get("results") or [],
        session=session,
        live=live,
    )
    simulation["context"] = context
    if openf1:
        simulation["track"] = await openf1.get_track_data(
            race=race,
            drivers=client.get_drivers(),
            session=session,
            live=live,
        )
    return simulation


@router.get("/races/{round_num}/track")
async def get_race_track(round_num: int, session: str = "race", live: bool = False):
    race = client.get_race_by_round(round_num)
    if not race:
        return {"ok": False, "reason": "Race not found"}
    if not openf1:
        return {"ok": False, "reason": "OpenF1 client unavailable"}
    if (session or "").lower() == "sprint":
        profile = await client.get_race_profile(round_num, predictor)
        context = profile.get("context") or {}
        if not context.get("has_sprint"):
            return {
                "ok": False,
                "reason": "Sprint session is not scheduled for this Grand Prix",
                "code": "sprint_unavailable",
                "context": context,
            }
    return await openf1.get_track_data(
        race=race,
        drivers=client.get_drivers(),
        session=session,
        live=live,
    )


@router.post("/refresh")
async def refresh():
    await client.refresh()
    features = await client.get_prediction_features()
    sentiment = await refresh_f1_sentiment(
        client.get_drivers(),
        client.get_constructors(),
        client.season,
    )
    if predictor:
        predictor.load_drivers(
            client.get_drivers(),
            client.get_constructors(),
            features,
            sentiment,
        )
    return {
        "ok": True,
        "drivers": len(client.get_drivers()),
        "constructors": len(client.get_constructors()),
        "races": len(client.get_races()),
        "completed_races": features.get("completed_races", 0),
        "sentiment_items": sentiment.get("source_items", 0),
        "sentiment_published_items": sentiment.get("published_items", 0),
        "sentiment_configured_sources": sentiment.get("configured_sources", 0),
        "prediction_model": "f1-live-historical-sentiment-v4",
    }
