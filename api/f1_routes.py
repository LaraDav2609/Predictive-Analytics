"""Formula 1 API routes."""

import json

from fastapi import APIRouter
import redis

from data.f1_client import F1Client
from data.f1_sentiment import read_f1_sentiment, refresh_f1_sentiment
from analytics.f1_predictor import F1Predictor

router = APIRouter(prefix="/f1", tags=["f1"])

# Shared instances (initialized in server.py lifespan)
client: F1Client | None = None
predictor: F1Predictor | None = None


def init(fc: F1Client, fp: F1Predictor):
    global client, predictor
    client, predictor = fc, fp


@router.get("/drivers")
async def get_drivers():
    drivers = client.get_drivers()
    return {"ok": True, "drivers": [d.model_dump() for d in drivers]}


@router.get("/drivers/{driver_id}")
async def get_driver(driver_id: str, season: int | None = None):
    return await client.get_driver_profile(driver_id, season)


@router.get("/constructors")
async def get_constructors():
    constructors = client.get_constructors()
    return {"ok": True, "constructors": [c.model_dump() for c in constructors]}


@router.get("/constructors/{constructor_id}")
async def get_constructor(constructor_id: str, season: int | None = None):
    return await client.get_constructor_profile(constructor_id, season)


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
        return {"ok": False, "reason": str(ex), "composite": None, "items": []}


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
        "prediction_model": "f1-live-historical-sentiment-v2",
    }
