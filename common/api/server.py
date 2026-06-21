"""FastAPI application for sports predictions."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from sports.f1.data.f1_client import F1Client
from sports.baseball.data.mlb_client import MLBClient
from sports.baseball.data.mlb_historical_client import MLBHistoricalClient
from sports.baseball.data.pybaseball_client import PybaseballClient
from sports.f1.analytics.f1_predictor import F1Predictor
from sports.baseball.analytics.baseball_predictor import BaseballPredictor
from sports.f1.data.f1_sentiment import read_f1_sentiment, refresh_f1_sentiment
from common.api import common_routes
from sports.f1.api import f1_routes
from sports.baseball.api import baseball_routes, baseball_history_routes
from games.csgo.data.csgo_client import CsgoDataClient
from games.csgo.data.factory import build_csgo_client
from games.csgo.analytics.csgo_predictor import CsgoPredictor
from games.csgo.analytics.pipeline import CsgoModelPipeline
from games.csgo.api import csgo_routes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Shared client instances
_f1_client: F1Client | None = None
_mlb_client: MLBClient | None = None
_mlb_historical_client: MLBHistoricalClient | None = None
_csgo_client: CsgoDataClient | None = None
_startup_tasks: set[asyncio.Task] = set()


def _track_startup_task(name: str, coro) -> None:
    task = asyncio.create_task(coro, name=f"startup:{name}")
    _startup_tasks.add(task)

    def _done(completed: asyncio.Task) -> None:
        _startup_tasks.discard(completed)
        try:
            completed.result()
        except asyncio.CancelledError:
            logger.debug("%s initial data load cancelled", name)
        except Exception as exc:
            logger.warning("%s initial data load failed (will use fallback): %s", name, exc)

    task.add_done_callback(_done)


async def _load_f1_data(client: F1Client, predictor: F1Predictor) -> None:
    await client.refresh()
    f1_features = await client.get_prediction_features()
    f1_sentiment = None
    try:
        cached_sentiment = read_f1_sentiment(client.get_drivers(), client.get_constructors())
        if (
            cached_sentiment.get("source_items")
            or cached_sentiment.get("published_items")
            or cached_sentiment.get("composite")
            or cached_sentiment.get("items")
        ):
            f1_sentiment = cached_sentiment
            logger.info("F1 sentiment startup reused cached snapshot")
    except Exception as exc:
        logger.debug("F1 sentiment startup cache read skipped: %s", exc)

    if not f1_sentiment:
        try:
            f1_sentiment = await asyncio.wait_for(
                refresh_f1_sentiment(
                    client.get_drivers(),
                    client.get_constructors(),
                    client.season,
                ),
                timeout=20,
            )
        except Exception as exc:
            logger.warning("F1 sentiment startup refresh skipped: %s", exc)
            f1_sentiment = None
    predictor.load_drivers(
        client.get_drivers(),
        client.get_constructors(),
        f1_features,
        f1_sentiment,
    )
    logger.info("F1 data loaded successfully")


async def _load_mlb_data(client: MLBClient, predictor: BaseballPredictor) -> None:
    await client.refresh()
    predictor.load_standings(client.get_standings())
    logger.info("MLB data loaded successfully")


async def _load_csgo_data(client: CsgoDataClient, model: CsgoModelPipeline) -> None:
    await client.refresh()
    model.fit(client.get_past_matches(), client.get_teams())
    logger.info(
        "CSGO data loaded: %d teams, %d matches",
        len(client.get_teams()), len(client.get_matches()),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _f1_client, _mlb_client, _mlb_historical_client, _csgo_client

    # Startup: initialize data clients and load data
    logger.info("Starting sports predictions service...")

    _f1_client = F1Client()
    _mlb_client = MLBClient()
    _mlb_historical_client = MLBHistoricalClient()
    pybaseball_client = PybaseballClient()

    f1_pred = F1Predictor()
    baseball_pred = BaseballPredictor()

    # Wire up route modules
    f1_routes.init(_f1_client, f1_pred)
    baseball_routes.init(_mlb_client, baseball_pred)
    baseball_history_routes.init(_mlb_historical_client, pybaseball_client)

    # CSGO (game category) — provider selected from env (PandaScore or stub).
    # The pipeline (ratings → features → ensemble) drives /matches predictions.
    _csgo_client = build_csgo_client()
    csgo_model = CsgoModelPipeline()
    csgo_routes.init(_csgo_client, csgo_model)
    csgo_model.load_teams(_csgo_client.get_teams())  # immediate (stub data, or empty until fit)

    # Initial data loads are intentionally non-blocking. Some upstream sports/F1
    # APIs can be slow or unavailable, and the dashboard should still boot.
    _track_startup_task("F1", _load_f1_data(_f1_client, f1_pred))
    _track_startup_task("MLB", _load_mlb_data(_mlb_client, baseball_pred))
    _track_startup_task("CSGO", _load_csgo_data(_csgo_client, csgo_model))

    yield

    # Shutdown
    for task in list(_startup_tasks):
        task.cancel()
    if _startup_tasks:
        await asyncio.gather(*_startup_tasks, return_exceptions=True)
    if _f1_client:
        await _f1_client.close()
    await f1_routes.close()
    if _mlb_client:
        await _mlb_client.close()
    if _mlb_historical_client:
        await _mlb_historical_client.close()
    if _csgo_client is not None and hasattr(_csgo_client, "close"):
        await _csgo_client.close()
    logger.info("Sports predictions service stopped")


app = FastAPI(
    title="4Schoolers Sports Predictions API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

app.include_router(common_routes.router, prefix="/api")
app.include_router(f1_routes.router, prefix="/api")
app.include_router(baseball_routes.router, prefix="/api")
app.include_router(baseball_history_routes.router, prefix="/api")
app.include_router(csgo_routes.router, prefix="/api")
