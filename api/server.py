"""FastAPI application for sports predictions."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from data.football_data_client import FootballDataClient
from data.f1_client import F1Client
from data.mlb_client import MLBClient
from analytics.soccer_predictor import SoccerPredictor
from analytics.f1_predictor import F1Predictor
from analytics.baseball_predictor import BaseballPredictor
from api import common_routes, soccer_routes, f1_routes, baseball_routes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Shared client instances
_football_client: FootballDataClient | None = None
_f1_client: F1Client | None = None
_mlb_client: MLBClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _football_client, _f1_client, _mlb_client

    # Startup: initialize data clients and load data
    logger.info("Starting sports predictions service...")

    _football_client = FootballDataClient()
    _f1_client = F1Client()
    _mlb_client = MLBClient()

    soccer_pred = SoccerPredictor()
    f1_pred = F1Predictor()
    baseball_pred = BaseballPredictor()

    # Wire up route modules
    soccer_routes.init(_football_client, soccer_pred)
    f1_routes.init(_f1_client, f1_pred)
    baseball_routes.init(_mlb_client, baseball_pred)

    # Initial data load
    try:
        await _football_client.refresh()
        soccer_pred.load_teams(_football_client.get_teams())
        logger.info("Soccer data loaded successfully")
    except Exception as e:
        logger.warning("Soccer data load failed (will use fallback): %s", e)

    try:
        await _f1_client.refresh()
        f1_pred.load_drivers(_f1_client.get_drivers())
        logger.info("F1 data loaded successfully")
    except Exception as e:
        logger.warning("F1 data load failed (will use fallback): %s", e)

    try:
        await _mlb_client.refresh()
        baseball_pred.load_standings(_mlb_client.get_standings())
        logger.info("MLB data loaded successfully")
    except Exception as e:
        logger.warning("MLB data load failed (will use fallback): %s", e)

    yield

    # Shutdown
    if _football_client:
        await _football_client.close()
    if _f1_client:
        await _f1_client.close()
    if _mlb_client:
        await _mlb_client.close()
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

app.include_router(common_routes.router, prefix="/api")
app.include_router(soccer_routes.router, prefix="/api")
app.include_router(f1_routes.router, prefix="/api")
app.include_router(baseball_routes.router, prefix="/api")
