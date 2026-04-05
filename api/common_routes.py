"""Common API routes: health check, sports listing."""

from fastapi import APIRouter
from models.sport import Sport, SportInfo

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "sports-predictions"}


@router.get("/sports")
async def list_sports() -> list[SportInfo]:
    return [
        SportInfo(
            sport=Sport.SOCCER, label="Soccer", icon="fa-futbol",
            available=True, description="FIFA World Cup 2026 predictions",
        ),
        SportInfo(
            sport=Sport.FORMULA_ONE, label="Formula 1", icon="fa-flag-checkered",
            available=True, description="F1 World Championship 2026",
        ),
        SportInfo(
            sport=Sport.BASKETBALL, label="Basketball", icon="fa-basketball",
            available=False, description="NBA predictions — coming soon",
        ),
        SportInfo(
            sport=Sport.BASEBALL, label="Baseball", icon="fa-baseball",
            available=True, description="MLB 2026 season predictions",
        ),
        SportInfo(
            sport=Sport.TENNIS, label="Tennis", icon="fa-table-tennis-paddle-ball",
            available=False, description="ATP/WTA predictions — coming soon",
        ),
        SportInfo(
            sport=Sport.AMERICAN_FOOTBALL, label="NFL", icon="fa-football",
            available=False, description="NFL predictions — coming soon",
        ),
        SportInfo(
            sport=Sport.CRICKET, label="Cricket", icon="fa-cricket-bat-ball",
            available=False, description="ICC predictions — coming soon",
        ),
    ]
