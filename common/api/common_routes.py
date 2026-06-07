"""Common API routes: health check, domain (sport/game) listing."""

from fastapi import APIRouter
from common.models.sport import Category, Sport, SportInfo

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "sports-predictions"}


@router.get("/sports")
async def list_sports() -> list[SportInfo]:
    # The supported prediction domains, grouped by category. Add a SportInfo entry
    # here (plus a sports/<sport>/ or games/<game>/ package) to surface a new one.
    return [
        SportInfo(
            sport=Sport.FORMULA_ONE, label="Formula 1", icon="fa-flag-checkered",
            available=True, category=Category.SPORT, description="F1 World Championship 2026",
        ),
        SportInfo(
            sport=Sport.BASEBALL, label="Baseball", icon="fa-baseball",
            available=True, category=Category.SPORT, href="/Sports#baseball",
            description="MLB 2026 season predictions",
        ),
        SportInfo(
            sport=Sport.CSGO, label="CS2 / CSGO", icon="fa-gamepad",
            available=True, category=Category.GAME, href="/Games/Csgo",
            description="CS2 esports match predictions",
        ),
    ]
