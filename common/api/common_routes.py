"""Common API routes: health check, sports listing."""

from fastapi import APIRouter
from common.models.sport import Sport, SportInfo

router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "ok", "service": "sports-predictions"}


@router.get("/sports")
async def list_sports() -> list[SportInfo]:
    # F1 and baseball are the supported sports. Add a SportInfo entry here
    # (plus a sports/<sport>/ package) to surface a new sport in the UI.
    return [
        SportInfo(
            sport=Sport.FORMULA_ONE, label="Formula 1", icon="fa-flag-checkered",
            available=True, description="F1 World Championship 2026",
        ),
        SportInfo(
            sport=Sport.BASEBALL, label="Baseball", icon="fa-baseball",
            available=True, href="/Sports#baseball", description="MLB 2026 season predictions",
        ),
    ]
