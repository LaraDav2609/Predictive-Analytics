"""Selects the CSGO data client from environment configuration.

    CSGO_DATA_PROVIDER   "pandascore" | "stub"  (default: "stub")
    PANDASCORE_TOKEN     bearer token for PandaScore (required for pandascore)
    PANDASCORE_BASE_URL  override API base (default https://api.pandascore.co)
    CSGO_PANDASCORE_GAME PandaScore videogame slug (default "csgo")
    CSGO_LOOKBACK_DAYS   recent-results window for provisional ratings (default 180)
    CSGO_MAX_TEAMS       cap on teams loaded (default 100)

Falls back to the in-memory stub when the provider is unset/unknown or when
``pandascore`` is selected without a token, so the service always boots.
"""

from __future__ import annotations

import logging
import os

from games.csgo.data.csgo_client import CsgoDataClient, StubCsgoClient

logger = logging.getLogger(__name__)


def build_csgo_client() -> CsgoDataClient:
    provider = os.getenv("CSGO_DATA_PROVIDER", "stub").strip().lower()

    if provider == "pandascore":
        token = os.getenv("PANDASCORE_TOKEN", "").strip()
        if not token:
            logger.warning(
                "CSGO_DATA_PROVIDER=pandascore but PANDASCORE_TOKEN is unset — "
                "falling back to stub CSGO data."
            )
            return StubCsgoClient()

        # Imported lazily so the stub path has no httpx import cost.
        from games.csgo.data.pandascore_client import PandaScoreCsgoClient

        def _int_env(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        logger.info("CSGO data provider: PandaScore")
        return PandaScoreCsgoClient(
            token=token,
            base_url=os.getenv("PANDASCORE_BASE_URL", "https://api.pandascore.co"),
            game=os.getenv("CSGO_PANDASCORE_GAME", "csgo"),
            lookback_days=_int_env("CSGO_LOOKBACK_DAYS", 180),
            max_teams=_int_env("CSGO_MAX_TEAMS", 100),
        )

    if provider not in ("stub", ""):
        logger.warning("Unknown CSGO_DATA_PROVIDER=%r — using stub.", provider)
    return StubCsgoClient()
