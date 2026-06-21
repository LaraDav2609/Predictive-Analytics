"""Selects the CSGO data client from environment configuration.

    CSGO_DATA_PROVIDER   "pandascore" | "localfile" | "stub"  (default: "stub")
    PANDASCORE_TOKEN     bearer token for PandaScore (required for pandascore)
    PANDASCORE_BASE_URL  override API base (default https://api.pandascore.co)
    CSGO_PANDASCORE_GAME PandaScore videogame slug (default "csgo")
    CSGO_LOOKBACK_DAYS   recent-results window for provisional ratings (default 180)
    CSGO_MAX_TEAMS       cap on teams loaded (default 100)
    CSGO_HISTORY_FILE    path to a CSV/JSON results export (required for localfile)
    LIQUIPEDIA_API_KEY   free Liquipedia v3 API key (required for liquipedia)
    LIQUIPEDIA_USER_AGENT  descriptive UA w/ contact, per Liquipedia's terms

Falls back to the in-memory stub when the provider is unset/unknown or when
``pandascore`` is selected without a token, so the service always boots.
"""

from __future__ import annotations

import logging
import os

from games.csgo.data.csgo_client import CsgoDataClient, StubCsgoClient

logger = logging.getLogger(__name__)


def _tag(client: CsgoDataClient, provider: str, synthetic: bool) -> CsgoDataClient:
    """Stamp the EFFECTIVE provider on the client so the API/dashboard can show whether
    the numbers are real or synthetic (a stub fallback reports provider='stub')."""
    client.provider_name = provider
    client.is_synthetic = synthetic
    return client


def build_csgo_client() -> CsgoDataClient:
    provider = os.getenv("CSGO_DATA_PROVIDER", "stub").strip().lower()

    if provider == "pandascore":
        token = os.getenv("PANDASCORE_TOKEN", "").strip()
        if not token:
            logger.warning(
                "CSGO_DATA_PROVIDER=pandascore but PANDASCORE_TOKEN is unset — "
                "falling back to stub CSGO data."
            )
            return _tag(StubCsgoClient(), "stub", True)

        # Imported lazily so the stub path has no httpx import cost.
        from games.csgo.data.pandascore_client import PandaScoreCsgoClient

        def _int_env(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)))
            except ValueError:
                return default

        logger.info("CSGO data provider: PandaScore")
        return _tag(PandaScoreCsgoClient(
            token=token,
            base_url=os.getenv("PANDASCORE_BASE_URL", "https://api.pandascore.co"),
            game=os.getenv("CSGO_PANDASCORE_GAME", "csgo"),
            lookback_days=_int_env("CSGO_LOOKBACK_DAYS", 180),
            max_teams=_int_env("CSGO_MAX_TEAMS", 100),
        ), "pandascore", False)

    if provider == "localfile":
        path = os.getenv("CSGO_HISTORY_FILE", "").strip()
        if not path:
            logger.warning("CSGO_DATA_PROVIDER=localfile but CSGO_HISTORY_FILE is unset — using stub.")
            return _tag(StubCsgoClient(), "stub", True)
        from games.csgo.data.localfile_client import LocalHistoryCsgoClient

        logger.info("CSGO data provider: local history file (%s)", path)
        return _tag(LocalHistoryCsgoClient(path), "localfile", False)

    if provider == "liquipedia":
        key = os.getenv("LIQUIPEDIA_API_KEY", "").strip()
        if not key:
            logger.warning("CSGO_DATA_PROVIDER=liquipedia but LIQUIPEDIA_API_KEY is unset — using stub.")
            return _tag(StubCsgoClient(), "stub", True)
        from games.csgo.data.liquipedia_client import LiquipediaCsgoClient

        try:
            lookback = int(os.getenv("CSGO_LOOKBACK_DAYS", "180"))
        except ValueError:
            lookback = 180
        logger.info("CSGO data provider: Liquipedia")
        return _tag(LiquipediaCsgoClient(
            key,
            user_agent=os.getenv("LIQUIPEDIA_USER_AGENT",
                                 "PredictiveAnalytics-CSGO/1.0 (set LIQUIPEDIA_USER_AGENT with contact)"),
            base_url=os.getenv("LIQUIPEDIA_BASE_URL", "https://api.liquipedia.net/api/v3"),
            wiki=os.getenv("CSGO_LIQUIPEDIA_WIKI", "counterstrike"),
            lookback_days=lookback,
        ), "liquipedia", False)

    if provider not in ("stub", ""):
        logger.warning("Unknown CSGO_DATA_PROVIDER=%r — using stub.", provider)
    return _tag(StubCsgoClient(), "stub", True)
