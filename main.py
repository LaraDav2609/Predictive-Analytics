"""Entry point for 4Schoolers Sports Predictions API."""

import os

import uvicorn

if __name__ == "__main__":
    reload_enabled = os.getenv("SPORTS_API_RELOAD", "0").lower() in {"1", "true", "yes"}
    uvicorn.run(
        "common.api.server:app",
        host="0.0.0.0",
        port=8100,
        reload=reload_enabled,
        access_log=False,
    )
