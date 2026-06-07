"""Entry point for 4Schoolers Sports Predictions API."""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("common.api.server:app", host="0.0.0.0", port=8100, reload=True)
