"""FastAPI entrypoint. Run from the repo root:

    set -a; . ./.env; set +a
    python -m online.backend.api_server

Launching by path instead of `-m` breaks the `core.*` imports.
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from online.backend.resources import resources
from online.backend.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logger.info("initialising datastores")
    resources.initialize_datastores()
    logger.info("initialising models")
    resources.initialize_models()
    logger.info("ready: %s", resources.status)
    yield
    if resources.qwen is not None:
        resources.qwen.close()


app = FastAPI(title="VQA on Video-MME-v2", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        log_level="info",
    )
