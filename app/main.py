from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from chainlit.utils import mount_chainlit
from fastapi import FastAPI

from app.api.health import router as health_router
from app.config import get_settings
from app.observability.logging import configure_logging


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(json_logs=settings.log_json)
    yield


app = FastAPI(title="AI Advent Agent", lifespan=lifespan)
app.include_router(health_router)

# Chainlit UI на корне; REST остаётся доступным
mount_chainlit(app=app, target="app/chainlit/app.py", path="/")
