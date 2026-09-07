from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from chainlit.utils import mount_chainlit
from fastapi import FastAPI

from app.api.health import router as health_router
from app.config import Settings, get_settings
from app.observability.logging import configure_logging

log = structlog.get_logger()


async def shutdown_lmstudio(settings: Settings) -> None:
    """При остановке выгрузить модель только если LMSTUDIO_UNLOAD_ON_SHUTDOWN=true."""
    if not settings.lmstudio_unload_on_shutdown:
        log.info("lmstudio_shutdown_skip_unload")
        return

    try:
        from app.services.lmstudio_state import get_lmstudio_state_manager

        mgr = get_lmstudio_state_manager()
        log.info("lmstudio_shutdown_unload_start")
        result = await mgr.unload_model()
        log.info(
            "lmstudio_shutdown_unload_done",
            status=result.status.value,
            message=result.message,
        )
    except Exception:
        # Остановка приложения не должна падать из‑за unload
        log.exception("lmstudio_shutdown_unload_failed")


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(json_logs=settings.log_json)
    yield
    await shutdown_lmstudio(settings)


app = FastAPI(title="AI Advent Agent", lifespan=lifespan)
app.include_router(health_router)

# Chainlit UI на корне; REST остаётся доступным
mount_chainlit(app=app, target="app/chainlit/app.py", path="/")
