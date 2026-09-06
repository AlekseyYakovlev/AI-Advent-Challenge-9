from typing import Any

from fastapi import APIRouter

from app.config import get_settings
from app.llm.factory import create_provider

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health() -> dict[str, Any]:
    settings = get_settings()
    provider_id = settings.default_provider
    llm_ok = False
    llm_error: str | None = None
    try:
        # Короткий timeout — иначе при down-провайдере health висит до 120с
        provider = create_provider(
            provider_id,
            settings,
            timeout=settings.health_timeout_sec,
        )
        await provider.list_models()
        llm_ok = True
    except Exception as exc:  # noqa: BLE001 — health не должен падать
        llm_error = str(exc)

    if not settings.health_verbose:
        return {"status": "ok" if llm_ok else "degraded"}

    return {
        "status": "ok" if llm_ok else "degraded",
        "env": settings.app_env,
        "provider": provider_id,
        "llm": {"ok": llm_ok, "error": llm_error},
    }
