"""LM Studio provider: OpenAI-compatible chat + Bionic control API (load/unload)."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import structlog

from app.llm.openai_compatible import OpenAICompatibleProvider
from app.schemas.lmstudio import ModelLoadRequest, ModelLoadResult, ModelLoadStatus

log = structlog.get_logger()

_APP_ERROR_STATUSES = frozenset({400, 404, 422, 500})
_INFRA_HTTP_STATUSES = frozenset({502, 503, 504})


def split_lmstudio_bases(base_url: str) -> tuple[str, str]:
    """Вернуть (openai_base с /v1, control_base без /v1)."""
    cleaned = base_url.rstrip("/")
    if cleaned.endswith("/v1"):
        control = cleaned[: -len("/v1")].rstrip("/")
        openai = cleaned
    else:
        control = cleaned
        openai = f"{cleaned}/v1"
    return openai, control


class LMStudioProvider(OpenAICompatibleProvider):
    """Chat через OpenAI SDK; load/unload через httpx → /api/v0."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        *,
        load_timeout: float = 120.0,
        emergency_unload_timeout: float = 5.0,
        allow_top_k: bool = False,
    ) -> None:
        openai_base, control_base = split_lmstudio_bases(base_url)
        super().__init__(
            openai_base,
            api_key,
            timeout,
            allow_top_k=allow_top_k,
        )
        self._control_base = control_base
        self._api_key = api_key or "lm-studio"
        self._load_timeout = load_timeout
        self._emergency_unload_timeout = emergency_unload_timeout

    @property
    def control_base(self) -> str:
        return self._control_base

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _load_url(self) -> str:
        return f"{self._control_base}/api/v0/models/load"

    def _unload_url(self) -> str:
        return f"{self._control_base}/api/v0/models/unload"

    def _load_payload(self, request: ModelLoadRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model_id,
            "gpu_offload": request.gpu_offload,
        }
        if request.context_length is not None:
            payload["context_length"] = request.context_length
        return payload

    async def load_model(
        self,
        request: ModelLoadRequest,
        *,
        previous_model_id: str | None = None,
    ) -> ModelLoadResult:
        if (
            request.force_unload_previous
            and previous_model_id
            and previous_model_id != request.model_id
        ):
            unload_result = await self.unload_model(previous_model_id)
            if unload_result.status in (
                ModelLoadStatus.UNREACHABLE,
                ModelLoadStatus.CIRCUIT_OPEN,
            ):
                return unload_result

        try:
            async with httpx.AsyncClient() as client:
                log.info(
                    "lmstudio_load_request",
                    model_id=request.model_id,
                    timeout=self._load_timeout,
                )
                response = await client.post(
                    self._load_url(),
                    json=self._load_payload(request),
                    headers=self._headers(),
                    timeout=self._load_timeout,
                )
        except httpx.ConnectTimeout as exc:
            log.info(
                "lmstudio_load_error",
                model_id=request.model_id,
                reason="connect_timeout",
                error=str(exc),
            )
            return ModelLoadResult(
                status=ModelLoadStatus.UNREACHABLE,
                message=f"LM Studio недоступен (connect timeout): {exc}",
                model_id=request.model_id,
            )
        except httpx.ReadTimeout:
            log.info(
                "lmstudio_load_error",
                model_id=request.model_id,
                reason="read_timeout",
            )
            return await self._handle_load_timeout(request.model_id)
        except httpx.TimeoutException as exc:
            # Pool/Write и пр. на этапе соединения — инфраструктура
            if isinstance(exc, httpx.ConnectTimeout):
                log.info(
                    "lmstudio_load_error",
                    model_id=request.model_id,
                    reason="connect_timeout",
                    error=str(exc),
                )
                return ModelLoadResult(
                    status=ModelLoadStatus.UNREACHABLE,
                    message=str(exc),
                    model_id=request.model_id,
                )
            log.info(
                "lmstudio_load_error",
                model_id=request.model_id,
                reason="timeout",
                error=str(exc),
            )
            return await self._handle_load_timeout(request.model_id)
        except (
            httpx.ConnectError,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
        ) as exc:
            log.info(
                "lmstudio_load_error",
                model_id=request.model_id,
                reason="connect",
                error=str(exc),
            )
            return ModelLoadResult(
                status=ModelLoadStatus.UNREACHABLE,
                message=f"LM Studio недоступен: {exc}",
                model_id=request.model_id,
            )
        except Exception as exc:
            log.exception("lmstudio_load_unexpected", model_id=request.model_id)
            return ModelLoadResult(
                status=ModelLoadStatus.ERROR,
                message=f"Неожиданная ошибка загрузки: {exc}",
                model_id=request.model_id,
            )

        result = self._map_control_response(
            response,
            model_id=request.model_id,
            action="load",
        )
        if result.status == ModelLoadStatus.LOADED:
            log.info("lmstudio_load_success", model_id=request.model_id)
        else:
            log.info(
                "lmstudio_load_error",
                model_id=request.model_id,
                status=result.status.value,
                message=result.message,
            )
        return result

    async def unload_model(self, model_id: str) -> ModelLoadResult:
        log.info("lmstudio_unload_request", model_id=model_id)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._unload_url(),
                    json={"model": model_id},
                    headers=self._headers(),
                    timeout=self._load_timeout,
                )
        except (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            httpx.TimeoutException,
        ) as exc:
            log.info(
                "lmstudio_unload_error",
                model_id=model_id,
                reason="unreachable",
                error=str(exc),
            )
            return ModelLoadResult(
                status=ModelLoadStatus.UNREACHABLE,
                message=f"LM Studio недоступен при выгрузке: {exc}",
                model_id=model_id,
            )
        except Exception as exc:
            log.exception("lmstudio_unload_unexpected", model_id=model_id)
            return ModelLoadResult(
                status=ModelLoadStatus.ERROR,
                message=f"Неожиданная ошибка выгрузки: {exc}",
                model_id=model_id,
            )

        result = self._map_control_response(
            response,
            model_id=model_id,
            action="unload",
        )
        if result.status == ModelLoadStatus.IDLE:
            log.info("lmstudio_unload_success", model_id=model_id)
        else:
            log.info(
                "lmstudio_unload_error",
                model_id=model_id,
                status=result.status.value,
                message=result.message,
            )
        return result

    async def _handle_load_timeout(self, model_id: str) -> ModelLoadResult:
        log.info("lmstudio_emergency_unload_started", model_id=model_id)
        unload_ok = await self._emergency_unload(model_id)
        if not unload_ok:
            log.critical(
                "lmstudio_vram_not_freed_after_load_timeout",
                model_id=model_id,
            )
            return ModelLoadResult(
                status=ModelLoadStatus.ERROR,
                message=(
                    "Таймаут загрузки; аварийная выгрузка не удалась. "
                    "Сервер может остаться в неопределённом состоянии."
                ),
                model_id=model_id,
            )
        log.info("lmstudio_emergency_unload_success", model_id=model_id)
        return ModelLoadResult(
            status=ModelLoadStatus.ERROR,
            message="Таймаут загрузки модели; выполнена аварийная выгрузка.",
            model_id=model_id,
        )

    async def _emergency_unload(self, model_id: str) -> bool:
        """Аварийная выгрузка с жёстким таймаутом 5с; False если не удалось."""
        try:
            async with httpx.AsyncClient() as client:
                response = await asyncio.wait_for(
                    client.post(
                        self._unload_url(),
                        json={"model": model_id},
                        headers=self._headers(),
                        timeout=self._emergency_unload_timeout,
                    ),
                    timeout=self._emergency_unload_timeout,
                )
            if 200 <= response.status_code < 300:
                return True
            log.critical(
                "lmstudio_emergency_unload_failed",
                model_id=model_id,
                status_code=response.status_code,
                body=(response.text or "")[:200],
            )
            return False
        except Exception as exc:  # noqa: BLE001 — аварийный unload не должен ронять load
            # Отдельный try/except: падение unload не блокирует вызывающий поток
            log.critical(
                "lmstudio_emergency_unload_error",
                model_id=model_id,
                error=str(exc),
            )
            return False

    def _map_control_response(
        self,
        response: httpx.Response,
        *,
        model_id: str,
        action: str,
    ) -> ModelLoadResult:
        code = response.status_code
        body_preview = (response.text or "")[:500]

        if 200 <= code < 300:
            ok_status = ModelLoadStatus.LOADED if action == "load" else ModelLoadStatus.IDLE
            return ModelLoadResult(
                status=ok_status,
                message="Загружена" if action == "load" else "Модель выгружена",
                model_id=model_id if action == "load" else None,
            )

        if code in _INFRA_HTTP_STATUSES:
            return ModelLoadResult(
                status=ModelLoadStatus.UNREACHABLE,
                message=f"LM Studio gateway error {code}: {body_preview}",
                model_id=model_id,
            )

        if code in _APP_ERROR_STATUSES:
            return ModelLoadResult(
                status=ModelLoadStatus.ERROR,
                message=f"Ошибка {action} ({code}): {body_preview}",
                model_id=model_id,
            )

        return ModelLoadResult(
            status=ModelLoadStatus.ERROR,
            message=f"Неожиданный ответ {action} ({code}): {body_preview}",
            model_id=model_id,
        )
