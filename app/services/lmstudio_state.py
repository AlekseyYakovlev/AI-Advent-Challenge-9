"""Глобальное состояние загруженной модели LM Studio + Circuit Breaker."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import structlog
from openai import APIConnectionError, APITimeoutError

from app.llm.lmstudio_provider import LMStudioProvider
from app.schemas.lmstudio import (
    ModelLoadRequest,
    ModelLoadResult,
    ModelLoadStatus,
)

log = structlog.get_logger()

_manager: LMStudioStateManager | None = None

_STATUS_MESSAGES: dict[ModelLoadStatus, str] = {
    ModelLoadStatus.IDLE: "Модель не выбрана",
    ModelLoadStatus.LOADING: "Загрузка...",
    ModelLoadStatus.LOADED: "Загружена",
    ModelLoadStatus.ERROR: "Ошибка загрузки",
    ModelLoadStatus.UNREACHABLE: "Сервис недоступен",
    ModelLoadStatus.CIRCUIT_OPEN: "Сервис временно недоступен",
}

_MODEL_UNLOADED_MARKERS = (
    "model is not loaded",
    "no model loaded",
    "model not found",
    "model isn't loaded",
    "no models loaded",
    "failed to find a model instance",
)


def is_infrastructure_error(exc: Exception) -> bool:
    """True только для сбоев инфраструктуры (открывают Circuit Breaker)."""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.NetworkError,
            httpx.RemoteProtocolError,
            APIConnectionError,
            ConnectionError,
            OSError,
        ),
    ):
        # ReadTimeout / общий TimeoutException — не сюда (см. load path)
        if isinstance(exc, httpx.ReadTimeout):
            return False
        if isinstance(exc, httpx.TimeoutException) and not isinstance(
            exc, httpx.ConnectTimeout
        ):
            return False
        return True

    if isinstance(exc, APITimeoutError):
        # Таймаут chat/control без connect — не CB по умолчанию
        return False

    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (502, 503, 504)

    status = getattr(exc, "status_code", None)
    if status is not None:
        try:
            return int(status) in (502, 503, 504)
        except (TypeError, ValueError):
            return False
    return False


def is_model_unloaded_error(exc: Exception) -> bool:
    """Ошибка чата, явно указывающая что модель не загружена в LM Studio."""
    text = str(exc).lower()
    return any(marker in text for marker in _MODEL_UNLOADED_MARKERS)


@dataclass
class LMStudioState:
    current_model: str | None = None
    status: ModelLoadStatus = ModelLoadStatus.IDLE
    message: str = field(default_factory=lambda: _STATUS_MESSAGES[ModelLoadStatus.IDLE])
    available_models: list[str] = field(default_factory=list)
    last_refresh_at: datetime | None = None
    circuit_open_until: datetime | None = None


@dataclass(frozen=True, slots=True)
class RefreshModelsResult:
    state: LMStudioState
    throttled: bool = False


class CircuitBreaker:
    def __init__(
        self,
        *,
        threshold: int = 3,
        cooldown_seconds: int = 120,
    ) -> None:
        self._threshold = threshold
        self._cooldown = timedelta(seconds=cooldown_seconds)
        self._failures = 0
        self._open_until: datetime | None = None

    def record_failure(self, exc: Exception) -> None:
        if not is_infrastructure_error(exc):
            return
        self._failures += 1
        if self._failures >= self._threshold:
            self._open_until = datetime.now(UTC) + self._cooldown
            log.warning(
                "lmstudio_circuit_open",
                failures=self._failures,
                open_until=self._open_until.isoformat(),
            )

    def record_success(self) -> None:
        self._failures = 0
        self._open_until = None

    def is_open(self) -> bool:
        if self._open_until is None:
            return False
        if datetime.now(UTC) >= self._open_until:
            self._open_until = None
            self._failures = 0
            return False
        return True

    def retry_after_seconds(self) -> int | None:
        if not self.is_open() or self._open_until is None:
            return None
        delta = self._open_until - datetime.now(UTC)
        return max(0, int(delta.total_seconds()))

    @property
    def open_until(self) -> datetime | None:
        return self._open_until if self.is_open() else None


class LMStudioStateManager:
    """Глобальное (process-wide) состояние модели; без broadcast."""

    def __init__(
        self,
        provider: LMStudioProvider,
        *,
        circuit_threshold: int = 3,
        circuit_cooldown_seconds: int = 120,
        models_cache_ttl_seconds: int = 300,
        refresh_min_interval_seconds: int = 5,
    ) -> None:
        self._provider = provider
        self._lock = asyncio.Lock()
        self._state = LMStudioState()
        self._circuit = CircuitBreaker(
            threshold=circuit_threshold,
            cooldown_seconds=circuit_cooldown_seconds,
        )
        self._cache_ttl = timedelta(seconds=models_cache_ttl_seconds)
        self._refresh_min_interval = timedelta(seconds=refresh_min_interval_seconds)

    def get_state(self) -> LMStudioState:
        self._sync_circuit_into_state()
        return LMStudioState(
            current_model=self._state.current_model,
            status=self._state.status,
            message=self._state.message,
            available_models=list(self._state.available_models),
            last_refresh_at=self._state.last_refresh_at,
            circuit_open_until=self._circuit.open_until,
        )

    async def refresh_models(self, force: bool = False) -> RefreshModelsResult:
        async with self._lock:
            if self._circuit.is_open():
                self._set_status(
                    ModelLoadStatus.CIRCUIT_OPEN,
                    keep_model=True,
                    retry_after=self._circuit.retry_after_seconds(),
                )
                return RefreshModelsResult(state=self.get_state(), throttled=False)

            now = datetime.now(UTC)
            if self._state.last_refresh_at is not None:
                elapsed = now - self._state.last_refresh_at
                if force and elapsed < self._refresh_min_interval:
                    return RefreshModelsResult(state=self.get_state(), throttled=True)
                if not force and elapsed < self._cache_ttl:
                    return RefreshModelsResult(state=self.get_state(), throttled=False)

            try:
                models = await self._provider.list_models()
            except Exception as exc:
                if is_infrastructure_error(exc):
                    self._circuit.record_failure(exc)
                    status = (
                        ModelLoadStatus.CIRCUIT_OPEN
                        if self._circuit.is_open()
                        else ModelLoadStatus.UNREACHABLE
                    )
                    self._set_status(status, keep_model=True, detail=str(exc))
                else:
                    self._set_status(
                        ModelLoadStatus.ERROR,
                        keep_model=True,
                        detail=str(exc),
                    )
                return RefreshModelsResult(state=self.get_state(), throttled=False)

            self._circuit.record_success()
            self._state.available_models = list(models)
            self._state.last_refresh_at = now
            if self._state.status in (
                ModelLoadStatus.UNREACHABLE,
                ModelLoadStatus.CIRCUIT_OPEN,
            ):
                # Восстанавливаем логический статус после успешного ping списка
                if self._state.current_model:
                    self._set_status(ModelLoadStatus.LOADED, keep_model=True)
                else:
                    self._set_status(ModelLoadStatus.IDLE)
            return RefreshModelsResult(state=self.get_state(), throttled=False)

    async def load_model(self, request: ModelLoadRequest) -> ModelLoadResult:
        async with self._lock:
            if self._circuit.is_open():
                retry = self._circuit.retry_after_seconds()
                self._set_status(
                    ModelLoadStatus.CIRCUIT_OPEN,
                    keep_model=True,
                    retry_after=retry,
                )
                return ModelLoadResult(
                    status=ModelLoadStatus.CIRCUIT_OPEN,
                    message=(
                        "Сервис временно недоступен. "
                        f"Повторная попытка примерно через {retry or 120} с."
                    ),
                    model_id=request.model_id,
                    retry_after_seconds=retry,
                )

            previous = self._state.current_model
            self._state.current_model = request.model_id
            self._set_status(ModelLoadStatus.LOADING, keep_model=True)

            result = await self._provider.load_model(
                request,
                previous_model_id=previous,
            )

            if result.status == ModelLoadStatus.LOADED:
                self._circuit.record_success()
                self._state.current_model = request.model_id
                self._set_status(ModelLoadStatus.LOADED, keep_model=True)
                return result

            if result.status == ModelLoadStatus.UNREACHABLE:
                self._circuit.record_failure(httpx.ConnectError(result.message))
                if self._circuit.is_open():
                    retry = self._circuit.retry_after_seconds()
                    self._set_status(
                        ModelLoadStatus.CIRCUIT_OPEN,
                        keep_model=True,
                        retry_after=retry,
                    )
                    return ModelLoadResult(
                        status=ModelLoadStatus.CIRCUIT_OPEN,
                        message=_STATUS_MESSAGES[ModelLoadStatus.CIRCUIT_OPEN],
                        model_id=request.model_id,
                        retry_after_seconds=retry,
                    )
                self._set_status(
                    ModelLoadStatus.UNREACHABLE,
                    keep_model=True,
                    detail=result.message,
                )
                return result

            # ERROR (включая таймаут загрузки + аварийный unload)
            self._state.current_model = request.model_id
            self._set_status(
                ModelLoadStatus.ERROR,
                keep_model=True,
                detail=result.message,
            )
            return result

    async def unload_model(self) -> ModelLoadResult:
        async with self._lock:
            if self._circuit.is_open():
                retry = self._circuit.retry_after_seconds()
                return ModelLoadResult(
                    status=ModelLoadStatus.CIRCUIT_OPEN,
                    message=_STATUS_MESSAGES[ModelLoadStatus.CIRCUIT_OPEN],
                    retry_after_seconds=retry,
                )

            model_id = self._state.current_model
            if not model_id:
                self._set_status(ModelLoadStatus.IDLE)
                return ModelLoadResult(
                    status=ModelLoadStatus.IDLE,
                    message=_STATUS_MESSAGES[ModelLoadStatus.IDLE],
                )

            result = await self._provider.unload_model(model_id)
            if result.status == ModelLoadStatus.UNREACHABLE:
                self._circuit.record_failure(httpx.ConnectError(result.message))
                status = (
                    ModelLoadStatus.CIRCUIT_OPEN
                    if self._circuit.is_open()
                    else ModelLoadStatus.UNREACHABLE
                )
                self._set_status(status, keep_model=True, detail=result.message)
                if status == ModelLoadStatus.CIRCUIT_OPEN:
                    return ModelLoadResult(
                        status=status,
                        message=_STATUS_MESSAGES[status],
                        model_id=model_id,
                        retry_after_seconds=self._circuit.retry_after_seconds(),
                    )
                return result

            if result.status == ModelLoadStatus.ERROR:
                self._set_status(
                    ModelLoadStatus.ERROR,
                    keep_model=True,
                    detail=result.message,
                )
                return result

            self._state.current_model = None
            self._circuit.record_success()
            self._set_status(ModelLoadStatus.IDLE)
            return ModelLoadResult(
                status=ModelLoadStatus.IDLE,
                message=_STATUS_MESSAGES[ModelLoadStatus.IDLE],
            )

    def mark_chat_success(self, model_id: str) -> None:
        """Оптимистично: IDLE → LOADED после успешного чата."""
        if self._state.status != ModelLoadStatus.IDLE:
            return
        self._state.current_model = model_id
        self._set_status(ModelLoadStatus.LOADED, keep_model=True)

    def mark_unloaded(self) -> None:
        self._state.current_model = None
        self._set_status(ModelLoadStatus.IDLE)

    def record_infrastructure_failure(self, exc: Exception) -> None:
        self._circuit.record_failure(exc)
        if self._circuit.is_open():
            self._set_status(
                ModelLoadStatus.CIRCUIT_OPEN,
                keep_model=True,
                retry_after=self._circuit.retry_after_seconds(),
            )
        elif is_infrastructure_error(exc):
            self._set_status(
                ModelLoadStatus.UNREACHABLE,
                keep_model=True,
                detail=str(exc),
            )

    def _sync_circuit_into_state(self) -> None:
        if self._circuit.is_open():
            if self._state.status != ModelLoadStatus.CIRCUIT_OPEN:
                self._set_status(
                    ModelLoadStatus.CIRCUIT_OPEN,
                    keep_model=True,
                    retry_after=self._circuit.retry_after_seconds(),
                )
            self._state.circuit_open_until = self._circuit.open_until
        else:
            self._state.circuit_open_until = None

    def _set_status(
        self,
        status: ModelLoadStatus,
        *,
        keep_model: bool = False,
        detail: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        self._state.status = status
        if not keep_model and status == ModelLoadStatus.IDLE:
            self._state.current_model = None
        base = _STATUS_MESSAGES[status]
        if status == ModelLoadStatus.CIRCUIT_OPEN and retry_after is not None:
            self._state.message = (
                f"{base}. Повторная попытка примерно через {retry_after} с."
            )
        elif detail:
            self._state.message = f"{base}: {detail}"
        else:
            self._state.message = base
        self._state.circuit_open_until = self._circuit.open_until


def build_lmstudio_state_manager(
    provider: LMStudioProvider,
    settings: Any,
) -> LMStudioStateManager:
    return LMStudioStateManager(
        provider,
        circuit_threshold=settings.lmstudio_circuit_breaker_threshold,
        circuit_cooldown_seconds=settings.lmstudio_circuit_breaker_cooldown_seconds,
        models_cache_ttl_seconds=settings.lmstudio_models_cache_ttl_seconds,
        refresh_min_interval_seconds=settings.lmstudio_refresh_min_interval_seconds,
    )


def get_lmstudio_state_manager() -> LMStudioStateManager:
    """Process-wide singleton; источник истины для Lazy UI Sync."""
    global _manager
    if _manager is not None:
        return _manager
    from app.config import get_settings
    from app.llm.factory import create_provider

    settings = get_settings()
    provider = create_provider("lmstudio", settings)
    if not isinstance(provider, LMStudioProvider):
        raise TypeError("ожидался LMStudioProvider для lmstudio")
    _manager = build_lmstudio_state_manager(provider, settings)
    return _manager


def reset_lmstudio_state_manager_for_tests() -> None:
    """Сброс singleton между unit-тестами."""
    global _manager
    _manager = None
