from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from app.schemas.lmstudio import ModelLoadRequest, ModelLoadResult, ModelLoadStatus
from app.services.lmstudio_state import (
    CircuitBreaker,
    LMStudioStateManager,
    is_infrastructure_error,
    is_model_unloaded_error,
)


def _provider() -> MagicMock:
    p = MagicMock()
    p.list_models = AsyncMock(return_value=["A", "B"])
    p.load_model = AsyncMock(
        return_value=ModelLoadResult(
            status=ModelLoadStatus.LOADED,
            message="Загружена",
            model_id="A",
        )
    )
    p.unload_model = AsyncMock(
        return_value=ModelLoadResult(
            status=ModelLoadStatus.IDLE,
            message="ok",
        )
    )
    return p


@pytest.mark.asyncio
async def test_load_success_sets_loaded() -> None:
    mgr = LMStudioStateManager(_provider())
    result = await mgr.load_model(ModelLoadRequest(model_id="A"))
    assert result.status == ModelLoadStatus.LOADED
    assert mgr.get_state().status == ModelLoadStatus.LOADED
    assert mgr.get_state().current_model == "A"


@pytest.mark.asyncio
async def test_load_400_error_does_not_open_circuit() -> None:
    provider = _provider()
    provider.load_model = AsyncMock(
        return_value=ModelLoadResult(
            status=ModelLoadStatus.ERROR,
            message="bad request",
            model_id="bad",
        )
    )
    mgr = LMStudioStateManager(provider, circuit_threshold=1)
    result = await mgr.load_model(ModelLoadRequest(model_id="bad"))
    assert result.status == ModelLoadStatus.ERROR
    assert mgr.get_state().status == ModelLoadStatus.ERROR
    assert not mgr._circuit.is_open()


@pytest.mark.asyncio
async def test_load_500_model_error_does_not_open_circuit() -> None:
    provider = _provider()
    provider.load_model = AsyncMock(
        return_value=ModelLoadResult(
            status=ModelLoadStatus.ERROR,
            message="OOM",
            model_id="huge",
        )
    )
    mgr = LMStudioStateManager(provider, circuit_threshold=1)
    await mgr.load_model(ModelLoadRequest(model_id="huge"))
    assert not mgr._circuit.is_open()
    assert mgr.get_state().status == ModelLoadStatus.ERROR


@pytest.mark.asyncio
async def test_connect_error_opens_circuit_after_threshold() -> None:
    provider = _provider()
    provider.load_model = AsyncMock(
        return_value=ModelLoadResult(
            status=ModelLoadStatus.UNREACHABLE,
            message="down",
            model_id="A",
        )
    )
    mgr = LMStudioStateManager(provider, circuit_threshold=3, circuit_cooldown_seconds=120)

    r1 = await mgr.load_model(ModelLoadRequest(model_id="A"))
    assert r1.status == ModelLoadStatus.UNREACHABLE
    assert not mgr._circuit.is_open()

    r2 = await mgr.load_model(ModelLoadRequest(model_id="A"))
    assert r2.status == ModelLoadStatus.UNREACHABLE

    r3 = await mgr.load_model(ModelLoadRequest(model_id="A"))
    assert r3.status == ModelLoadStatus.CIRCUIT_OPEN
    assert mgr._circuit.is_open()
    assert r3.retry_after_seconds is not None

    r4 = await mgr.load_model(ModelLoadRequest(model_id="A"))
    assert r4.status == ModelLoadStatus.CIRCUIT_OPEN
    assert provider.load_model.await_count == 3


def test_circuit_breaker_ignores_model_http_errors() -> None:
    cb = CircuitBreaker(threshold=1, cooldown_seconds=60)
    req = httpx.Request("POST", "http://x/api/v0/models/load")
    resp = httpx.Response(400, request=req)
    exc = httpx.HTTPStatusError("bad", request=req, response=resp)
    assert not is_infrastructure_error(exc)
    cb.record_failure(exc)
    assert not cb.is_open()

    resp500 = httpx.Response(500, request=req)
    exc500 = httpx.HTTPStatusError("oom", request=req, response=resp500)
    assert not is_infrastructure_error(exc500)
    cb.record_failure(exc500)
    assert not cb.is_open()


def test_circuit_breaker_opens_on_connect_error() -> None:
    cb = CircuitBreaker(threshold=2, cooldown_seconds=60)
    cb.record_failure(httpx.ConnectError("x"))
    assert not cb.is_open()
    cb.record_failure(httpx.ConnectError("x"))
    assert cb.is_open()
    assert cb.retry_after_seconds() is not None


def test_is_model_unloaded_error() -> None:
    assert is_model_unloaded_error(RuntimeError("Model is not loaded"))
    assert is_model_unloaded_error(RuntimeError("No model loaded in server"))
    assert is_model_unloaded_error(RuntimeError("model not found"))
    assert not is_model_unloaded_error(RuntimeError("context length exceeded"))


@pytest.mark.asyncio
async def test_mark_chat_success_from_idle() -> None:
    mgr = LMStudioStateManager(_provider())
    assert mgr.get_state().status == ModelLoadStatus.IDLE
    mgr.mark_chat_success("Bionic")
    state = mgr.get_state()
    assert state.status == ModelLoadStatus.LOADED
    assert state.current_model == "Bionic"


@pytest.mark.asyncio
async def test_mark_chat_success_ignores_loading() -> None:
    mgr = LMStudioStateManager(_provider())
    mgr._state.status = ModelLoadStatus.LOADING
    mgr._state.current_model = "X"
    mgr.mark_chat_success("Y")
    assert mgr.get_state().status == ModelLoadStatus.LOADING
    assert mgr.get_state().current_model == "X"


@pytest.mark.asyncio
async def test_mark_chat_success_ignores_error() -> None:
    mgr = LMStudioStateManager(_provider())
    mgr._state.status = ModelLoadStatus.ERROR
    mgr._state.current_model = "X"
    mgr.mark_chat_success("Y")
    assert mgr.get_state().status == ModelLoadStatus.ERROR


@pytest.mark.asyncio
async def test_refresh_models_throttle_force() -> None:
    provider = _provider()
    mgr = LMStudioStateManager(provider, refresh_min_interval_seconds=5)
    result = await mgr.refresh_models(force=True)
    assert not result.throttled
    assert provider.list_models.await_count == 1
    mgr._state.last_refresh_at = datetime.now(UTC)
    result2 = await mgr.refresh_models(force=True)
    assert result2.throttled
    assert provider.list_models.await_count == 1


@pytest.mark.asyncio
async def test_refresh_models_uses_cache_ttl() -> None:
    provider = _provider()
    mgr = LMStudioStateManager(provider, models_cache_ttl_seconds=300)
    await mgr.refresh_models(force=False)
    assert provider.list_models.await_count == 1
    mgr._state.last_refresh_at = datetime.now(UTC) - timedelta(seconds=10)
    await mgr.refresh_models(force=False)
    assert provider.list_models.await_count == 1
