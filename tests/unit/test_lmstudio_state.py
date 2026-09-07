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
async def test_mark_chat_success_from_unreachable() -> None:
    mgr = LMStudioStateManager(_provider())
    mgr._state.status = ModelLoadStatus.UNREACHABLE
    mgr._state.current_model = None
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
async def test_ensure_loads_when_model_differs() -> None:
    provider = _provider()
    mgr = LMStudioStateManager(provider)
    mgr._state.status = ModelLoadStatus.LOADED
    mgr._state.current_model = "Old"

    result = await mgr.ensure_ready_for_generation("A")

    assert result.ok
    assert provider.load_model.await_count == 1
    assert mgr.get_state().current_model == "A"
    assert mgr.get_state().status == ModelLoadStatus.LOADED


@pytest.mark.asyncio
async def test_ensure_load_failure_blocks_generation() -> None:
    provider = _provider()
    provider.load_model = AsyncMock(
        return_value=ModelLoadResult(
            status=ModelLoadStatus.ERROR,
            message="OOM",
            model_id="huge",
        )
    )
    mgr = LMStudioStateManager(provider)

    result = await mgr.ensure_ready_for_generation("huge")

    assert not result.ok
    assert result.message is not None
    assert "OOM" in result.message
    assert provider.load_model.await_count == 1
    assert mgr.get_state().status == ModelLoadStatus.ERROR


@pytest.mark.asyncio
async def test_ensure_skips_load_when_circuit_open() -> None:
    provider = _provider()
    mgr = LMStudioStateManager(provider, circuit_threshold=1)
    mgr._circuit.record_failure(httpx.ConnectError("down"))
    assert mgr._circuit.is_open()

    result = await mgr.ensure_ready_for_generation("A")

    assert not result.ok
    assert result.message == (
        "Сервис временно недоступен. Повторите через 2 минуты."
    )
    provider.load_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_no_infinite_auto_reload_after_failure() -> None:
    provider = _provider()
    provider.load_model = AsyncMock(
        return_value=ModelLoadResult(
            status=ModelLoadStatus.ERROR,
            message="fail",
            model_id="A",
        )
    )
    mgr = LMStudioStateManager(provider)

    first = await mgr.ensure_ready_for_generation("A")
    assert not first.ok
    assert provider.load_model.await_count == 1

    # ERROR блокирует без повторной загрузки
    second = await mgr.ensure_ready_for_generation("A")
    assert not second.ok
    assert provider.load_model.await_count == 1

    # После mark_unloaded (как при chat "model not loaded") — тоже без авто-reload
    mgr.mark_unloaded(block_auto_reload_for="A")
    third = await mgr.ensure_ready_for_generation("A")
    assert not third.ok
    assert third.message == "Модель не загружена. Выберите модель заново."
    assert provider.load_model.await_count == 1

    # Явный выбор снимает блок
    mgr.clear_auto_load_block()
    provider.load_model = AsyncMock(
        return_value=ModelLoadResult(
            status=ModelLoadStatus.LOADED,
            message="Загружена",
            model_id="A",
        )
    )
    fourth = await mgr.ensure_ready_for_generation("A")
    assert fourth.ok
    assert provider.load_model.await_count == 1


@pytest.mark.asyncio
async def test_ensure_loading_does_not_start_parallel_load() -> None:
    provider = _provider()
    mgr = LMStudioStateManager(provider)
    mgr._state.status = ModelLoadStatus.LOADING
    mgr._state.current_model = "A"

    result = await mgr.ensure_ready_for_generation("B")

    assert not result.ok
    assert result.message == "Модель ещё загружается. Подождите завершения."
    provider.load_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_format_chat_error_model_unloaded_resets_status() -> None:
    mgr = LMStudioStateManager(_provider())
    mgr._state.status = ModelLoadStatus.LOADED
    mgr._state.current_model = "A"

    msg = mgr.format_chat_error(
        RuntimeError("Model is not loaded"),
        model_id="A",
    )

    assert msg == "Модель не загружена. Выберите модель заново."
    assert mgr.get_state().status == ModelLoadStatus.IDLE
    assert mgr.get_state().current_model is None

    # Повторная автозагрузка заблокирована
    again = await mgr.ensure_ready_for_generation("A")
    assert not again.ok
    mgr._provider.load_model.assert_not_awaited()


@pytest.mark.asyncio
async def test_format_chat_error_ordinary_does_not_change_status() -> None:
    mgr = LMStudioStateManager(_provider())
    await mgr.load_model(ModelLoadRequest(model_id="A"))
    assert mgr.get_state().status == ModelLoadStatus.LOADED

    msg = mgr.format_chat_error(
        RuntimeError("context length exceeded"),
        model_id="A",
    )

    assert msg.startswith("Ошибка LLM:")
    assert mgr.get_state().status == ModelLoadStatus.LOADED
    assert mgr.get_state().current_model == "A"
    assert not mgr._circuit.is_open()


@pytest.mark.asyncio
async def test_format_chat_error_infra_keeps_model_identity() -> None:
    mgr = LMStudioStateManager(_provider(), circuit_threshold=3)
    await mgr.load_model(ModelLoadRequest(model_id="A"))

    msg = mgr.format_chat_error(httpx.ConnectError("down"), model_id="A")

    assert msg == "Сервис временно недоступен."
    state = mgr.get_state()
    assert state.status == ModelLoadStatus.UNREACHABLE
    assert state.current_model == "A"


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
