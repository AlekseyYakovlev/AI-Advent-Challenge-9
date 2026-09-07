"""Фаза 4: конфигурация, shutdown, CB, аварийная очистка, безопасность."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.llm.lmstudio_provider import LMStudioProvider
from app.main import shutdown_lmstudio
from app.schemas.lmstudio import ModelLoadRequest, ModelLoadResult, ModelLoadStatus
from app.services.lmstudio_state import CircuitBreaker, LMStudioStateManager

CONTROL = "http://lmstudio.test"
OPENAI = f"{CONTROL}/v1"


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_unload_on_shutdown_default_false() -> None:
    settings = Settings(_env_file=None)
    assert settings.lmstudio_unload_on_shutdown is False


def test_unload_on_shutdown_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LMSTUDIO_UNLOAD_ON_SHUTDOWN", "true")
    settings = Settings(_env_file=None)
    assert settings.lmstudio_unload_on_shutdown is True


@pytest.mark.asyncio
async def test_shutdown_skips_unload_when_flag_false() -> None:
    settings = Settings(_env_file=None, lmstudio_unload_on_shutdown=False)
    with patch("app.services.lmstudio_state.get_lmstudio_state_manager") as get_mgr:
        await shutdown_lmstudio(settings)
        get_mgr.assert_not_called()


@pytest.mark.asyncio
async def test_shutdown_calls_unload_when_flag_true() -> None:
    settings = Settings(_env_file=None, lmstudio_unload_on_shutdown=True)
    mgr = MagicMock()
    mgr.unload_model = AsyncMock(
        return_value=ModelLoadResult(status=ModelLoadStatus.IDLE, message="ok")
    )
    with patch(
        "app.services.lmstudio_state.get_lmstudio_state_manager",
        return_value=mgr,
    ):
        await shutdown_lmstudio(settings)
    mgr.unload_model.assert_awaited_once()


def test_circuit_three_network_errors_open() -> None:
    cb = CircuitBreaker(threshold=3, cooldown_seconds=120)
    cb.record_failure(httpx.ConnectError("1"))
    cb.record_failure(httpx.ConnectError("2"))
    assert not cb.is_open()
    cb.record_failure(httpx.ConnectError("3"))
    assert cb.is_open()


def test_circuit_500_model_error_does_not_open() -> None:
    cb = CircuitBreaker(threshold=1, cooldown_seconds=60)
    req = httpx.Request("POST", "http://x/api/v0/models/load")
    resp = httpx.Response(500, request=req)
    exc = httpx.HTTPStatusError("oom", request=req, response=resp)
    cb.record_failure(exc)
    assert not cb.is_open()


def test_circuit_400_does_not_open() -> None:
    cb = CircuitBreaker(threshold=1, cooldown_seconds=60)
    req = httpx.Request("POST", "http://x/api/v0/models/load")
    resp = httpx.Response(400, request=req)
    exc = httpx.HTTPStatusError("bad", request=req, response=resp)
    cb.record_failure(exc)
    assert not cb.is_open()


def test_circuit_success_resets_failure_counter() -> None:
    cb = CircuitBreaker(threshold=3, cooldown_seconds=60)
    cb.record_failure(httpx.ConnectError("1"))
    cb.record_failure(httpx.ConnectError("2"))
    cb.record_success()
    cb.record_failure(httpx.ConnectError("1"))
    cb.record_failure(httpx.ConnectError("2"))
    assert not cb.is_open()
    cb.record_failure(httpx.ConnectError("3"))
    assert cb.is_open()


def test_optimistic_idle_to_loaded() -> None:
    mgr = LMStudioStateManager(MagicMock())
    assert mgr.get_state().status == ModelLoadStatus.IDLE
    mgr.mark_chat_success("Bionic")
    assert mgr.get_state().status == ModelLoadStatus.LOADED
    assert mgr.get_state().current_model == "Bionic"


def test_optimistic_does_not_overwrite_loading() -> None:
    mgr = LMStudioStateManager(MagicMock())
    mgr._state.status = ModelLoadStatus.LOADING
    mgr._state.current_model = "X"
    mgr.mark_chat_success("Y")
    assert mgr.get_state().status == ModelLoadStatus.LOADING
    assert mgr.get_state().current_model == "X"


def test_optimistic_does_not_overwrite_error() -> None:
    mgr = LMStudioStateManager(MagicMock())
    mgr._state.status = ModelLoadStatus.ERROR
    mgr._state.current_model = "X"
    mgr.mark_chat_success("Y")
    assert mgr.get_state().status == ModelLoadStatus.ERROR


@pytest.mark.parametrize(
    "model_id",
    [
        "..",
        "/../",
        "models/../secret",
        "has space",
        "a b",
    ],
)
def test_model_id_path_and_spaces_rejected(model_id: str) -> None:
    with pytest.raises(ValidationError):
        ModelLoadRequest(model_id=model_id)


@respx.mock
@pytest.mark.asyncio
async def test_load_timeout_triggers_emergency_unload() -> None:
    provider = LMStudioProvider(
        OPENAI,
        "x",
        load_timeout=1.0,
        emergency_unload_timeout=0.5,
    )
    respx.post(f"{CONTROL}/api/v0/models/load").mock(
        side_effect=httpx.ReadTimeout("slow", request=httpx.Request("POST", OPENAI))
    )
    unload = respx.post(f"{CONTROL}/api/v0/models/unload").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    result = await provider.load_model(ModelLoadRequest(model_id="Bionic"))
    assert result.status == ModelLoadStatus.ERROR
    assert unload.called
    assert "таймаут" in result.message.lower() or "аварийная" in result.message.lower()


@respx.mock
@pytest.mark.asyncio
async def test_emergency_unload_failure_does_not_hang_and_logs_critical() -> None:
    provider = LMStudioProvider(
        OPENAI,
        "x",
        load_timeout=1.0,
        emergency_unload_timeout=0.3,
    )
    respx.post(f"{CONTROL}/api/v0/models/load").mock(
        side_effect=httpx.ReadTimeout("slow", request=httpx.Request("POST", OPENAI))
    )

    async def _hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200)

    respx.post(f"{CONTROL}/api/v0/models/unload").mock(side_effect=_hang)

    with patch("app.llm.lmstudio_provider.log") as mock_log:
        mock_log.info = MagicMock()
        mock_log.exception = MagicMock()
        mock_log.critical = MagicMock()
        result = await asyncio.wait_for(
            provider.load_model(ModelLoadRequest(model_id="Bionic")),
            timeout=3.0,
        )

    assert result.status == ModelLoadStatus.ERROR
    assert "Traceback" not in result.message
    assert "аварийная" in result.message.lower() or "таймаут" in result.message.lower()
    assert mock_log.critical.called
    events = [c.args[0] for c in mock_log.critical.call_args_list if c.args]
    assert any(
        name in events
        for name in (
            "lmstudio_emergency_unload_error",
            "lmstudio_vram_not_freed_after_load_timeout",
            "lmstudio_emergency_unload_failed",
        )
    )


@pytest.mark.asyncio
async def test_load_timeout_sets_manager_error_with_safe_message() -> None:
    provider = MagicMock()
    provider.load_model = AsyncMock(
        return_value=ModelLoadResult(
            status=ModelLoadStatus.ERROR,
            message="Таймаут загрузки; аварийная выгрузка не удалась.",
            model_id="Bionic",
        )
    )
    mgr = LMStudioStateManager(provider)
    result = await mgr.load_model(ModelLoadRequest(model_id="Bionic"))
    assert result.status == ModelLoadStatus.ERROR
    assert mgr.get_state().status == ModelLoadStatus.ERROR
    assert "Traceback" not in (mgr.get_state().message or "")
