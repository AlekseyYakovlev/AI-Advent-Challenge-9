import asyncio

import httpx
import pytest
import respx

from app.llm.lmstudio_provider import LMStudioProvider, split_lmstudio_bases
from app.schemas.lmstudio import ModelLoadRequest, ModelLoadStatus

CONTROL = "http://lmstudio.test"
OPENAI = f"{CONTROL}/v1"


@pytest.fixture
def provider() -> LMStudioProvider:
    return LMStudioProvider(
        OPENAI,
        "x",
        timeout=5.0,
        load_timeout=2.0,
        emergency_unload_timeout=0.5,
    )


def test_split_bases_strips_v1() -> None:
    openai, control = split_lmstudio_bases("http://localhost:1234/v1")
    assert openai == "http://localhost:1234/v1"
    assert control == "http://localhost:1234"


def test_split_bases_adds_v1() -> None:
    openai, control = split_lmstudio_bases("http://localhost:1234")
    assert openai == "http://localhost:1234/v1"
    assert control == "http://localhost:1234"


@respx.mock
@pytest.mark.asyncio
async def test_load_success(provider: LMStudioProvider) -> None:
    respx.post(f"{CONTROL}/api/v0/models/load").mock(
        return_value=httpx.Response(200, json={"status": "loaded"})
    )
    result = await provider.load_model(ModelLoadRequest(model_id="Bionic"))
    assert result.status == ModelLoadStatus.LOADED
    assert result.model_id == "Bionic"


@respx.mock
@pytest.mark.asyncio
async def test_load_400_is_error_not_unreachable(provider: LMStudioProvider) -> None:
    respx.post(f"{CONTROL}/api/v0/models/load").mock(
        return_value=httpx.Response(400, text="bad model")
    )
    result = await provider.load_model(ModelLoadRequest(model_id="bad"))
    assert result.status == ModelLoadStatus.ERROR


@respx.mock
@pytest.mark.asyncio
async def test_load_500_model_error(provider: LMStudioProvider) -> None:
    respx.post(f"{CONTROL}/api/v0/models/load").mock(
        return_value=httpx.Response(500, text="out of memory")
    )
    result = await provider.load_model(ModelLoadRequest(model_id="huge"))
    assert result.status == ModelLoadStatus.ERROR


@respx.mock
@pytest.mark.asyncio
async def test_load_connect_error_unreachable(provider: LMStudioProvider) -> None:
    respx.post(f"{CONTROL}/api/v0/models/load").mock(
        side_effect=httpx.ConnectError("boom")
    )
    result = await provider.load_model(ModelLoadRequest(model_id="Bionic"))
    assert result.status == ModelLoadStatus.UNREACHABLE


@respx.mock
@pytest.mark.asyncio
async def test_load_timeout_triggers_emergency_unload(
    provider: LMStudioProvider,
) -> None:
    respx.post(f"{CONTROL}/api/v0/models/load").mock(
        side_effect=httpx.ReadTimeout("slow", request=httpx.Request("POST", OPENAI))
    )
    unload = respx.post(f"{CONTROL}/api/v0/models/unload").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    result = await provider.load_model(ModelLoadRequest(model_id="Bionic"))
    assert result.status == ModelLoadStatus.ERROR
    assert unload.called


@respx.mock
@pytest.mark.asyncio
async def test_emergency_unload_timeout_does_not_hang(
    provider: LMStudioProvider,
) -> None:
    respx.post(f"{CONTROL}/api/v0/models/load").mock(
        side_effect=httpx.ReadTimeout("slow", request=httpx.Request("POST", OPENAI))
    )

    async def _hang(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(30)
        return httpx.Response(200)

    respx.post(f"{CONTROL}/api/v0/models/unload").mock(side_effect=_hang)

    result = await asyncio.wait_for(
        provider.load_model(ModelLoadRequest(model_id="Bionic")),
        timeout=3.0,
    )
    assert result.status == ModelLoadStatus.ERROR
    assert "аварийная выгрузка" in result.message.lower()


@respx.mock
@pytest.mark.asyncio
async def test_force_unload_previous(provider: LMStudioProvider) -> None:
    unload = respx.post(f"{CONTROL}/api/v0/models/unload").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    load = respx.post(f"{CONTROL}/api/v0/models/load").mock(
        return_value=httpx.Response(200, json={"status": "loaded"})
    )
    result = await provider.load_model(
        ModelLoadRequest(model_id="new-model", force_unload_previous=True),
        previous_model_id="old-model",
    )
    assert result.status == ModelLoadStatus.LOADED
    assert unload.called
    assert load.called


@respx.mock
@pytest.mark.asyncio
async def test_unload_success(provider: LMStudioProvider) -> None:
    respx.post(f"{CONTROL}/api/v0/models/unload").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )
    result = await provider.unload_model("Bionic")
    assert result.status == ModelLoadStatus.IDLE
