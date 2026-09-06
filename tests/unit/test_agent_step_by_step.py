from collections.abc import AsyncIterator

import pytest

from app.config import Settings
from app.llm.base import STEP_BY_STEP_INSTRUCTION, ChatMessage, LLMProvider, ModelSettings
from app.services.agent import AgentService


class _CaptureProvider(LLMProvider):
    def __init__(self) -> None:
        self.last_messages: list[ChatMessage] = []

    async def list_models(self) -> list[str]:
        return []

    async def chat(
        self,
        messages: list[ChatMessage],
        settings: ModelSettings,
    ) -> str:
        self.last_messages = messages
        return "ok"

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        settings: ModelSettings,
    ) -> AsyncIterator[str]:
        self.last_messages = messages
        yield "ok"


@pytest.mark.asyncio
async def test_step_by_step_appends_instruction_to_system_prompt() -> None:
    provider = _CaptureProvider()
    agent = AgentService(provider, Settings())
    settings = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        system_prompt="Base prompt.",
        step_by_step=True,
    )

    tokens = [
        token
        async for token in agent.astream(
            [ChatMessage(role="user", content="hi")],
            settings,
        )
    ]

    assert tokens == ["ok"]
    assert provider.last_messages[0].role == "system"
    assert provider.last_messages[0].content == (
        f"Base prompt.\n\n{STEP_BY_STEP_INSTRUCTION}"
    )
    assert "identical to the language of the user's request" in STEP_BY_STEP_INSTRUCTION


@pytest.mark.asyncio
async def test_step_by_step_off_keeps_system_prompt() -> None:
    provider = _CaptureProvider()
    agent = AgentService(provider, Settings())
    settings = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        system_prompt="Base prompt.",
        step_by_step=False,
    )

    async for _ in agent.astream(
        [ChatMessage(role="user", content="hi")],
        settings,
    ):
        pass

    assert provider.last_messages[0].content == "Base prompt."
