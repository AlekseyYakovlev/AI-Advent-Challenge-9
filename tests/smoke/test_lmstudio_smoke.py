import os

import pytest

from app.llm.base import ChatMessage, ModelSettings
from app.llm.openai_compatible import OpenAICompatibleProvider

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LMSTUDIO_SMOKE") != "1",
    reason="Нужен живой LM Studio и RUN_LMSTUDIO_SMOKE=1",
)


@pytest.mark.asyncio
async def test_lmstudio_models_and_chat() -> None:
    p = OpenAICompatibleProvider("http://localhost:1234/v1", "lm-studio")
    models = await p.list_models()
    assert models, "LM Studio не вернул модели — сервер выключен?"

    text = await p.chat(
        [ChatMessage(role="user", content="Ответь одним словом: pong")],
        ModelSettings(provider="lmstudio", model=models[0], max_tokens=32),
    )
    assert text.strip()
