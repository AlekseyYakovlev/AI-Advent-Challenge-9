import json

import httpx
import pytest
import respx

from app.chainlit.app import (
    generate_improved_prompt,
    wrap_user_question_for_prompt_generation,
)
from app.chainlit.settings_schema import build_chat_settings
from app.config import Settings
from app.llm.base import ChatMessage, ModelSettings
from app.llm.openai_compatible import OpenAICompatibleProvider
from app.services.agent import AgentService

BASE = "http://lmstudio.test/v1"


def _request_body_bytes(request: httpx.Request) -> bytes:
    content = getattr(request, "content", None)
    if content:
        return content if isinstance(content, bytes) else bytes(content)
    read = getattr(request, "read", None)
    if read is None:
        return b""
    result = read()
    if hasattr(result, "__await__"):
        raise AssertionError("request.read() вернул awaitable — используйте request.content")
    return result  # type: ignore[no-any-return]


def _chat_completion_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        },
    )


def _sse_stream_response(content: str) -> httpx.Response:
    sse = (
        'data: {"id":"1","object":"chat.completion.chunk","choices":'
        f'[{{"index":0,"delta":{{"content":{json.dumps(content)}}},'
        '"finish_reason":null}]}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(
        200,
        content=sse,
        headers={"content-type": "text/event-stream"},
    )


@pytest.fixture
def provider() -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(BASE, "x", timeout=5.0)


def test_model_settings_pre_generated_prompt_default() -> None:
    settings = ModelSettings(provider="lmstudio", model="Bionic")
    assert settings.pre_generated_prompt is False


def test_build_chat_settings_includes_pre_generated_prompt_switch() -> None:
    widgets = build_chat_settings(
        ModelSettings(provider="lmstudio", model="Bionic")
    )
    ids = [getattr(w, "id", None) for w in widgets]
    assert "pre_generated_prompt" in ids


def test_wrap_user_question_for_prompt_generation() -> None:
    assert wrap_user_question_for_prompt_generation("how to catch a butterfly?") == (
        'Generate a prompt to solve the following question: "how to catch a butterfly?". '
        "Return only the prompt."
    )


@respx.mock
@pytest.mark.asyncio
async def test_generate_improved_prompt_sends_only_wrapped_user_message(
    provider: OpenAICompatibleProvider,
) -> None:
    generated = (
        "Write a structured overview of Python async/await "
        "with examples and pitfalls."
    )
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=_chat_completion_response(generated)
    )

    history = [ChatMessage(role="user", content="расскажи про async")]
    settings = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        pre_generated_prompt=True,
        step_by_step=True,
        system_prompt="не должен уйти на этапе 1",
    )
    result = await generate_improved_prompt(
        history, settings, provider, Settings()
    )

    assert result == generated
    assert route.call_count == 1
    body = json.loads(_request_body_bytes(route.calls[0].request))
    roles = [m["role"] for m in body["messages"]]
    assert "system" not in roles
    assert body["messages"] == [
        {
            "role": "user",
            "content": wrap_user_question_for_prompt_generation("расскажи про async"),
        }
    ]
    assert body["stream"] is False


@respx.mock
@pytest.mark.asyncio
async def test_pre_generated_prompt_two_stage_updates_history(
    provider: OpenAICompatibleProvider,
) -> None:
    generated = "Explain recursion with a base case and an inductive step."
    final_answer = "Recursion is when a function calls itself."
    route = respx.post(f"{BASE}/chat/completions").mock(
        side_effect=[
            _chat_completion_response(generated),
            _sse_stream_response(final_answer),
        ]
    )

    user_question = "что такое рекурсия?"
    history = [ChatMessage(role="user", content=user_question)]
    model_settings = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        pre_generated_prompt=True,
        system_prompt="Ты полезный ассистент.",
    )
    app_settings = Settings()

    improved = await generate_improved_prompt(
        history, model_settings, provider, app_settings
    )
    history[-1] = ChatMessage(role="user", content=improved)

    agent = AgentService(provider, app_settings)
    tokens = [
        token
        async for token in agent.astream(history, model_settings)
    ]
    history.append(ChatMessage(role="assistant", content="".join(tokens)))

    assert route.call_count == 2
    assert history[0].role == "user"
    assert history[0].content == generated
    assert history[1].role == "assistant"
    assert history[1].content == final_answer

    body1 = json.loads(_request_body_bytes(route.calls[0].request))
    assert body1["stream"] is False
    assert "system" not in [m["role"] for m in body1["messages"]]
    assert body1["messages"][-1]["content"] == wrap_user_question_for_prompt_generation(
        user_question
    )

    # Второй вызов — stream с оригинальным system_prompt пользователя
    body2 = json.loads(_request_body_bytes(route.calls[1].request))
    assert body2["stream"] is True
    assert body2["messages"][0]["content"] == "Ты полезный ассистент."
    assert body2["messages"][-1]["content"] == generated


@respx.mock
@pytest.mark.asyncio
async def test_generate_improved_prompt_error_stops_before_second_call(
    provider: OpenAICompatibleProvider,
) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "boom"})
    )

    history = [ChatMessage(role="user", content="hi")]
    settings = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        pre_generated_prompt=True,
    )

    with pytest.raises(Exception):
        await generate_improved_prompt(
            history, settings, provider, Settings()
        )

    assert route.call_count >= 1
    # История пользователя не заменяется при ошибке этапа 1
    assert history[-1].content == "hi"
