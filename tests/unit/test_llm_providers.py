import httpx
import pytest
import respx
from openai import APIError, APITimeoutError, BadRequestError, RateLimitError

from app.llm.base import ChatMessage, ModelSettings
from app.llm.openai_compatible import OpenAICompatibleProvider, _is_retryable_openai_error
from app.services.agent import _is_context_overflow

BASE = "http://lmstudio.test/v1"


def _request() -> httpx.Request:
    return httpx.Request("POST", f"{BASE}/chat/completions")


def _api_error(status: int, message: str = "err") -> APIError:
    """Фабрика, устойчивая к минорным отличиям сигнатур openai SDK (микро-риск T1)."""
    req = _request()
    resp = httpx.Response(status, request=req)
    try:
        if status == 429:
            return RateLimitError(message, response=resp, body=None)
        if status == 400:
            return BadRequestError(message, response=resp, body=None)
        err = APIError(message, request=req, body=None)
        err.status_code = status  # type: ignore[attr-defined]
        return err
    except TypeError:
        # Fallback без MagicMock: предикату нужен реальный подкласс + status_code
        class _E(APIError):
            pass

        real = _E.__new__(_E)
        real.status_code = status  # type: ignore[attr-defined]
        real.message = message
        return real  # type: ignore[return-value]


def _request_body_bytes(request: httpx.Request) -> bytes:
    """respx/httpx: предпочитать .content; .read() — fallback (микро-риск T3)."""
    content = getattr(request, "content", None)
    if content:
        return content if isinstance(content, bytes) else bytes(content)
    read = getattr(request, "read", None)
    if read is None:
        return b""
    result = read()
    # на случай async read в будущих версиях — тесты sync-only
    if hasattr(result, "__await__"):
        raise AssertionError("request.read() вернул awaitable — используйте request.content")
    return result  # type: ignore[no-any-return]


@pytest.fixture
def provider() -> OpenAICompatibleProvider:
    # Без custom transport: иначе respx может не перехватить (микро-риск T2)
    return OpenAICompatibleProvider(BASE, "x", timeout=5.0)


@respx.mock
@pytest.mark.asyncio
async def test_chat_via_respx(provider: OpenAICompatibleProvider) -> None:
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "pong"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    text = await provider.chat(
        [ChatMessage(role="user", content="ping")],
        ModelSettings(provider="lmstudio", model="Bionic", max_tokens=16),
    )
    assert text == "pong"


@respx.mock
@pytest.mark.asyncio
async def test_stream_chat_via_respx(provider: OpenAICompatibleProvider) -> None:
    # Упрощённый SSE: один data-chunk + [DONE]
    sse = (
        'data: {"id":"1","object":"chat.completion.chunk","choices":'
        '[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}\n\n'
        "data: [DONE]\n\n"
    )
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )
    )
    tokens = [
        t
        async for t in provider.stream_chat(
            [ChatMessage(role="user", content="ping")],
            ModelSettings(provider="lmstudio", model="Bionic"),
        )
    ]
    assert tokens == ["Hi"]


@respx.mock
@pytest.mark.asyncio
async def test_payload_omits_top_k_by_default(provider: OpenAICompatibleProvider) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    await provider.chat(
        [ChatMessage(role="user", content="x")],
        ModelSettings(provider="lmstudio", model="Bionic", top_k=40),
    )
    body = _request_body_bytes(route.calls.last.request)
    assert b"top_k" not in body  # M3


@respx.mock
@pytest.mark.asyncio
async def test_payload_includes_stop(provider: OpenAICompatibleProvider) -> None:
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "1",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )
    )
    await provider.chat(
        [ChatMessage(role="user", content="x")],
        ModelSettings(provider="lmstudio", model="Bionic", stop=["###", "END"]),
    )
    body = _request_body_bytes(route.calls.last.request)
    assert b'"stop"' in body
    assert b"###" in body
    assert b"END" in body


def test_model_settings_parses_stop_and_seed() -> None:
    settings = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        seed="",
        stop="###, END",
    )
    assert settings.seed is None
    assert settings.stop == ["###", "END"]


def test_model_settings_rejects_bad_temperature() -> None:
    with pytest.raises(Exception):
        ModelSettings(provider="lmstudio", model="Bionic", temperature=3.0)


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (503, True),
        (429, True),
        (400, False),
        (401, False),
        (404, False),
    ],
)
def test_is_retryable_by_status(status: int, expected: bool) -> None:
    assert _is_retryable_openai_error(_api_error(status)) is expected


def test_is_retryable_timeout() -> None:
    assert _is_retryable_openai_error(APITimeoutError(request=_request())) is True


def test_is_context_overflow_detects_markers() -> None:
    req = _request()
    try:
        exc = BadRequestError(
            "maximum context length exceeded",
            response=httpx.Response(400, request=req),
            body={"error": {"code": "context_length_exceeded"}},
        )
    except TypeError:
        exc = _api_error(400, "context_length_exceeded")  # type: ignore[assignment]
    assert _is_context_overflow(exc) is True


def test_is_context_overflow_ignores_other_400() -> None:
    try:
        exc = BadRequestError(
            "invalid temperature",
            response=httpx.Response(400, request=_request()),
            body=None,
        )
    except TypeError:
        exc = _api_error(400, "invalid temperature")  # type: ignore[assignment]
    assert _is_context_overflow(exc) is False
