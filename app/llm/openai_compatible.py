from collections.abc import AsyncIterator
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from app.llm.base import ChatMessage, LLMProvider, ModelSettings


def _is_retryable_openai_error(exc: BaseException) -> bool:
    """Ретраить только транзиентные сбои — НЕ 400/401/403/404.

    BadRequestError наследует APIError: ретрай по типу APIError заставляет
    context-overflow и auth-ошибки повторяться 3 раза зря.
    """
    if isinstance(exc, (APITimeoutError, APIConnectionError, ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, RateLimitError):
        return True  # 429
    if isinstance(exc, APIError):
        code = getattr(exc, "status_code", None)
        # 5xx — да; 4xx (кроме 429 выше) — нет
        return code is not None and 500 <= int(code) < 600
    return False


_RETRY: dict[str, Any] = dict(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception(_is_retryable_openai_error),
    reraise=True,
)


class OpenAICompatibleProvider(LLMProvider):
    """LM Studio / Ollama через OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float = 120.0,
        *,
        allow_top_k: bool = False,
    ) -> None:
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key or "not-needed",
            timeout=timeout,
        )
        # M3: по умолчанию False; True только после smoke LM Studio
        self._allow_top_k = allow_top_k
        # Для unit/respx: не передавать custom http_client/transport в AsyncOpenAI

    @retry(**_RETRY)
    async def list_models(self) -> list[str]:
        models = await self._client.models.list()
        return sorted(m.id for m in models.data)

    @retry(**_RETRY)
    async def chat(self, messages: list[ChatMessage], settings: ModelSettings) -> str:
        resp = await self._client.chat.completions.create(
            **self._build_payload(messages, settings, stream=False)
        )
        return resp.choices[0].message.content or ""

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        settings: ModelSettings,
    ) -> AsyncIterator[str]:
        # Retry только на старте стрима: повторный yield уже отданных токенов
        # сломает UI. После первого chunk ошибки пробрасываем наверх.
        stream = await self._open_stream(messages, settings)
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    @retry(**_RETRY)
    async def _open_stream(
        self,
        messages: list[ChatMessage],
        settings: ModelSettings,
    ) -> Any:
        return await self._client.chat.completions.create(
            **self._build_payload(messages, settings, stream=True)
        )
