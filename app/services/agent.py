import time
from collections.abc import AsyncIterator

import structlog
from openai import BadRequestError

from app.config import Settings
from app.llm.base import ChatMessage, LLMProvider, ModelSettings
from app.services.context import truncate_messages

log = structlog.get_logger()

# Типичные маркеры переполнения контекста в теле 400
_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
)


class ContextOverflowError(Exception):
    """Превышен лимит контекста модели — нужен новый чат."""


class AgentService:
    def __init__(self, provider: LLMProvider, settings: Settings) -> None:
        self._provider = provider
        self._settings = settings

    async def astream(
        self,
        history: list[ChatMessage],
        settings: ModelSettings,
    ) -> AsyncIterator[str]:
        system_prompt = settings.system_prompt or self._settings.default_system_prompt
        messages = truncate_messages(
            history,
            max_messages=self._settings.max_history_messages,
            max_chars=self._settings.max_context_chars,
            system_prompt=system_prompt,
        )
        safe_settings = settings.model_copy(
            update={
                "max_tokens": min(settings.max_tokens, self._settings.max_allowed_tokens),
                "system_prompt": system_prompt,
            }
        )

        started = time.perf_counter()
        status = "success"
        completion_chars = 0
        try:
            async for token in self._provider.stream_chat(messages, safe_settings):
                completion_chars += len(token)
                yield token
        except BadRequestError as exc:
            # M1: эвристика символов неточна (код/мультиязык) → 400 от модели
            if _is_context_overflow(exc):
                status = "context_overflow"
                raise ContextOverflowError(
                    "Превышен лимит контекста, начните новый чат"
                ) from exc
            status = "error"
            raise
        except Exception:
            status = "error"
            raise
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            log.info(
                "llm_call",
                provider=safe_settings.provider,
                model=safe_settings.model,
                prompt_chars=sum(len(m.content) for m in messages),
                completion_chars=completion_chars,
                max_tokens=safe_settings.max_tokens,
                latency_ms=latency_ms,
                status=status,
            )


def _is_context_overflow(exc: BadRequestError) -> bool:
    text = (getattr(exc, "message", None) or str(exc)).lower()
    body = str(getattr(exc, "body", "") or "").lower()
    return any(m in text or m in body for m in _CONTEXT_OVERFLOW_MARKERS)
