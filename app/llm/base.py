from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ModelSettings:
    """Параметры генерации в рамках сессии Chainlit.

    MVP: dataclass + ручная валидация в UI.
    При росте виджетов — мигрировать на pydantic.BaseModel (см. микро-риск M2).
    """

    provider: str
    model: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    seed: int | None = None
    top_k: int | None = None  # не слать в API, пока не подтверждён бэкенд (M3)


@dataclass(slots=True)
class ChatMessage:
    role: str  # system | user | assistant
    content: str


class LLMProvider(ABC):
    """Единый интерфейс для LM Studio / Ollama / DeepSeek."""

    # Провайдеры, для которых top_k безопасно включать в payload
    SUPPORTS_TOP_K: frozenset[str] = frozenset()  # пусто по умолчанию (M3)

    @abstractmethod
    async def list_models(self) -> list[str]:
        ...

    @abstractmethod
    async def chat(
        self,
        messages: list[ChatMessage],
        settings: ModelSettings,
    ) -> str:
        ...

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[ChatMessage],
        settings: ModelSettings,
    ) -> AsyncIterator[str]:
        ...

    def _build_payload(
        self,
        messages: list[ChatMessage],
        settings: ModelSettings,
        *,
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": settings.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
            "stream": stream,
        }
        if settings.seed is not None:
            payload["seed"] = settings.seed
        # M3: top_k НЕ добавлять «если не None». Только флаг + allowlist.
        if self._should_include_top_k(settings):
            payload["top_k"] = settings.top_k
        return payload

    def _should_include_top_k(self, settings: ModelSettings) -> bool:
        """Единая проверка M3 — вызывается из _build_payload."""
        allow = getattr(self, "_allow_top_k", False)
        return (
            bool(allow)
            and settings.provider in self.SUPPORTS_TOP_K
            and settings.top_k is not None
        )
