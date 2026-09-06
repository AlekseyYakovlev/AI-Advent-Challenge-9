from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator

STEP_BY_STEP_INSTRUCTION = (
    "Please use a step-by-step approach. For each step, briefly explain "
    "your reasoning before moving to the next one. Finally, summarize "
    "the solution at the end."
)


class ModelSettings(BaseModel):
    """Параметры генерации в рамках сессии Chainlit."""

    provider: str
    model: str
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=1)
    seed: int | None = 42
    top_k: int | None = Field(None, ge=1)  # не слать в API, пока не подтверждён бэкенд (M3)
    system_prompt: str = "Ты полезный ассистент."
    stop: list[str] | None = None
    step_by_step: bool = False

    @field_validator("seed", mode="before")
    @classmethod
    def _coerce_seed(cls, value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("seed должен быть целым числом или пустым")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value)
        raise ValueError(f"ожидалось целое число, получено: {value!r}")

    @field_validator("stop", mode="before")
    @classmethod
    def _coerce_stop(cls, value: object) -> list[str] | None:
        if value is None or value == "":
            return None
        if isinstance(value, list):
            parts = [str(item).strip() for item in value]
        else:
            parts = [part.strip() for part in str(value).split(",")]
        cleaned = [part for part in parts if part]
        return cleaned or None


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
        if settings.stop:
            payload["stop"] = settings.stop
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
