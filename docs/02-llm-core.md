# Этап 2: Абстракция LLM, провайдер и retries

**Итоговые файлы:**
- `app/llm/__init__.py`
- `app/llm/base.py`
- `app/llm/openai_compatible.py`
- `app/llm/deepseek.py`
- `app/llm/factory.py`
- `app/schemas/__init__.py`
- `app/schemas/models.py`

---

### Поправки по результатам ревью (обязательно в MVP) — LLM / retries

| # | Блокер / слабое место | Решение в плане |
|---|----------------------|-----------------|
| 2 | Нет retries | `tenacity` на `chat` / `stream_chat` (5xx, timeout, connection) |
| 7 | Retries на 400/401/403 | Ретраить только 5xx/429/timeout/connection — предикат по `status_code`, не весь `APIError` |

---

### Микро-риски (контроль при реализации)

| # | Риск | Контроль при коде / тестах |
|---|------|----------------------------|
| M3 | `top_k` в UI/payload без проверки → 400 у LM Studio/DeepSeek | Виджет закомментирован; `_should_include_top_k()`; `ALLOW_TOP_K=false` в Settings; factory → `allow_top_k=...`. В payload только при флаге **и** allowlist |

---

#### Ключевые модули и ответственность (LLM)

| Модуль | Ответственность |
|--------|-----------------|
| `llm/base.py` | Контракт: `list_models`, `chat`, `stream_chat` |
| `llm/openai_compatible.py` | LM Studio / Ollama + retries (`tenacity`) |
| `llm/deepseek.py` | DeepSeek (тот же протокол, другой base_url + key) |
| `llm/factory.py` | Выбор провайдера по id из UI/конфига |

**Правило недели 1:** только `OpenAICompatibleProvider` + LM Studio. DeepSeek — копия с другим `base_url` и `api_key`.

---

### 3.1 Абстракция LLM

```python
# app/llm/base.py
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
```

```python
# app/llm/openai_compatible.py
from collections.abc import AsyncIterator

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


_RETRY = dict(
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
    async def _open_stream(self, messages: list[ChatMessage], settings: ModelSettings):
        return await self._client.chat.completions.create(
            **self._build_payload(messages, settings, stream=True)
        )
```

> **Критично:** не использовать `retry_if_exception_type(APIError)` — `BadRequestError` (400) и auth 401/403 попадут в ретраи. Только предикат по `status_code` ∈ 5xx + 429 + timeout/connection.

> **Важно:** `@retry` на самом `async for` стриме опасен (дубли токенов). Рестарт — только до первого chunk (`_open_stream`).

`DeepSeekProvider` на шаге 2 — тот же класс с другим `base_url`/`api_key`, либо тонкая обёртка над `OpenAICompatibleProvider`. Отдельный класс нужен только если появятся различия в параметрах/ошибках.

```python
# app/llm/factory.py
from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.openai_compatible import OpenAICompatibleProvider


def create_provider(
    provider_id: str,
    settings: Settings,
    *,
    timeout: float | None = None,
) -> LLMProvider:
    """timeout=None → settings.request_timeout_sec; для health передавать короткий."""
    mapping = {
        "lmstudio": (settings.lmstudio_base_url, settings.lmstudio_api_key),
        "ollama": (settings.ollama_base_url, settings.ollama_api_key),
        "deepseek": (settings.deepseek_base_url, settings.deepseek_api_key),
    }
    if provider_id not in mapping:
        raise ValueError(f"Неизвестный провайдер: {provider_id}")
    base_url, api_key = mapping[provider_id]
    if provider_id == "deepseek" and not api_key:
        raise ValueError("DEEPSEEK_API_KEY не задан")
    return OpenAICompatibleProvider(
        base_url,
        api_key,
        timeout if timeout is not None else settings.request_timeout_sec,
        allow_top_k=settings.allow_top_k,
    )
```

---

### Critical blockers (LLM / retries)

| Риск | Митигация |
|------|-----------|
| Нет retries | `tenacity` на non-stream / `_open_stream`; предикат `_is_retryable_openai_error` (5xx/429/timeout) — **не** весь `APIError` |

### Микро-риски (чеклист при реализации)

| ID | Контроль | Когда проверить |
|----|----------|-----------------|
| **M3** | `top_k` не в payload; unit через respx: `assert b"top_k" not in body` | Smoke день 3–5 + `test_payload_omits_top_k_by_default` |

### Consistency gate

| # | Несоответствие | Каноническое решение в плане |
|---|----------------|------------------------------|
| 1 | UI/`payload` слали `top_k` безусловно | Виджет закомментирован; `_should_include_top_k()`; factory → `allow_top_k` |
| 4 | Retries на весь `APIError` (вкл. 400) | `_is_retryable_openai_error`: только 5xx / 429 / timeout / connection |

Smoke-проверка top_k (до включения в UI):

```powershell
# Ожидание: 200 OK или документированное игнорирование поля — НЕ 400
curl http://localhost:1234/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"Bionic\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16,\"top_k\":40}"
```

### Расширяемость `ModelSettings`

- Базовые поля стабильны: `provider`, `model`, `temperature`, `top_p`, `max_tokens`, `seed`.
- `top_k` — в модели данных есть, в API/UI по умолчанию выключен (M3).
- Провайдер-специфичные параметры (например `mirostat`) — в опциональном `extras: dict[str, Any]`; UI показывает виджеты по `provider`.

### Решения по вопросам ревью (Q&A)

### 3. Третья модель с `mirostat` и т.п.?

**Решение:** `ModelSettings.extras: dict[str, Any]` + провайдер кладёт extras в payload только если знает ключи. UI: условные виджеты по `provider`. Обратная совместимость базовых полей сохраняется.

### Календарный план (дни 3–5, 10–11 — LLM)

| Дни | Фокус | Критерий готовности |
|-----|--------|---------------------|
| **3–5** | `LLMProvider` + retries + Chainlit streaming; curl-smoke `top_k` (M3) | Диалог с Bionic; обрыв стрима корректно закрывается; решение include/exclude `top_k` |
| **10–11** | DeepSeek + UI option | Переключение lmstudio ↔ deepseek |

## Порядок реализации (строго) — шаги LLM

2. Streaming чат в Chainlit (+ fallback без mount)
3. Retries (`tenacity`) + корректное завершение mid-stream ошибок
7. DeepSeek
