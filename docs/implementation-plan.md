# План реализации AI-агента: FastAPI + Chainlit

**Вердикт:** реализуемо одним разработчиком за **2–3 недели** при строгом приоритете «сначала LM Studio на Windows, потом DeepSeek и Docker». Ниже — поэтапный план без лишней инфраструктуры.

**Стек:** FastAPI (Python 3.11+) · Chainlit · async LLM-клиенты · Docker / docker-compose  
**Приоритет:** LM Studio `http://localhost:1234/v1` → DeepSeek → Docker с `host.docker.internal`

**Оценка зрелости (ревью): 8.5 / 10** — сильные стороны плана сохранены; ниже зафиксированы обязательные доработки по блокерам (контекст, retries, лимиты токенов, observability).

---

### Поправки по результатам ревью (обязательно в MVP)

| # | Блокер / слабое место | Решение в плане |
|---|----------------------|-----------------|
| 1 | Неограниченная история | `truncate_messages()` в `AgentService` перед каждым вызовом LLM |
| 2 | Нет retries | `tenacity` на `chat` / `stream_chat` (5xx, timeout, connection) |
| 3 | Token bomb через UI | `MAX_ALLOWED_TOKENS` в settings; clamp в `on_settings_update` и на бэкенде |
| 4 | Слабая observability | `configure_logging()` в lifespan **и** в `chainlit/app.py` (fallback `chainlit run`) |
| 5 | `/health` слишком болтливый | В prod — только `status` + `ok`; детали LLM — за флагом `HEALTH_VERBOSE` |
| 6 | mount + `--reload` | С дня 1 готов fallback: `chainlit run` без FastAPI mount |
| 7 | Retries на 400/401/403 | Ретраить только 5xx/429/timeout/connection — предикат по `status_code`, не весь `APIError` |
| 8 | Health захардкожен на LM Studio | Пинг через `create_provider(settings.default_provider)` |
| 9 | Ошибка LLM попадает в history | В историю добавлять assistant только при успешном ответе |
| 10 | truncate не режет «одну огромную пару» | Цикл `while tail and over_budget`; при 1–2 сообщениях — обрезать content последнего user |
| 11 | `_parse_optional_int` → ValueError | try/except + сообщение в UI |
| 12 | Logging только в FastAPI lifespan | Тот же `configure_logging` при импорте `chainlit/app.py` (idempotent) |
| 13 | Health висит до `request_timeout_sec` | `HEALTH_TIMEOUT_SEC` (≈5с) → `create_provider(..., timeout=…)` |
| 14 | Тесты: `respx` vs AsyncMock | Юнит LLM через `respx` (httpx внутри openai SDK); не смешивать стили |
| 15 | Нет unit на retry/truncate | `test_is_retryable_*`, `_is_context_overflow`, полный `test_context_truncate.py` |
| 16 | Хрупкие конструкторы `openai` exceptions | Хелперы-фабрики + fallback на `MagicMock(status_code=…)`; pin `openai` в lockfile |
| 17 | `respx` может не перехватить кастомный transport | Unit: дефолтный `AsyncOpenAI` без custom transport; smoke «тесты зелёные» |
| 18 | `request.read()` vs `request.content` в respx | Предпочитать `request.content`; fallback `read()` только если content пуст |

**Не делаем в MVP (осознанно):** суммаризация истории, LangSmith, slowapi rate-limit (только если UI выйдет наружу), auth.

---

### Микро-риски (контроль при реализации)

| # | Риск | Контроль при коде / тестах |
|---|------|----------------------------|
| M1 | Эвристика `MAX_CONTEXT_CHARS` (~4 символа ≈ 1 токен) занижает токены на коде/мультиязыке → возможен `context_length_exceeded` | В `AgentService` перехватывать `openai.BadRequestError` (HTTP 400); пользователю: «Превышен лимит контекста, начните новый чат». Лог: `status=context_overflow` |
| M2 | Валидация `ModelSettings` вручную в `on_settings_update` рассинхронизируется с UI | MVP: dataclass + ручные проверки ок. При расширении виджетов — перенести `ModelSettings` на `pydantic.BaseModel` с `Field(ge=…, le=…)` и единой точкой валидации |
| M3 | `top_k` в UI/payload без проверки → 400 у LM Studio/DeepSeek | Виджет закомментирован; `_should_include_top_k()`; `ALLOW_TOP_K=false` в Settings; factory → `allow_top_k=...`. В payload только при флаге **и** allowlist |

---

### Этап 1: Анализ и архитектура

#### Диаграмма компонентов (текстовое)

```
┌─────────────────────────────────────────────────────────────┐
│  Клиент (браузер)                                           │
│  Chainlit UI: чат, streaming, chat_settings                 │
└────────────────────────────┬────────────────────────────────┘
                             │ WS/HTTP :8000
┌────────────────────────────▼────────────────────────────────┐
│  App process                                                │
│  ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │ Chainlit     │───▶│ AgentService (оркестрация)       │   │
│  │ handlers     │    │ - история сессии                 │   │
│  └──────────────┘    │ - применение ModelSettings       │   │
│  ┌──────────────┐    └──────────────┬───────────────────┘   │
│  │ FastAPI      │                   │                       │
│  │ /health      │                   ▼                       │
│  └──────────────┘    ┌──────────────────────────────────┐   │
│                      │ LLMProvider (ABC)                │   │
│                      │  ├─ OpenAICompatibleProvider     │   │
│                      │  │   (LM Studio / Ollama)        │   │
│                      │  └─ DeepSeekProvider             │   │
│                      └──────────────┬───────────────────┘   │
│  Settings (pydantic-settings) ◀─────┘                       │
│  .env / env vars                                            │
└────────────────────────────┬────────────────────────────────┘
                             │ httpx async
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
 LM Studio :1234/v1     Ollama :11434/v1     DeepSeek API
 (host / host.docker.internal)
```

**Поток запроса:** `on_message` → clamp `max_tokens` → `truncate_messages` (+ system prompt) → провайдер через фабрику → `stream_chat()` (retry на open) → токены в `cl.Message.stream_token` → лог `llm_call` → `finally: reply.update()`.

#### Структура проекта

```
AiAdventAgent/
├── app/
│   ├── __init__.py
│   ├── main.py                 # точка входа: FastAPI + mount Chainlit
│   ├── config.py               # pydantic-settings
│   ├── api/
│   │   ├── __init__.py
│   │   └── health.py           # GET /health
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── base.py             # ABC + DTO
│   │   ├── openai_compatible.py
│   │   ├── deepseek.py
│   │   └── factory.py          # create_provider(provider_id)
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent.py            # оркестрация + truncate_messages
│   │   └── context.py          # усечение истории (эвристика по сообщениям/токенам)
│   ├── observability/
│   │   ├── __init__.py
│   │   └── logging.py          # JSON / structlog setup
│   ├── chainlit/
│   │   ├── __init__.py
│   │   ├── app.py              # on_chat_start / on_message / settings
│   │   └── settings_schema.py  # ChatSettings widgets
│   └── schemas/
│       ├── __init__.py
│       └── models.py           # ModelSettings, ChatMessage
├── tests/
│   ├── unit/
│   │   ├── test_llm_providers.py
│   │   └── test_context_truncate.py
│   └── smoke/
│       └── test_lmstudio_smoke.py
├── scripts/
│   ├── start.bat               # uvicorn + mount (основной)
│   ├── start-chainlit-only.bat # fallback без mount
│   └── start.sh
├── .chainlit/
│   └── config.toml
├── public/                     # опционально: логотип Chainlit
├── .env.example
├── .env                        # gitignored
├── .gitignore
├── .pre-commit-config.yaml     # ruff + mypy (лёгкий)
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml              # или requirements.txt
├── README.md
└── chainlit.md                 # welcome-текст
```

#### Ключевые модули и ответственность

| Модуль | Ответственность |
|--------|-----------------|
| `config.py` | Env → typed settings: endpoints, keys, defaults, лимиты токенов |
| `llm/base.py` | Контракт: `list_models`, `chat`, `stream_chat` |
| `llm/openai_compatible.py` | LM Studio / Ollama + retries (`tenacity`) |
| `llm/deepseek.py` | DeepSeek (тот же протокол, другой base_url + key) |
| `llm/factory.py` | Выбор провайдера по id из UI/конфига |
| `services/context.py` | `truncate_messages`: сохранить system, обрезать хвост |
| `services/agent.py` | System prompt + truncate + вызов LLM + лог метаданных |
| `observability/logging.py` | JSON-логи LLM-вызовов |
| `chainlit/app.py` | UI-хуки, session state, streaming, clamp настроек |
| `chainlit/settings_schema.py` | Dropdown + слайдеры параметров |
| `api/health.py` | Liveness; verbose-детали только при флаге |
| `main.py` | Сборка ASGI-приложения |

**Правило недели 1:** только `OpenAICompatibleProvider` + LM Studio. DeepSeek — копия с другим `base_url` и `api_key`.

---

### Этап 2: Настройка окружения

#### `pyproject.toml` (основные зависимости)

```toml
[project]
name = "ai-advent-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "fastapi>=0.115.0",
  "uvicorn[standard]>=0.32.0",
  "chainlit>=1.3.0",
  "httpx>=0.27.0",
  "openai>=1.50.0",          # async OpenAI SDK
  "pydantic>=2.9.0",
  "pydantic-settings>=2.5.0",
  "python-dotenv>=1.0.0",
  "tenacity>=9.0.0",         # retries для нестабильных локальных LLM
  "structlog>=24.0.0",       # JSON-логи метаданных вызовов
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0.0",
  "pytest-asyncio>=0.24.0",
  "respx>=0.21.0",           # моки httpx (openai SDK); обязателен для unit LLM
  "ruff>=0.6.0",
  "mypy>=1.11.0",
  "pre-commit>=3.8.0",
]
```

> Не удалять `respx`: юниты LLM идут через HTTP-мок, не через `unittest.mock` на методы SDK.
Альтернатива — `requirements.txt` + `requirements-dev.txt` (проще для Advent-челленджа).

#### Пример `.env` (разработка)

```env
# Режим: local | docker | prod
APP_ENV=local

# UI
CHAINLIT_HOST=0.0.0.0
CHAINLIT_PORT=8000

# Провайдер по умолчанию: lmstudio | ollama | deepseek
DEFAULT_PROVIDER=lmstudio
DEFAULT_MODEL=Bionic

# LM Studio (Windows host)
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio

# Ollama (опционально)
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_API_KEY=ollama

# DeepSeek (шаг 2)
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
DEEPSEEK_API_KEY=

# Параметры модели по умолчанию
DEFAULT_TEMPERATURE=0.7
DEFAULT_TOP_P=0.9
DEFAULT_MAX_TOKENS=2048
DEFAULT_SEED=
REQUEST_TIMEOUT_SEC=120
HEALTH_TIMEOUT_SEC=5

# Защита от token bomb / переполнения контекста
MAX_ALLOWED_TOKENS=4096
MAX_HISTORY_MESSAGES=40
# Грубая эвристика: ~4 символа ≈ 1 токен (достаточно для MVP)
MAX_CONTEXT_CHARS=24000
DEFAULT_SYSTEM_PROMPT=Ты полезный ассистент.

# Observability / health
HEALTH_VERBOSE=true
LOG_JSON=true

# M3: не слать top_k в OpenAI-compatible payload, пока не проверен бэкенд
ALLOW_TOP_K=false
```

> **Consistency gate (M3):** виджет `top_k` в UI закомментирован; в `_build_payload` поле попадает **только** если `ALLOW_TOP_K=true` **и** `provider ∈ SUPPORTS_TOP_K`. Иначе `top_k` в запросе отсутствует, даже если значение есть в session.

Для Docker на Windows хосте:

```env
APP_ENV=docker
LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1
```

#### Инициализация на Windows (без Docker)

```powershell
cd C:\Projects\AiAdventAgent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -e ".[dev]"   # или: pip install -r requirements.txt

copy .env.example .env
# В LM Studio: загрузить Bionic, включить Local Server на :1234

# Проверка API
curl http://localhost:1234/v1/models

# Запуск
.\scripts\start.bat
# или: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Открыть `http://localhost:8000`.

---

### Этап 3: Ядро приложения

#### 3.1 Абстракция LLM

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

#### 3.2 Конфиг

```python
# app/config.py
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    chainlit_host: str = "0.0.0.0"
    chainlit_port: int = 8000

    default_provider: str = "lmstudio"
    default_model: str = "Bionic"

    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_api_key: str = ""

    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_max_tokens: int = 2048
    default_seed: int | None = None
    request_timeout_sec: float = 120.0
    health_timeout_sec: float = 5.0  # короткий ping в /api/health

    # Лимиты безопасности (бэкенд всегда сильнее UI)
    max_allowed_tokens: int = 4096
    max_history_messages: int = 40
    max_context_chars: int = 24_000
    default_system_prompt: str = "Ты полезный ассистент."

    health_verbose: bool = True
    log_json: bool = True
    allow_top_k: bool = False  # M3: False до smoke LM Studio

    @field_validator("default_temperature")
    @classmethod
    def _temp(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature должна быть в [0, 2]")
        return v

    @field_validator("default_max_tokens")
    @classmethod
    def _max_tokens_default(cls, v: int) -> int:
        if v < 1:
            raise ValueError("default_max_tokens должен быть ≥ 1")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

#### 3.3 FastAPI + Chainlit в одном процессе

Рекомендуемый паттерн для Chainlit ≥1.x: монтировать Chainlit в FastAPI.

```python
# app/main.py
from contextlib import asynccontextmanager

from fastapi import FastAPI
from chainlit.utils import mount_chainlit

from app.api.health import router as health_router
from app.config import get_settings
from app.observability.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(json_logs=settings.log_json)  # применяет LOG_JSON
    yield


app = FastAPI(title="AI Advent Agent", lifespan=lifespan)
app.include_router(health_router)

# Chainlit UI на корне; REST остаётся доступным
mount_chainlit(app=app, target="app/chainlit/app.py", path="/")
```

```python
# app/observability/logging.py
import logging
import sys

import structlog

_configured = False


def configure_logging(*, json_logs: bool = True) -> None:
    """Idempotent: безопасно вызывать из lifespan и из chainlit/app.py."""
    global _configured
    if _configured:
        return
    _configured = True

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if json_logs:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
```

```python
# app/api/health.py
from fastapi import APIRouter

from app.config import get_settings
from app.llm.factory import create_provider

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    provider_id = settings.default_provider
    llm_ok = False
    llm_error: str | None = None
    try:
        # Короткий timeout — иначе при down-провайдере health висит до 120с
        provider = create_provider(
            provider_id,
            settings,
            timeout=settings.health_timeout_sec,
        )
        await provider.list_models()
        llm_ok = True
    except Exception as exc:  # noqa: BLE001 — health не должен падать
        llm_error = str(exc)

    if not settings.health_verbose:
        return {"status": "ok" if llm_ok else "degraded"}

    return {
        "status": "ok" if llm_ok else "degraded",
        "env": settings.app_env,
        "provider": provider_id,
        "llm": {"ok": llm_ok, "error": llm_error},
    }
```

#### 3.4 Chainlit handlers

```python
# app/chainlit/app.py
import chainlit as cl

from app.chainlit.settings_schema import build_chat_settings
from app.config import get_settings
from app.llm.base import ChatMessage, ModelSettings
from app.llm.factory import create_provider
from app.observability.logging import configure_logging
from app.services.agent import AgentService, ContextOverflowError

# Важно: при `chainlit run` FastAPI lifespan не выполняется —
# конфигурируем логи здесь (idempotent, дубль с main.py безопасен).
configure_logging(json_logs=get_settings().log_json)


@cl.on_chat_start
async def on_chat_start() -> None:
    settings = get_settings()
    model_settings = ModelSettings(
        provider=settings.default_provider,
        model=settings.default_model,
        temperature=settings.default_temperature,
        top_p=settings.default_top_p,
        max_tokens=settings.default_max_tokens,
        seed=settings.default_seed,
    )
    cl.user_session.set("model_settings", model_settings)
    cl.user_session.set("history", [])

    await cl.ChatSettings(build_chat_settings(model_settings)).send()
    await cl.Message(
        content=f"Готов. Провайдер: `{model_settings.provider}`, модель: `{model_settings.model}`."
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict) -> None:
    current: ModelSettings = cl.user_session.get("model_settings")
    # M2: ручная валидация OK для MVP; при расширении UI → pydantic.BaseModel
    try:
        seed = _parse_optional_int(settings.get("seed"))
    except ValueError:
        await cl.Message(content="seed должен быть целым числом или пустым").send()
        return

    updated = ModelSettings(
        provider=settings.get("provider", current.provider),
        model=settings.get("model", current.model),
        temperature=float(settings.get("temperature", current.temperature)),
        top_p=float(settings.get("top_p", current.top_p)),
        max_tokens=int(settings.get("max_tokens", current.max_tokens)),
        seed=seed,
        top_k=current.top_k,  # M3: виджет скрыт — не читать из UI
    )
    app_settings = get_settings()
    if updated.max_tokens > app_settings.max_allowed_tokens:
        updated.max_tokens = app_settings.max_allowed_tokens
    if not 0.0 <= updated.temperature <= 2.0:
        await cl.Message(content="temperature должна быть в диапазоне 0..2").send()
        return
    if not 0.0 <= updated.top_p <= 1.0:
        await cl.Message(content="top_p должна быть в диапазоне 0..1").send()
        return
    if updated.max_tokens < 1:
        await cl.Message(content="max_tokens должен быть ≥ 1").send()
        return

    cl.user_session.set("model_settings", updated)
    await cl.Message(
        content=f"Настройки обновлены: {updated.provider}/{updated.model}"
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    model_settings: ModelSettings = cl.user_session.get("model_settings")
    history: list[ChatMessage] = cl.user_session.get("history")
    app_settings = get_settings()

    if model_settings.max_tokens > app_settings.max_allowed_tokens:
        model_settings.max_tokens = app_settings.max_allowed_tokens
        cl.user_session.set("model_settings", model_settings)

    history.append(ChatMessage(role="user", content=message.content))
    reply = cl.Message(content="")
    await reply.send()

    provider = create_provider(model_settings.provider, app_settings)
    agent = AgentService(provider, app_settings)

    succeeded = False
    try:
        async for token in agent.astream(history, model_settings):
            await reply.stream_token(token)
        succeeded = True
    except ContextOverflowError as exc:
        # Явно ПЕРЕД except Exception — иначе пользователь не увидит точный текст (M1)
        await reply.stream_token(f"\n\n{exc}")
    except Exception as exc:
        await reply.stream_token(f"\n\nОшибка LLM: {exc}")
    finally:
        await reply.update()

    # Не класть текст ошибки в history — иначе загрязнит следующие запросы
    if succeeded:
        history.append(ChatMessage(role="assistant", content=reply.content))
    cl.user_session.set("history", history)


def _parse_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"ожидалось целое число, получено: {value!r}") from exc
```

> Порядок except критичен: `ContextOverflowError` → затем `Exception`. Ошибочный assistant **не** в history.

```python
# app/services/context.py
from app.llm.base import ChatMessage


def truncate_messages(
    history: list[ChatMessage],
    *,
    max_messages: int,
    max_chars: int,
    system_prompt: str,
) -> list[ChatMessage]:
    """
    Сохраняет system + хвост диалога.
    MVP: молча обрезать старые пары; при одной огромной паре — усечь content.
    """
    system = ChatMessage(role="system", content=system_prompt)
    tail = [m for m in history if m.role != "system"]

    if len(tail) > max_messages:
        tail = tail[-max_messages:]
        if tail and tail[0].role == "assistant":
            tail = tail[1:]

    def total_chars(msgs: list[ChatMessage]) -> int:
        return sum(len(m.content) for m in msgs)

    # Пока хвост длиннее 2 — выкидываем старые пары
    while len(tail) > 2 and total_chars([system, *tail]) > max_chars:
        drop = 1
        if len(tail) >= 2 and tail[0].role == "user" and tail[1].role == "assistant":
            drop = 2
        tail = tail[drop:]

    # Граничный случай: 1–2 огромных сообщения всё ещё > max_chars
    # (раньше while len(tail) > 2 пропускал эту пару → только ContextOverflowError)
    budget = max_chars - len(system.content)
    while tail and total_chars(tail) > budget:
        if len(tail) >= 2:
            if tail[0].role == "user" and tail[1].role == "assistant":
                tail = tail[2:]
            else:
                tail = tail[1:]
            continue
        # Осталось одно сообщение — жёстко режем content (хвост промпта важнее начала)
        only = tail[0]
        keep = max(0, budget - 64)
        if keep <= 0:
            tail = []
            break
        if len(only.content) > keep:
            tail = [ChatMessage(role=only.role, content=only.content[-keep:])]
        break

    return [system, *tail]
```

```python
# app/services/agent.py
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
        messages = truncate_messages(
            history,
            max_messages=self._settings.max_history_messages,
            max_chars=self._settings.max_context_chars,
            system_prompt=self._settings.default_system_prompt,
        )
        safe_settings = ModelSettings(
            provider=settings.provider,
            model=settings.model,
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_tokens=min(settings.max_tokens, self._settings.max_allowed_tokens),
            seed=settings.seed,
            top_k=settings.top_k,
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
```

**Индикаторы:** Chainlit сам показывает typing/streaming при `stream_token`. Для явного «thinking» — `async with cl.Step(name="thinking"):` вокруг вызова LLM (опционально на неделе 2).

**Сбой стрима:** `try/except/finally` + флаг `succeeded` — ошибка в UI, но **не** в session history.

---

### Этап 4: Настройки модели в UI

```python
# app/chainlit/settings_schema.py
from chainlit.input_widget import Select, Slider, TextInput

from app.llm.base import ModelSettings


def build_chat_settings(current: ModelSettings) -> list:
    """Виджеты панели настроек Chainlit."""
    return [
        Select(
            id="provider",
            label="Провайдер",
            values=["lmstudio", "ollama", "deepseek"],
            initial_value=current.provider,
        ),
        TextInput(
            id="model",
            label="Модель",
            initial=current.model,
            placeholder="Bionic / deepseek-chat / ...",
        ),
        # Позже: Select по list_models() — после стабилизации LM Studio
        Slider(
            id="temperature",
            label="Temperature",
            initial=current.temperature,
            min=0.0,
            max=2.0,
            step=0.1,
        ),
        Slider(
            id="top_p",
            label="Top P",
            initial=current.top_p,
            min=0.0,
            max=1.0,
            step=0.05,
        ),
        Slider(
            id="max_tokens",
            label="Max tokens",
            initial=float(min(current.max_tokens, 4096)),  # UI ≤ бэкенд-лимита
            min=64.0,
            max=4096.0,  # не 8192: жёсткий потолок в UI = MAX_ALLOWED_TOKENS
            step=64.0,
        ),
        TextInput(
            id="seed",
            label="Seed (пусто = random)",
            initial="" if current.seed is None else str(current.seed),
        ),
        # M3: top_k скрыт/закомментирован до smoke на LM Studio.
        # После подтверждения, что бэкенд принимает поле без 400 — раскомментировать
        # и добавить провайдер в LLMProvider.SUPPORTS_TOP_K.
        # TextInput(
        #     id="top_k",
        #     label="Top K (опционально)",
        #     initial="" if current.top_k is None else str(current.top_k),
        # ),
    ]
```

**Улучшение недели 2:** при старте чата вызвать `provider.list_models()` и заполнить `Select` списком моделей LM Studio вместо ручного TextInput.

**Валидация (M2):** в MVP диапазоны проверяются в `on_settings_update`. При добавлении новых виджетов — мигрировать `ModelSettings` на `pydantic.BaseModel`:

```python
from pydantic import BaseModel, Field

class ModelSettings(BaseModel):
    provider: str
    model: str
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=1)
    seed: int | None = None
    top_k: int | None = Field(None, ge=1)
```

**Хранение:** только `cl.user_session` (сессия браузера). Persist на диск не делать в MVP.

---

### Этап 5: Деплой

#### Dockerfile (multi-stage, non-root)

```dockerfile
# ===== builder =====
FROM python:3.11-slim AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY app ./app
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -U pip \
    && /opt/venv/bin/pip install --no-cache-dir .

# ===== runtime =====
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_ENV=docker

# non-root
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser .chainlit ./.chainlit
COPY --chown=appuser:appuser chainlit.md ./

USER appuser
EXPOSE 8000

# HEALTHCHECK без curl в slim-образе — через stdlib
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health')"

# uvicorn поднимает FastAPI с примонтированным Chainlit
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### docker-compose.yml

```yaml
services:
  agent:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    environment:
      APP_ENV: docker
      # LM Studio на Windows-хосте
      LMSTUDIO_BASE_URL: http://host.docker.internal:1234/v1
    extra_hosts:
      # Docker Desktop Windows / Linux compatibility
      - "host.docker.internal:host-gateway"
    restart: unless-stopped

  # Опционально: раскомментировать при необходимости
  # ollama:
  #   image: ollama/ollama
  #   ports:
  #     - "11434:11434"
  #   volumes:
  #     - ollama_data:/root/.ollama
  #   restart: unless-stopped

# volumes:
#   ollama_data:
```

#### Скрипты запуска

`scripts/start.bat` (Windows):

```bat
@echo off
cd /d %~dp0..
call .venv\Scripts\activate.bat
if not exist .env copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`scripts/start-chainlit-only.bat` (fallback без mount — с дня 1):

```bat
@echo off
cd /d %~dp0..
call .venv\Scripts\activate.bat
if not exist .env copy .env.example .env
rem Логи: configure_logging вызывается при импорте app/chainlit/app.py
chainlit run app/chainlit/app.py --host 0.0.0.0 --port 8000
```

> При этом режиме FastAPI lifespan **не** выполняется — без module-level `configure_logging` в `app.py` флаг `LOG_JSON` игнорируется.
`scripts/start.sh` (Linux/Docker host):

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
[[ -f .env ]] || cp .env.example .env
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

### Этап 6: Тестирование

#### Юнит-тесты абстракции (через `respx` — httpx внутри openai SDK)

> **Решение по respx:** зависимость оставляем. Тесты бьют в HTTP-слой через `respx`, а не `AsyncMock` на `chat.completions.create` — так ближе к реальному пути openai SDK.

```python
# tests/unit/test_llm_providers.py
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
    return result


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
                    {"index": 0, "message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}
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
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
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
        exc = _api_error(400, "context_length_exceeded")
    assert _is_context_overflow(exc) is True


def test_is_context_overflow_ignores_other_400() -> None:
    try:
        exc = BadRequestError(
            "invalid temperature",
            response=httpx.Response(400, request=_request()),
            body=None,
        )
    except TypeError:
        exc = _api_error(400, "invalid temperature")
    assert _is_context_overflow(exc) is False
```

> **Микро-риски тестов (T1–T3):**
> 1. Сигнатуры `RateLimitError` / `BadRequestError` плавают между минорами `openai` — использовать `_api_error()` + pin версии в lockfile.
> 2. `OpenAICompatibleProvider` в unit **не** передаёт custom `http_client`/`transport` в `AsyncOpenAI` — иначе `respx` не перехватит. Проверка: `pytest tests/unit -q` должен быть зелёным на чистом SDK.
> 3. Тело запроса читать через `_request_body_bytes()` (`.content` → fallback `.read()`), не вызывать `.read()` вслепую.

> Конструкторы исключений openai SDK могут отличаться по минорной версии — при падении сборки подправить `_api_error` / `_request_body_bytes`.

```python
# tests/unit/test_context_truncate.py
from app.llm.base import ChatMessage
from app.services.context import truncate_messages


def test_truncate_keeps_system_and_tail() -> None:
    history = [
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="user", content="u2"),
        ChatMessage(role="assistant", content="a2"),
        ChatMessage(role="user", content="u3"),
        ChatMessage(role="assistant", content="a3"),
    ]
    result = truncate_messages(
        history,
        max_messages=2,
        max_chars=10_000,
        system_prompt="SYS",
    )
    assert result[0].role == "system"
    assert result[0].content == "SYS"
    # хвост из последних max_messages (с поправкой на одинокий assistant)
    assert [m.content for m in result[1:]] == ["u3", "a3"]


def test_truncate_huge_single_pair_trims_content() -> None:
    huge = "x" * 5_000
    history = [
        ChatMessage(role="user", content=huge),
        ChatMessage(role="assistant", content="short"),
    ]
    result = truncate_messages(
        history,
        max_messages=40,
        max_chars=200,
        system_prompt="SYS",
    )
    assert result[0].role == "system"
    assert result[0].content == "SYS"
    total = sum(len(m.content) for m in result)
    assert total <= 200
    # либо выкинули пару, либо обрезали content последнего user
    assert all(m.role != "system" or m.content == "SYS" for m in result)


def test_system_always_present_even_on_empty_history() -> None:
    result = truncate_messages(
        [],
        max_messages=40,
        max_chars=100,
        system_prompt="SYS",
    )
    assert result == [ChatMessage(role="system", content="SYS")]


def test_truncate_does_not_drop_system_when_over_budget() -> None:
    history = [ChatMessage(role="user", content="y" * 10_000)]
    result = truncate_messages(
        history,
        max_messages=40,
        max_chars=100,
        system_prompt="KEEP-ME",
    )
    assert result[0] == ChatMessage(role="system", content="KEEP-ME")
    assert len(result) >= 1
```

#### Smoke с реальным LM Studio

```python
# tests/smoke/test_lmstudio_smoke.py
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
```

Запуск:

```powershell
$env:RUN_LMSTUDIO_SMOKE=1
pytest tests/smoke -q
```

#### Docker → LM Studio на хосте

1. LM Studio: Local Server слушает `0.0.0.0:1234` (не только `127.0.0.1`).
2. `docker compose up --build`
3. `curl http://localhost:8000/api/health` → `"provider": "<default>"`, `"llm": {"ok": true}`
4. Чат в UI, смена temperature, повторный запрос

Если `ok: false` — проверить firewall Windows и `host.docker.internal`.

При `DEFAULT_PROVIDER=deepseek` health должен пинговать DeepSeek, а не LM Studio.

---

### Этап 7: Риски и рекомендации

#### Critical blockers (закрыть до DeepSeek / внешнего доступа)

| Риск | Митигация |
|------|-----------|
| Unbounded context | `truncate_messages` + `MAX_HISTORY_MESSAGES` / `MAX_CONTEXT_CHARS`; system всегда сохраняется |
| Нет retries | `tenacity` на non-stream / `_open_stream`; предикат `_is_retryable_openai_error` (5xx/429/timeout) — **не** весь `APIError` |
| Token bomb через UI | `MAX_ALLOWED_TOKENS`; clamp в settings + AgentService; UI max ≤ лимита |
| Утечка деталей в `/health` | `HEALTH_VERBOSE=false` в prod; путь `/api/health` |
| Health ≠ default provider | `create_provider(settings.default_provider, timeout=health_timeout_sec)` |
| Health hang 120с | Отдельный `HEALTH_TIMEOUT_SEC` (default 5) в factory |
| Ошибка в history | `succeeded`-флаг: assistant в history только при успехе |
| mount + reload | С дня 1: `scripts/start-chainlit-only.bat` → `chainlit run app/chainlit/app.py` |

#### Микро-риски (чеклист при реализации)

| ID | Контроль | Когда проверить |
|----|----------|-----------------|
| **M1** | `BadRequestError` → overflow; unit: `_is_context_overflow` + `_is_retryable_*` (400/401 false, 503/429 true) | `tests/unit/test_llm_providers.py` |
| **M2** | Не дублировать правила валидации; при 3+ виджетах — `ModelSettings` → pydantic; `_parse_optional_int` с try/except | Перед расширением ChatSettings; seed=`abc` → сообщение в UI |
| **M3** | `top_k` не в payload; unit через respx: `assert b"top_k" not in body` | Smoke день 3–5 + `test_payload_omits_top_k_by_default` |
| **M4** | `truncate_messages`: хвост / огромная пара / system всегда | `tests/unit/test_context_truncate.py` |
| **M5** | `configure_logging` в lifespan **и** при импорте `chainlit/app.py` (idempotent) | `start.bat` и `start-chainlit-only.bat` → JSON `llm_call` |
| **M6** | Health: `create_provider(..., timeout=health_timeout_sec)` ≤ 5с | Провайдер down → `/api/health` отвечает за секунды, не 120с |
| **M7** | Юниты LLM только через `respx`, не `AsyncMock` на SDK | HTTP-путь только через `respx`; без мёртвого `MagicMock` в `_api_error` |
| **T1** | Хрупкие конструкторы `openai` exceptions | `_api_error()` + pin `openai` в lockfile; при CI fail — править фабрику |
| **T2** | `respx` ↔ custom transport `AsyncOpenAI` | Unit-провайдер без `http_client=`; `pytest tests/unit` зелёный |
| **T3** | `request.read()` vs `.content` | `_request_body_bytes()` — сначала `.content` |

#### Consistency gate (закрытие микро-несоответствий ревью)

Перед мержем кода сверить — в плане и в коде должно быть одно и то же:

| # | Несоответствие | Каноническое решение в плане |
|---|----------------|------------------------------|
| 1 | UI/`payload` слали `top_k` безусловно | Виджет закомментирован; `_should_include_top_k()`; factory → `allow_top_k` |
| 2 | Только `except Exception` глотал overflow | Сначала `except ContextOverflowError`, потом `except Exception` |
| 3 | Ручная валидация vs pydantic | MVP: ручные проверки (M2); `BaseModel` при расширении UI |
| 4 | Retries на весь `APIError` (вкл. 400) | `_is_retryable_openai_error`: только 5xx / 429 / timeout / connection |
| 5 | Health всегда LM Studio | `create_provider(default_provider).list_models()` |
| 6 | Ошибка в history | `if succeeded: history.append(assistant)` |
| 7 | `structlog` без configure / только FastAPI | `configure_logging` в lifespan **и** module-level в `chainlit/app.py` |
| 8 | truncate пропускает «одну огромную пару» | Второй цикл + trim content последнего сообщения |
| 9 | `_parse_optional_int("abc")` падает | try/except → сообщение в UI |
| 10 | Health висит на `request_timeout_sec` | `timeout=settings.health_timeout_sec` в `create_provider` |
| 11 | `respx` в deps, тесты на AsyncMock | Unit на `respx`; AsyncMock не для LLM HTTP |
| 12 | Нет unit retry/overflow/truncate | Полные `test_llm_providers.py` + `test_context_truncate.py` |
| 13 | Хрупкие ctor openai / respx transport / `.read()` | `_api_error`, без custom transport, `_request_body_bytes` |

Smoke-проверка top_k (до включения в UI):

```powershell
# Ожидание: 200 OK или документированное игнорирование поля — НЕ 400
curl http://localhost:1234/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d "{\"model\":\"Bionic\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":16,\"top_k\":40}"
```

#### Chainlit + FastAPI в одном процессе

| Риск | Митигация |
|------|-----------|
| Конфликт путей (`/` vs `/health`) | REST под `/api/*`; health до mount |
| Двойной uvicorn/chainlit CLI | Один вход: `uvicorn app.main:app`; не запускать оба в prod |
| Hot-reload ломает session | В Docker — без `--reload`; при багах session — fallback CLI |
| Версии Chainlit меняют API `mount_chainlit` | Зафиксировать версию в lockfile; проверить на 1.3.x |

#### Docker на Windows → localhost LLM

| Риск | Митигация |
|------|-----------|
| `localhost` внутри контейнера ≠ хост | Только `host.docker.internal` |
| LM Studio на `127.0.0.1` | В настройках сервера — bind `0.0.0.0` |
| Firewall блокирует Docker | Разрешить входящие на 1234 для Docker subnet |
| WSL2 networking quirks | `extra_hosts: host.docker.internal:host-gateway` |

#### Мониторинг и логирование

- `structlog` через idempotent `configure_logging(json_logs=…)`:
  - FastAPI: lifespan в `main.py`
  - Fallback `chainlit run`: module-level в `chainlit/app.py` (иначе `LOG_JSON` не применяется)
- Событие `llm_call`: `provider`, `model`, `prompt_chars`, `completion_chars`, `max_tokens`, `latency_ms`, `status`.
- Не логировать полный prompt/completion в prod без политики.
- `HEALTHCHECK` в Dockerfile + `/api/health` (провайдер = `default_provider`, timeout = `HEALTH_TIMEOUT_SEC`).
- Таймауты: чат — `REQUEST_TIMEOUT_SEC`; health — `HEALTH_TIMEOUT_SEC` (не смешивать).
- Rate-limit (`slowapi`) — только если UI станет доступен извне (post-MVP).

#### Расширяемость `ModelSettings`

- Базовые поля стабильны: `provider`, `model`, `temperature`, `top_p`, `max_tokens`, `seed`.
- `top_k` — в модели данных есть, в API/UI по умолчанию выключен (M3).
- Провайдер-специфичные параметры (например `mirostat`) — в опциональном `extras: dict[str, Any]`; UI показывает виджеты по `provider`.
- **M2:** при росте числа виджетов мигрировать dataclass → `pydantic.BaseModel`, чтобы диапазоны жили в одном месте, а не в `on_settings_update`.

---

## Решения по вопросам ревью (Q&A)

### 1. Что при 50 итерациях чата?

**Решение MVP:** молча обрезать историю (`truncate_messages`), сохраняя system prompt и хвост диалога. Суммаризация — out of scope. При срабатывании truncation — debug-лог `context_truncated=true` (и опционально `cl.Step`). Если эвристика символов не спасла и API вернул 400 (`context_length_exceeded` и аналоги) — `ContextOverflowError` с текстом «Превышен лимит контекста, начните новый чат» (M1).

### 2. Сбой LM Studio mid-stream?

**Решение:** `try/except/finally` в `on_message` → дописать ошибку в сообщение → `await reply.update()`. Сообщение не остаётся в вечном «печатает…». Retry mid-stream **не** делаем (риск дублей токенов); retry только на открытие стрима.

### 3. Третья модель с `mirostat` и т.п.?

**Решение:** `ModelSettings.extras: dict[str, Any]` + провайдер кладёт extras в payload только если знает ключи. UI: условные виджеты по `provider`. Обратная совместимость базовых полей сохраняется.

### 4. Безопасность `/health` в облаке?

**Решение:**
- Путь `/api/health`, в prod `HEALTH_VERBOSE=false` → только `{"status":"ok"|"degraded"}`.
- Не отдавать base_url, ключи, тексты исключений наружу.
- Auth / rate-limit — когда появится публичный деплой (не Advent MVP).

---

## Календарный план (2–3 недели, 1 разработчик)

| Дни | Фокус | Критерий готовности |
|-----|--------|---------------------|
| **1–2** | Каркас, config, venv, LM Studio smoke, fallback `chainlit run` | `GET /v1/models` + юнит-мок |
| **3–5** | `LLMProvider` + retries + Chainlit streaming; curl-smoke `top_k` (M3) | Диалог с Bionic; обрыв стрима корректно закрывается; решение include/exclude `top_k` |
| **6–7** | `ChatSettings` + clamp `MAX_ALLOWED_TOKENS` | Смена temperature; UI не даёт > лимита |
| **8** | `truncate_messages` + system prompt + `BadRequestError`→overflow (M1) | 50+ сообщений не роняют UI; понятное сообщение при overflow |
| **9** | FastAPI `/api/health` + mount + structlog `llm_call` | health + JSON-логи latency |
| **10–11** | DeepSeek + UI option | Переключение lmstudio ↔ deepseek |
| **12–13** | Dockerfile (HEALTHCHECK) + compose + host.docker.internal | Чат из контейнера к LM Studio |
| **14–15** | Тесты, README, `HEALTH_VERBOSE`, полировка | Smoke + unit зелёные |

**Out of scope для MVP:** RAG, tools/function calling, auth, БД истории, суммаризация контекста, multi-user persist, Prometheus, Kubernetes, публичный rate-limit.

---

## Порядок реализации (строго)

1. Windows + venv + LM Studio `http://localhost:1234/v1`
2. Streaming чат в Chainlit (+ fallback без mount)
3. Retries (`tenacity`) + корректное завершение mid-stream ошибок
4. Session settings + `MAX_ALLOWED_TOKENS` clamp
5. `truncate_messages` + system prompt
6. FastAPI `/api/health` + structlog
7. DeepSeek
8. Docker + `host.docker.internal` + HEALTHCHECK
9. Опциональный Ollama в compose
