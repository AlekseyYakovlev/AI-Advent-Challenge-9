# Этап 1: Структура проекта, конфиг и логирование

**Итоговые файлы:**
- `app/__init__.py`
- `app/main.py`
- `app/config.py`
- `app/observability/__init__.py`
- `app/observability/logging.py`
- `pyproject.toml`
- `.env.example` / `.env`
- `.gitignore`
- `.pre-commit-config.yaml`
- `.chainlit/config.toml`
- `README.md`
- `chainlit.md`

---

**Вердикт:** реализуемо одним разработчиком за **2–3 недели** при строгом приоритете «сначала LM Studio на Windows, потом DeepSeek и Docker». Ниже — поэтапный план без лишней инфраструктуры.

**Стек:** FastAPI (Python 3.11+) · Chainlit · async LLM-клиенты · Docker / docker-compose  
**Приоритет:** LM Studio `http://localhost:1234/v1` → DeepSeek → Docker с `host.docker.internal`

**Оценка зрелости (ревью): 8.5 / 10** — сильные стороны плана сохранены; ниже зафиксированы обязательные доработки по блокерам (контекст, retries, лимиты токенов, observability).

---

### Поправки по результатам ревью (обязательно в MVP) — скелет / config / logging

| # | Блокер / слабое место | Решение в плане |
|---|----------------------|-----------------|
| 4 | Слабая observability | `configure_logging()` в lifespan **и** в `chainlit/app.py` (fallback `chainlit run`) |
| 12 | Logging только в FastAPI lifespan | Тот же `configure_logging` при импорте `chainlit/app.py` (idempotent) |

**Не делаем в MVP (осознанно):** суммаризация истории, LangSmith, slowapi rate-limit (только если UI выйдет наружу), auth.

---

### Микро-риски (чеклист при реализации)

| ID | Контроль | Когда проверить |
|----|----------|-----------------|
| **M5** | `configure_logging` в lifespan **и** при импорте `chainlit/app.py` (idempotent) | `start.bat` и `start-chainlit-only.bat` → JSON `llm_call` |

---

### Диаграмма компонентов (текстовое)

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

#### Ключевые модули и ответственность (скелет / config / logging)

| Модуль | Ответственность |
|--------|-----------------|
| `config.py` | Env → typed settings: endpoints, keys, defaults, лимиты токенов |
| `observability/logging.py` | JSON-логи LLM-вызовов |
| `main.py` | Сборка ASGI-приложения |

---

### Настройка окружения

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

### Конфиг

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

### FastAPI + Chainlit в одном процессе (точка входа + logging)

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

---

### Мониторинг и логирование

- `structlog` через idempotent `configure_logging(json_logs=…)`:
  - FastAPI: lifespan в `main.py`
  - Fallback `chainlit run`: module-level в `chainlit/app.py` (иначе `LOG_JSON` не применяется)
- Событие `llm_call`: `provider`, `model`, `prompt_chars`, `completion_chars`, `max_tokens`, `latency_ms`, `status`.
- Не логировать полный prompt/completion в prod без политики.
- Таймауты: чат — `REQUEST_TIMEOUT_SEC`; health — `HEALTH_TIMEOUT_SEC` (не смешивать).
- Rate-limit (`slowapi`) — только если UI станет доступен извне (post-MVP).

### Consistency gate (скелет / logging)

| # | Несоответствие | Каноническое решение в плане |
|---|----------------|------------------------------|
| 7 | `structlog` без configure / только FastAPI | `configure_logging` в lifespan **и** module-level в `chainlit/app.py` |

### Календарный план (дни 1–2 — каркас)

| Дни | Фокус | Критерий готовности |
|-----|--------|---------------------|
| **1–2** | Каркас, config, venv, LM Studio smoke, fallback `chainlit run` | `GET /v1/models` + юнит-мок |

**Out of scope для MVP:** RAG, tools/function calling, auth, БД истории, суммаризация контекста, multi-user persist, Prometheus, Kubernetes, публичный rate-limit.

## Порядок реализации (строго) — шаги скелета

1. Windows + venv + LM Studio `http://localhost:1234/v1`
