# Этап 5: Тесты, Docker и скрипты запуска

**Итоговые файлы:**
- `tests/unit/test_llm_providers.py`
- `tests/unit/test_context_truncate.py`
- `tests/smoke/test_lmstudio_smoke.py`
- `Dockerfile`
- `docker-compose.yml`
- `scripts/start.bat`
- `scripts/start-chainlit-only.bat`
- `scripts/start.sh`

---

### Поправки по результатам ревью (обязательно в MVP) — тесты / Docker

| # | Блокер / слабое место | Решение в плане |
|---|----------------------|-----------------|
| 14 | Тесты: `respx` vs AsyncMock | Юнит LLM через `respx` (httpx внутри openai SDK); не смешивать стили |
| 15 | Нет unit на retry/truncate | `test_is_retryable_*`, `_is_context_overflow`, полный `test_context_truncate.py` |
| 16 | Хрупкие конструкторы `openai` exceptions | Хелперы-фабрики + fallback на `MagicMock(status_code=…)`; pin `openai` в lockfile |
| 17 | `respx` может не перехватить кастомный transport | Unit: дефолтный `AsyncOpenAI` без custom transport; smoke «тесты зелёные» |
| 18 | `request.read()` vs `request.content` в respx | Предпочитать `request.content`; fallback `read()` только если content пуст |

---

### Деплой

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

### Тестирование

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

### Микро-риски (чеклист при реализации)

| ID | Контроль | Когда проверить |
|----|----------|-----------------|
| **M7** | Юниты LLM только через `respx`, не `AsyncMock` на SDK | HTTP-путь только через `respx`; без мёртвого `MagicMock` в `_api_error` |
| **T1** | Хрупкие конструкторы `openai` exceptions | `_api_error()` + pin `openai` в lockfile; при CI fail — править фабрику |
| **T2** | `respx` ↔ custom transport `AsyncOpenAI` | Unit-провайдер без `http_client=`; `pytest tests/unit` зелёный |
| **T3** | `request.read()` vs `.content` | `_request_body_bytes()` — сначала `.content` |

### Consistency gate

| # | Несоответствие | Каноническое решение в плане |
|---|----------------|------------------------------|
| 11 | `respx` в deps, тесты на AsyncMock | Unit на `respx`; AsyncMock не для LLM HTTP |
| 12 | Нет unit retry/overflow/truncate | Полные `test_llm_providers.py` + `test_context_truncate.py` |
| 13 | Хрупкие ctor openai / respx transport / `.read()` | `_api_error`, без custom transport, `_request_body_bytes` |

#### Docker на Windows → localhost LLM

| Риск | Митигация |
|------|-----------|
| `localhost` внутри контейнера ≠ хост | Только `host.docker.internal` |
| LM Studio на `127.0.0.1` | В настройках сервера — bind `0.0.0.0` |
| Firewall блокирует Docker | Разрешить входящие на 1234 для Docker subnet |
| WSL2 networking quirks | `extra_hosts: host.docker.internal:host-gateway` |

### Календарный план (дни 12–15 — Docker / тесты)

| Дни | Фокус | Критерий готовности |
|-----|--------|---------------------|
| **12–13** | Dockerfile (HEALTHCHECK) + compose + host.docker.internal | Чат из контейнера к LM Studio |
| **14–15** | Тесты, README, `HEALTH_VERBOSE`, полировка | Smoke + unit зелёные |

## Порядок реализации (строго) — шаги Docker / тесты

8. Docker + `host.docker.internal` + HEALTHCHECK
9. Опциональный Ollama в compose
