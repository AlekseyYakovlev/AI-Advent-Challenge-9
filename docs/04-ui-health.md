# Этап 4: Chainlit UI и FastAPI /api/health

**Итоговые файлы:**
- `app/chainlit/__init__.py`
- `app/chainlit/app.py`
- `app/chainlit/settings_schema.py`
- `app/api/__init__.py`
- `app/api/health.py`

---

### Поправки по результатам ревью (обязательно в MVP) — UI / health

| # | Блокер / слабое место | Решение в плане |
|---|----------------------|-----------------|
| 3 | Token bomb через UI | `MAX_ALLOWED_TOKENS` в settings; clamp в `on_settings_update` и на бэкенде |
| 5 | `/health` слишком болтливый | В prod — только `status` + `ok`; детали LLM — за флагом `HEALTH_VERBOSE` |
| 6 | mount + `--reload` | С дня 1 готов fallback: `chainlit run` без FastAPI mount |
| 8 | Health захардкожен на LM Studio | Пинг через `create_provider(settings.default_provider)` |
| 11 | `_parse_optional_int` → ValueError | try/except + сообщение в UI |
| 13 | Health висит до `request_timeout_sec` | `HEALTH_TIMEOUT_SEC` (≈5с) → `create_provider(..., timeout=…)` |

---

### Микро-риски (контроль при реализации)

| # | Риск | Контроль при коде / тестах |
|---|------|----------------------------|
| M2 | Валидация `ModelSettings` вручную в `on_settings_update` рассинхронизируется с UI | MVP: dataclass + ручные проверки ок. При расширении виджетов — перенести `ModelSettings` на `pydantic.BaseModel` с `Field(ge=…, le=…)` и единой точкой валидации |

---

#### Ключевые модули и ответственность (UI / health)

| Модуль | Ответственность |
|--------|-----------------|
| `chainlit/app.py` | UI-хуки, session state, streaming, clamp настроек |
| `chainlit/settings_schema.py` | Dropdown + слайдеры параметров |
| `api/health.py` | Liveness; verbose-детали только при флаге |

---

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

### Chainlit handlers

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

---

### Настройки модели в UI

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

### Critical blockers (UI / health)

| Риск | Митигация |
|------|-----------|
| Утечка деталей в `/health` | `HEALTH_VERBOSE=false` в prod; путь `/api/health` |
| Health ≠ default provider | `create_provider(settings.default_provider, timeout=health_timeout_sec)` |
| Health hang 120с | Отдельный `HEALTH_TIMEOUT_SEC` (default 5) в factory |
| mount + reload | С дня 1: `scripts/start-chainlit-only.bat` → `chainlit run app/chainlit/app.py` |

### Микро-риски (чеклист при реализации)

| ID | Контроль | Когда проверить |
|----|----------|-----------------|
| **M2** | Не дублировать правила валидации; при 3+ виджетах — `ModelSettings` → pydantic; `_parse_optional_int` с try/except | Перед расширением ChatSettings; seed=`abc` → сообщение в UI |
| **M5** | `configure_logging` в lifespan **и** при импорте `chainlit/app.py` (idempotent) | `start.bat` и `start-chainlit-only.bat` → JSON `llm_call` |
| **M6** | Health: `create_provider(..., timeout=health_timeout_sec)` ≤ 5с | Провайдер down → `/api/health` отвечает за секунды, не 120с |

### Consistency gate

| # | Несоответствие | Каноническое решение в плане |
|---|----------------|------------------------------|
| 3 | Ручная валидация vs pydantic | MVP: ручные проверки (M2); `BaseModel` при расширении UI |
| 5 | Health всегда LM Studio | `create_provider(default_provider).list_models()` |
| 9 | `_parse_optional_int("abc")` падает | try/except → сообщение в UI |
| 10 | Health висит на `request_timeout_sec` | `timeout=settings.health_timeout_sec` в `create_provider` |

#### Chainlit + FastAPI в одном процессе

| Риск | Митигация |
|------|-----------|
| Конфликт путей (`/` vs `/health`) | REST под `/api/*`; health до mount |
| Двойной uvicorn/chainlit CLI | Один вход: `uvicorn app.main:app`; не запускать оба в prod |
| Hot-reload ломает session | В Docker — без `--reload`; при багах session — fallback CLI |
| Версии Chainlit меняют API `mount_chainlit` | Зафиксировать версию в lockfile; проверить на 1.3.x |

### Решения по вопросам ревью (Q&A)

### 2. Сбой LM Studio mid-stream?

**Решение:** `try/except/finally` в `on_message` → дописать ошибку в сообщение → `await reply.update()`. Сообщение не остаётся в вечном «печатает…». Retry mid-stream **не** делаем (риск дублей токенов); retry только на открытие стрима.

### 4. Безопасность `/health` в облаке?

**Решение:**
- Путь `/api/health`, в prod `HEALTH_VERBOSE=false` → только `{"status":"ok"|"degraded"}`.
- Не отдавать base_url, ключи, тексты исключений наружу.
- Auth / rate-limit — когда появится публичный деплой (не Advent MVP).

### Календарный план (дни 6–7, 9 — UI / health)

| Дни | Фокус | Критерий готовности |
|-----|--------|---------------------|
| **6–7** | `ChatSettings` + clamp `MAX_ALLOWED_TOKENS` | Смена temperature; UI не даёт > лимита |
| **9** | FastAPI `/api/health` + mount + structlog `llm_call` | health + JSON-логи latency |

## Порядок реализации (строго) — шаги UI / health

2. Streaming чат в Chainlit (+ fallback без mount)
4. Session settings + `MAX_ALLOWED_TOKENS` clamp
6. FastAPI `/api/health` + structlog

### Расширяемость `ModelSettings`

- **M2:** при росте числа виджетов мигрировать dataclass → `pydantic.BaseModel`, чтобы диапазоны жили в одном месте, а не в `on_settings_update`.

### Мониторинг и логирование (health)

- `HEALTHCHECK` в Dockerfile + `/api/health` (провайдер = `default_provider`, timeout = `HEALTH_TIMEOUT_SEC`).
- Таймауты: чат — `REQUEST_TIMEOUT_SEC`; health — `HEALTH_TIMEOUT_SEC` (не смешивать).
