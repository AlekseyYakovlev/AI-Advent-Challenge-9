# Этап 3: AgentService, контекст и ContextOverflowError

**Итоговые файлы:**
- `app/services/__init__.py`
- `app/services/agent.py`
- `app/services/context.py`

---

### Поправки по результатам ревью (обязательно в MVP) — services / context

| # | Блокер / слабое место | Решение в плане |
|---|----------------------|-----------------|
| 1 | Неограниченная история | `truncate_messages()` в `AgentService` перед каждым вызовом LLM |
| 9 | Ошибка LLM попадает в history | В историю добавлять assistant только при успешном ответе |
| 10 | truncate не режет «одну огромную пару» | Цикл `while tail and over_budget`; при 1–2 сообщениях — обрезать content последнего user |

---

### Микро-риски (контроль при реализации)

| # | Риск | Контроль при коде / тестах |
|---|------|----------------------------|
| M1 | Эвристика `MAX_CONTEXT_CHARS` (~4 символа ≈ 1 токен) занижает токены на коде/мультиязыке → возможен `context_length_exceeded` | В `AgentService` перехватывать `openai.BadRequestError` (HTTP 400); пользователю: «Превышен лимит контекста, начните новый чат». Лог: `status=context_overflow` |

---

#### Ключевые модули и ответственность (services)

| Модуль | Ответственность |
|--------|-----------------|
| `services/context.py` | `truncate_messages`: сохранить system, обрезать хвост |
| `services/agent.py` | System prompt + truncate + вызов LLM + лог метаданных |

---

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

### Critical blockers (context / agent)

| Риск | Митигация |
|------|-----------|
| Unbounded context | `truncate_messages` + `MAX_HISTORY_MESSAGES` / `MAX_CONTEXT_CHARS`; system всегда сохраняется |
| Token bomb через UI | `MAX_ALLOWED_TOKENS`; clamp в settings + AgentService; UI max ≤ лимита |
| Ошибка в history | `succeeded`-флаг: assistant в history только при успехе |

### Микро-риски (чеклист при реализации)

| ID | Контроль | Когда проверить |
|----|----------|-----------------|
| **M1** | `BadRequestError` → overflow; unit: `_is_context_overflow` + `_is_retryable_*` (400/401 false, 503/429 true) | `tests/unit/test_llm_providers.py` |
| **M4** | `truncate_messages`: хвост / огромная пара / system всегда | `tests/unit/test_context_truncate.py` |

### Consistency gate

| # | Несоответствие | Каноническое решение в плане |
|---|----------------|------------------------------|
| 2 | Только `except Exception` глотал overflow | Сначала `except ContextOverflowError`, потом `except Exception` |
| 6 | Ошибка в history | `if succeeded: history.append(assistant)` |
| 8 | truncate пропускает «одну огромную пару» | Второй цикл + trim content последнего сообщения |

### Решения по вопросам ревью (Q&A)

### 1. Что при 50 итерациях чата?

**Решение MVP:** молча обрезать историю (`truncate_messages`), сохраняя system prompt и хвост диалога. Суммаризация — out of scope. При срабатывании truncation — debug-лог `context_truncated=true` (и опционально `cl.Step`). Если эвристика символов не спасла и API вернул 400 (`context_length_exceeded` и аналоги) — `ContextOverflowError` с текстом «Превышен лимит контекста, начните новый чат» (M1).

### Календарный план (день 8 — truncate / overflow)

| Дни | Фокус | Критерий готовности |
|-----|--------|---------------------|
| **8** | `truncate_messages` + system prompt + `BadRequestError`→overflow (M1) | 50+ сообщений не роняют UI; понятное сообщение при overflow |

## Порядок реализации (строго) — шаги services

4. Session settings + `MAX_ALLOWED_TOKENS` clamp
5. `truncate_messages` + system prompt
