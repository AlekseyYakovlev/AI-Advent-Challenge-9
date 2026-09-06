import re
import time
from collections.abc import AsyncIterator

import structlog
from openai import BadRequestError

from app.config import Settings
from app.llm.base import (
    DEFAULT_EXPERTS_CONFIG,
    EXPERT_PANEL_PROMPT_TEMPLATE,
    STEP_BY_STEP_INSTRUCTION,
    ChatMessage,
    LLMProvider,
    ModelSettings,
)
from app.services.context import truncate_messages

log = structlog.get_logger()

# Типичные маркеры переполнения контекста в теле 400
_CONTEXT_OVERFLOW_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "context window",
    "too many tokens",
)

_DEFAULT_ROLE_DESCRIPTIONS: dict[str, str] = {
    "аналитик": (
        "Разбирает задачу на части, выявляет ключевые факты, "
        "риски и допущения."
    ),
    "инженер": (
        "Предлагает практичные технические решения "
        "и шаги реализации."
    ),
    "критик": (
        "Ищет слабые места, противоречия и упущенные риски "
        "в аргументах."
    ),
    "analyst": (
        "Breaks the task into parts and identifies key facts, "
        "risks, and assumptions."
    ),
    "engineer": (
        "Proposes practical technical solutions "
        "and implementation steps."
    ),
    "critic": (
        "Finds weak spots, contradictions, and missed risks "
        "in the arguments."
    ),
}

_ROLE_RE = re.compile(
    r"(?im)^\s*(?:\d+[.)]\s*)?(?:роль|role)\s*:\s*(.+?)\s*$"
)
_DESC_RE = re.compile(
    r"(?im)^\s*(?:описание|description)\s*:\s*(.+?)\s*$"
)
# Однострочный формат: "1. Role: X. Description: Y." / "Роль: X. Описание: Y."
_INLINE_EXPERT_RE = re.compile(
    r"(?im)(?:\d+[.)]\s*)?(?:роль|role)\s*:\s*(.+?)\s*[.]\s*"
    r"(?:описание|description)\s*:\s*(.+?)(?:\s*[.]?\s*$)"
)


def parse_experts(experts_config: str) -> list[dict[str, str]]:
    """Парсит многострочный конфиг ролей в список {role, description}."""
    text = (experts_config or "").strip() or DEFAULT_EXPERTS_CONFIG
    experts: list[dict[str, str]] = []
    seen_roles: set[str] = set()

    current_role: str | None = None
    current_description = ""

    def _flush() -> None:
        nonlocal current_role, current_description
        if not current_role:
            return
        role_key = current_role.casefold()
        if role_key in seen_roles:
            current_role = None
            current_description = ""
            return
        description = current_description.strip()
        if not description:
            description = _DEFAULT_ROLE_DESCRIPTIONS.get(role_key, "")
        experts.append({"role": current_role, "description": description})
        seen_roles.add(role_key)
        current_role = None
        current_description = ""

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        inline = _INLINE_EXPERT_RE.fullmatch(line)
        if inline:
            _flush()
            current_role = inline.group(1).strip().rstrip(".")
            current_description = inline.group(2).strip().rstrip(".")
            _flush()
            continue

        role_match = _ROLE_RE.match(line)
        if role_match:
            _flush()
            current_role = role_match.group(1).strip().rstrip(".")
            current_description = ""
            continue

        desc_match = _DESC_RE.match(line)
        if desc_match and current_role is not None:
            current_description = desc_match.group(1).strip().rstrip(".")
            continue

        # Продолжение описания на следующей строке без префикса
        if current_role is not None and current_description:
            current_description = f"{current_description} {line}".strip()

    _flush()
    return experts


def format_experts_list(experts: list[dict[str, str]]) -> str:
    """Нумерованный список ролей для вставки в <experts_panel>."""
    if not experts:
        experts = parse_experts(DEFAULT_EXPERTS_CONFIG)
    return "\n".join(
        f"{index}. Role: {item['role']}. Description: {item['description']}."
        for index, item in enumerate(experts, start=1)
    )


def build_expert_panel_system_prompt(
    experts_config: str,
    user_task: str,
) -> str:
    """Рендерит полный system prompt режима «Группа экспертов»."""
    experts = parse_experts(experts_config)
    return EXPERT_PANEL_PROMPT_TEMPLATE.format(
        EXPERTS_LIST=format_experts_list(experts),
        USER_TASK=user_task.strip(),
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
        if settings.expert_panel_enabled:
            user_task = ""
            for message in reversed(history):
                if message.role == "user":
                    user_task = message.content
                    break
            system_prompt = build_expert_panel_system_prompt(
                settings.experts_config,
                user_task,
            )
        else:
            system_prompt = settings.system_prompt or self._settings.default_system_prompt
            if settings.step_by_step:
                system_prompt = (
                    f"{system_prompt}\n\n{STEP_BY_STEP_INSTRUCTION}"
                    if system_prompt
                    else STEP_BY_STEP_INSTRUCTION
                )

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
