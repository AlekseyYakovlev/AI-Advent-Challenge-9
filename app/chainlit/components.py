"""UI-хелперы Chainlit для LM Studio (Lazy UI Sync, без broadcast)."""

from __future__ import annotations

from typing import Any

import chainlit as cl

from app.schemas.lmstudio import ModelLoadStatus
from app.services.lmstudio_state import LMStudioState

ACTION_SELECT_MODEL = "lmstudio_select_model"
MAX_CHAT_MODEL_ACTIONS = 12

GLOBAL_STATE_WARNING = (
    "⚠️ Состояние модели общее для всех сессий сервера"
)

CIRCUIT_OPEN_CHAT_TEXT = (
    "🔒 Сервис временно недоступен. Повторите позже."
)

CIRCUIT_OPEN_SETTINGS_TEXT = (
    "Сервис временно недоступен. "
    "Следующая попытка примерно через 2 минуты."
)

REFRESH_THROTTLED_TEXT = (
    "Обновление списка доступно не чаще 1 раза в 5 секунд"
)

STATUS_LABELS: dict[ModelLoadStatus, str] = {
    ModelLoadStatus.IDLE: "⚪ Модель не выбрана",
    ModelLoadStatus.LOADING: "🟡 Загрузка...",
    ModelLoadStatus.LOADED: "🟢 Загружена",
    ModelLoadStatus.ERROR: "🔴 Ошибка загрузки",
    ModelLoadStatus.UNREACHABLE: "⚪ Сервис недоступен",
    ModelLoadStatus.CIRCUIT_OPEN: CIRCUIT_OPEN_CHAT_TEXT,
}


def format_status_label(status: ModelLoadStatus) -> str:
    return STATUS_LABELS.get(status, str(status.value))


def format_global_warning() -> str:
    return GLOBAL_STATE_WARNING


def format_circuit_open_message(*, for_settings: bool = False) -> str:
    return CIRCUIT_OPEN_SETTINGS_TEXT if for_settings else CIRCUIT_OPEN_CHAT_TEXT


def actions_blocked(state: LMStudioState) -> bool:
    return state.status in (
        ModelLoadStatus.LOADING,
        ModelLoadStatus.CIRCUIT_OPEN,
    )


def build_model_actions(
    state: LMStudioState,
    *,
    current_model: str | None = None,
) -> list[cl.Action]:
    """Кнопки выбора модели для сообщения в чате. Пусто если действия заблокированы."""
    if actions_blocked(state):
        return []

    models = list(state.available_models)
    if current_model and current_model not in models:
        models = [current_model, *models]

    actions: list[cl.Action] = []
    for model_id in models[:MAX_CHAT_MODEL_ACTIONS]:
        label = f"✓ {model_id}" if model_id == current_model else model_id
        actions.append(
            cl.Action(
                name=ACTION_SELECT_MODEL,
                payload={"model_id": model_id},
                label=label,
                tooltip=f"Загрузить {model_id}",
            )
        )
    return actions


def build_switcher_content(
    *,
    provider: str,
    session_model: str,
    state: LMStudioState | None,
) -> str:
    """Текст системного сообщения-переключателя."""
    if provider != "lmstudio":
        return (
            f"**Модель:** `{session_model}`\n"
            f"Провайдер: `{provider}` (управление LM Studio скрыто)."
        )

    if state is None:
        return (
            f"**LM Studio**\n"
            f"Модель сессии: `{session_model}`\n"
            f"{format_global_warning()}"
        )

    lines = [
        "**LM Studio — переключатель моделей**",
        f"Модель сессии: `{session_model}`",
        f"На сервере: `{state.current_model or '—'}`",
        f"Статус: {format_status_label(state.status)}",
    ]
    if state.message and state.status in (
        ModelLoadStatus.ERROR,
        ModelLoadStatus.UNREACHABLE,
        ModelLoadStatus.CIRCUIT_OPEN,
    ):
        lines.append(f"Детали: {state.message}")
    if state.status == ModelLoadStatus.LOADING:
        lines.append("⏳ Дождитесь завершения загрузки перед сменой модели.")
    if state.status == ModelLoadStatus.CIRCUIT_OPEN:
        lines.append(format_circuit_open_message(for_settings=False))
    lines.append(format_global_warning())
    if (
        state.available_models
        and len(state.available_models) > MAX_CHAT_MODEL_ACTIONS
        and not actions_blocked(state)
    ):
        lines.append(
            f"_В чате показаны первые {MAX_CHAT_MODEL_ACTIONS} моделей; "
            "полный список — в Settings._"
        )
    return "\n".join(lines)


async def send_model_status_message(
    *,
    provider: str,
    session_model: str,
    state: LMStudioState | None,
) -> cl.Message:
    """Показать статус (без кнопок) — короткий снимок."""
    if provider == "lmstudio" and state is not None:
        content = (
            f"{format_status_label(state.status)}\n"
            f"Модель: `{state.current_model or session_model}`\n"
            f"{format_global_warning()}"
        )
    else:
        content = f"Модель: `{session_model}` (`{provider}`)"
    msg = cl.Message(content=content)
    await msg.send()
    return msg


async def send_or_update_model_switcher(
    *,
    provider: str,
    session_model: str,
    state: LMStudioState | None,
    existing: cl.Message | None = None,
) -> cl.Message:
    """Переключатель в чате: новое сообщение или update существующего."""
    content = build_switcher_content(
        provider=provider,
        session_model=session_model,
        state=state,
    )
    actions: list[cl.Action] = []
    if provider == "lmstudio" and state is not None:
        actions = build_model_actions(state, current_model=session_model)

    if existing is not None:
        existing.content = content
        existing.actions = actions
        await existing.update()
        return existing

    msg = cl.Message(content=content, actions=actions)
    await msg.send()
    return msg


def lmstudio_settings_widgets(
    *,
    current_model: str,
    state: LMStudioState,
) -> list[Any]:
    """Виджеты Settings только для провайдера lmstudio."""
    from chainlit.input_widget import Select, Switch, TextInput

    blocked = actions_blocked(state)
    circuit = state.status == ModelLoadStatus.CIRCUIT_OPEN
    loading = state.status == ModelLoadStatus.LOADING

    models = list(state.available_models)
    if current_model and current_model not in models:
        models = [current_model, *models]
    if not models:
        models = [current_model or "Bionic"]

    initial_model = current_model if current_model in models else models[0]

    widgets: list[Any] = [
        TextInput(
            id="lmstudio_status",
            label="Статус LM Studio",
            initial=format_status_label(state.status),
            description=state.message or format_global_warning(),
            disabled=True,
        ),
        TextInput(
            id="lmstudio_global_warning",
            label="Важно",
            initial=format_global_warning(),
            disabled=True,
        ),
    ]

    if circuit:
        widgets.append(
            TextInput(
                id="lmstudio_circuit_notice",
                label="Circuit Breaker",
                initial=format_circuit_open_message(for_settings=True),
                disabled=True,
            )
        )
    elif loading:
        widgets.append(
            TextInput(
                id="lmstudio_loading_notice",
                label="Прогресс",
                initial="🟡 Идёт загрузка модели. Выбор другой модели временно недоступен.",
                disabled=True,
            )
        )

    widgets.append(
        Select(
            id="model",
            label="Модель LM Studio",
            values=models,
            initial_value=initial_model,
            description=format_global_warning(),
            disabled=blocked,
        )
    )
    widgets.append(
        Switch(
            id="lmstudio_refresh_models",
            label="🔄 Обновить список",
            initial=False,
            description=(
                format_circuit_open_message(for_settings=True)
                if circuit
                else "Принудительно обновить список моделей с сервера"
            ),
            disabled=circuit,
        )
    )
    return widgets
