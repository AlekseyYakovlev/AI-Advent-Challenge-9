from typing import Any

from chainlit.input_widget import Select, Slider, Switch, TextInput

from app.llm.base import DEFAULT_EXPERTS_CONFIG, ModelSettings


EXCLUSIVE_MODE_KEYS = (
    "step_by_step",
    "pre_generated_prompt",
    "expert_panel_enabled",
)


def resolve_exclusive_modes(
    *,
    current: ModelSettings,
    step_by_step: bool,
    pre_generated_prompt: bool,
    expert_panel_enabled: bool,
) -> tuple[bool, bool, bool]:
    """Взаимно исключает «Группу экспертов» и step-by-step / pre-gen.

    Включение expert panel гасит step_by_step и pre_generated_prompt.
    Включение step_by_step или pre_generated_prompt гасит expert panel.
    Приоритет у режима, который только что включили.
    """
    expert_just_on = expert_panel_enabled and not current.expert_panel_enabled
    step_just_on = step_by_step and not current.step_by_step
    pre_gen_just_on = pre_generated_prompt and not current.pre_generated_prompt

    if expert_just_on:
        return False, False, True
    if step_just_on or pre_gen_just_on:
        return step_by_step, pre_generated_prompt, False
    if expert_panel_enabled:
        return False, False, True
    return step_by_step, pre_generated_prompt, False


def exclusive_modes_snapshot(
    step_by_step: bool,
    pre_generated_prompt: bool,
    expert_panel_enabled: bool,
) -> dict[str, bool]:
    """Снимок взаимоисключающих переключателей для сравнения в on_settings_edit."""
    return {
        "step_by_step": step_by_step,
        "pre_generated_prompt": pre_generated_prompt,
        "expert_panel_enabled": expert_panel_enabled,
    }


def draft_settings_from_ui(
    settings: dict[str, object],
    current: ModelSettings,
    *,
    step_by_step: bool,
    pre_generated_prompt: bool,
    expert_panel_enabled: bool,
) -> ModelSettings:
    """Собирает ModelSettings из UI-формы с уже разрешёнными exclusive-флагами."""
    experts_raw = settings.get("experts_config", current.experts_config)
    experts_config = (
        DEFAULT_EXPERTS_CONFIG if experts_raw is None else str(experts_raw)
    )
    return current.model_copy(
        update={
            "provider": str(settings.get("provider", current.provider)),
            "model": str(settings.get("model", current.model)),
            "temperature": settings.get("temperature", current.temperature),
            "top_p": settings.get("top_p", current.top_p),
            "max_tokens": settings.get("max_tokens", current.max_tokens),
            "seed": settings.get("seed", current.seed),
            "system_prompt": (
                ""
                if settings.get("system_prompt", current.system_prompt) is None
                else str(settings.get("system_prompt", current.system_prompt))
            ),
            "stop": settings.get("stop", current.stop),
            "step_by_step": step_by_step,
            "pre_generated_prompt": pre_generated_prompt,
            "expert_panel_enabled": expert_panel_enabled,
            "experts_config": experts_config,
        }
    )


def build_chat_settings(current: ModelSettings) -> list[Any]:
    """Виджеты панели настроек Chainlit."""
    experts_config = current.experts_config or DEFAULT_EXPERTS_CONFIG

    return [
        Switch(
            id="expert_panel_enabled",
            label="Группа экспертов",
            initial=current.expert_panel_enabled,
        ),
        TextInput(
            id="experts_config",
            label="Роли экспертов",
            multiline=True,
            initial=experts_config,
            placeholder=(
                "1. Роль: X\n"
                "Описание: Y\n"
                "2. Роль: ...\n"
                "Описание: ..."
            ),
        ),
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
        TextInput(
            id="system_prompt",
            label="System prompt",
            initial=current.system_prompt,
            placeholder="Инструкция для модели…",
            multiline=True,
        ),
        Switch(
            id="step_by_step",
            label="step-by-step approach",
            initial=current.step_by_step,
        ),
        Switch(
            id="pre_generated_prompt",
            label="Сначала составить промпт, затем ответить",
            initial=current.pre_generated_prompt,
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
            label="Ограничение длины ответа (max tokens)",
            initial=float(min(current.max_tokens, 4096)),  # UI ≤ бэкенд-лимита
            min=64.0,
            max=4096.0,  # не 8192: жёсткий потолок в UI = MAX_ALLOWED_TOKENS
            step=64.0,
        ),
        TextInput(
            id="stop",
            label="Stop sequence",
            initial=", ".join(current.stop) if current.stop else "",
            placeholder="через запятую, напр. ###, END",
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
