from typing import Any

from chainlit.input_widget import Select, Slider, TextInput

from app.llm.base import ModelSettings


def build_chat_settings(current: ModelSettings) -> list[Any]:
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
