from collections.abc import AsyncIterator

import pytest

from app.chainlit.settings_schema import build_chat_settings, resolve_exclusive_modes
from app.config import Settings
from app.llm.base import (
    DEFAULT_EXPERTS_CONFIG,
    EXPERT_PANEL_PROMPT_TEMPLATE,
    ChatMessage,
    LLMProvider,
    ModelSettings,
)
from app.services.agent import (
    AgentService,
    build_expert_panel_system_prompt,
    format_experts_list,
    parse_experts,
)


class _CaptureProvider(LLMProvider):
    def __init__(self) -> None:
        self.last_messages: list[ChatMessage] = []

    async def list_models(self) -> list[str]:
        return []

    async def chat(
        self,
        messages: list[ChatMessage],
        settings: ModelSettings,
    ) -> str:
        self.last_messages = messages
        return "ok"

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        settings: ModelSettings,
    ) -> AsyncIterator[str]:
        self.last_messages = messages
        yield "ok"


def test_parse_experts_multiline_russian() -> None:
    config = (
        "1. Роль: Аналитик\n"
        "Описание: Факты и риски.\n"
        "2. Роль: Инженер\n"
        "Описание: Решения.\n"
        "3. Роль: Критик\n"
        "Описание: Слабые места."
    )
    experts = parse_experts(config)
    assert [e["role"] for e in experts] == ["Аналитик", "Инженер", "Критик"]
    assert experts[0]["description"] == "Факты и риски"
    assert experts[1]["description"] == "Решения"
    assert experts[2]["description"] == "Слабые места"


def test_parse_experts_inline_and_missing_description() -> None:
    config = (
        "1. Role: Analyst. Description: Metrics.\n"
        "2. Роль: Инженер\n"
        "3. Role: Critic. Description: Risks."
    )
    experts = parse_experts(config)
    assert experts[0] == {"role": "Analyst", "description": "Metrics"}
    assert experts[1]["role"] == "Инженер"
    assert "технические решения" in experts[1]["description"]
    assert experts[2] == {"role": "Critic", "description": "Risks"}


def test_parse_experts_skips_duplicates_and_uses_defaults() -> None:
    experts = parse_experts("")
    assert len(experts) == 3
    assert experts[0]["role"] == "Аналитик"

    experts = parse_experts(
        "1. Роль: Аналитик\n"
        "Описание: Первое.\n"
        "2. Роль: аналитик\n"
        "Описание: Дубликат."
    )
    assert len(experts) == 1
    assert experts[0]["description"] == "Первое"


def test_format_experts_list() -> None:
    text = format_experts_list(
        [
            {"role": "A", "description": "desc A"},
            {"role": "B", "description": "desc B"},
        ]
    )
    assert text == (
        "1. Role: A. Description: desc A.\n"
        "2. Role: B. Description: desc B."
    )


def test_model_settings_experts_defaults_and_serialization() -> None:
    settings = ModelSettings(provider="lmstudio", model="Bionic")
    assert settings.expert_panel_enabled is False
    assert settings.experts_config == DEFAULT_EXPERTS_CONFIG

    payload = settings.model_dump()
    restored = ModelSettings.model_validate(payload)
    assert restored.experts_config == settings.experts_config
    assert "Аналитик" in restored.experts_config


def test_build_chat_settings_includes_expert_widgets_first() -> None:
    widgets = build_chat_settings(
        ModelSettings(provider="lmstudio", model="Bionic")
    )
    ids = [getattr(w, "id", None) for w in widgets]
    assert ids[0] == "expert_panel_enabled"
    assert ids[1] == "experts_config"
    assert "provider" in ids
    assert "system_prompt" in ids
    assert "expert_count" not in ids
    assert "expert_0_role" not in ids


def test_resolve_exclusive_modes_enabling_expert_disables_others() -> None:
    current = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        step_by_step=True,
        pre_generated_prompt=True,
        expert_panel_enabled=False,
    )
    step, pre_gen, expert = resolve_exclusive_modes(
        current=current,
        step_by_step=True,
        pre_generated_prompt=True,
        expert_panel_enabled=True,
    )
    assert (step, pre_gen, expert) == (False, False, True)

    widgets = build_chat_settings(
        ModelSettings(
            provider="lmstudio",
            model="Bionic",
            step_by_step=step,
            pre_generated_prompt=pre_gen,
            expert_panel_enabled=expert,
        )
    )
    by_id = {getattr(w, "id", None): w for w in widgets}
    assert by_id["expert_panel_enabled"].initial is True
    assert by_id["step_by_step"].initial is False
    assert by_id["pre_generated_prompt"].initial is False


def test_resolve_exclusive_modes_enabling_step_or_pregen_disables_expert() -> None:
    current = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        expert_panel_enabled=True,
    )
    step, pre_gen, expert = resolve_exclusive_modes(
        current=current,
        step_by_step=True,
        pre_generated_prompt=False,
        expert_panel_enabled=True,
    )
    assert (step, pre_gen, expert) == (True, False, False)

    step2, pre_gen2, expert2 = resolve_exclusive_modes(
        current=current,
        step_by_step=False,
        pre_generated_prompt=True,
        expert_panel_enabled=True,
    )
    assert (step2, pre_gen2, expert2) == (False, True, False)


def test_resolve_exclusive_modes_steady_state_keeps_expert_exclusive() -> None:
    # Уже включён expert; step/pre_gen тоже True (битое состояние) —
    # при отсутствии нового включения expert остаётся единственным.
    current = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        step_by_step=True,
        pre_generated_prompt=True,
        expert_panel_enabled=True,
    )
    step, pre_gen, expert = resolve_exclusive_modes(
        current=current,
        step_by_step=True,
        pre_generated_prompt=True,
        expert_panel_enabled=True,
    )
    assert (step, pre_gen, expert) == (False, False, True)

    # Чистый steady-state только с expert.
    clean = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        expert_panel_enabled=True,
    )
    step2, pre_gen2, expert2 = resolve_exclusive_modes(
        current=clean,
        step_by_step=False,
        pre_generated_prompt=False,
        expert_panel_enabled=True,
    )
    assert (step2, pre_gen2, expert2) == (False, False, True)


@pytest.mark.asyncio
async def test_expert_panel_replaces_system_prompt() -> None:
    provider = _CaptureProvider()
    agent = AgentService(provider, Settings())
    config = (
        "1. Роль: Аналитик\n"
        "Описание: Факты\n"
        "2. Роль: Критик\n"
        "Описание: Риски"
    )
    settings = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        system_prompt="Base prompt. MUST BE IGNORED.",
        expert_panel_enabled=True,
        experts_config=config,
        step_by_step=True,  # тоже не должен влиять в режиме экспертов
    )

    tokens = [
        token
        async for token in agent.astream(
            [ChatMessage(role="user", content="Как снизить churn?")],
            settings,
        )
    ]

    assert tokens == ["ok"]
    system = provider.last_messages[0].content
    expected = build_expert_panel_system_prompt(config, "Как снизить churn?")
    assert provider.last_messages[0].role == "system"
    assert system == expected
    assert "Base prompt" not in system
    assert "step-by-step approach" not in system
    assert "<experts_panel>" in system
    assert "<context_and_task>" in system
    assert "Как снизить churn?" in system
    assert "1. Role: Аналитик. Description: Факты." in system
    assert "elite Moderator" in system
    assert system == EXPERT_PANEL_PROMPT_TEMPLATE.format(
        EXPERTS_LIST=format_experts_list(parse_experts(config)),
        USER_TASK="Как снизить churn?",
    )


@pytest.mark.asyncio
async def test_expert_panel_disabled_keeps_system_prompt_clean() -> None:
    provider = _CaptureProvider()
    agent = AgentService(provider, Settings())
    settings = ModelSettings(
        provider="lmstudio",
        model="Bionic",
        system_prompt="Base prompt.",
        expert_panel_enabled=False,
        experts_config=DEFAULT_EXPERTS_CONFIG,
    )

    async for _ in agent.astream(
        [ChatMessage(role="user", content="hi")],
        settings,
    ):
        pass

    system = provider.last_messages[0].content
    assert system == "Base prompt."
    assert "<experts_panel>" not in system
    assert "elite Moderator" not in system
