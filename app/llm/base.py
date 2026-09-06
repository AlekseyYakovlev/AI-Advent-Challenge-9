from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, field_validator

STEP_BY_STEP_INSTRUCTION = (
    "Please use a step-by-step approach. For each step, briefly explain "
    "your reasoning before moving to the next one. Finally, summarize "
    "the solution at the end."
)

DEFAULT_EXPERTS_CONFIG = (
    "1. Роль: Аналитик\n"
    "Описание: Разбирает задачу на части, выявляет ключевые факты, "
    "риски и допущения.\n"
    "2. Роль: Инженер\n"
    "Описание: Предлагает практичные технические решения "
    "и шаги реализации.\n"
    "3. Роль: Критик\n"
    "Описание: Ищет слабые места, противоречия и упущенные риски "
    "в аргументах."
)

EXPERT_PANEL_PROMPT_TEMPLATE = """You are an elite Moderator of a closed expert panel. Your task is to orchestrate a highly structured debate among the defined experts regarding the user's problem, and then synthesize a balanced, practical, and actionable final solution.

<instructions>
1. Analyze the user's task.
2. Read the <experts_panel> list. You will simulate each expert strictly adhering to their defined Role and Description.
3. Conduct a 3-stage discussion: "Positions", "Crossfire", and "Moderator's Final Verdict".
4. CRITICAL: Keep the language of the output identical to the language of the <context_and_task> and <experts_panel>.
</instructions>

<experts_panel>
{EXPERTS_LIST}
</experts_panel>

<context_and_task>
{USER_TASK}
</context_and_task>

<discussion_rules>
1. **Positions**: Each expert provides a concise perspective (2-4 sentences) strictly based on their domain.
2. **Crossfire**: Experts must ask one challenging but constructive question to an opponent, exposing weaknesses in their reasoning.
3. **Moderator's Verdict**: You analyze all arguments, discard emotional bias, and formulate a balanced, step-by-step final decision.
</discussion_rules>

<output_format>
### 💬 Expert Positions
- **[Role 1]**: [Position]
- **[Role 2]**: [Position]
- **[Role 3]**: [Position]

### 🔥 Crossfire (Critique & Questions)
- **[Role X] → [Role Y]**: [Challenging question/critique]
- **[Role Y] → [Role Z]**: [Challenging question/critique]

### 🎯 Moderator's Final Verdict
[Structured, step-by-step action plan that integrates the best ideas and mitigates identified risks. No fluff.]
</output_format>

<few_shot_example>
### 💬 Expert Positions
- **[Analyst]**: Based on current market metrics, the churn rate suggests a UX bottleneck in the onboarding flow. We need data-driven A/B testing before any rewrite.
- **[Engineer]**: Rewriting the frontend is technically feasible but resource-intensive. Our current architecture supports modular microservices, so a partial rewrite is safer.
- **[Critic]**: Relying on A/B tests assumes we have enough traffic for statistical significance. What if the underlying tech debt causes a total outage before the tests finish?

### 🔥 Crossfire (Critique & Questions)
- **[Critic] → [Analyst]**: How exactly do you plan to isolate UX variables from backend latency issues in your A/B tests?
- **[Engineer] → [Critic]**: You mention total outage risks, but aren't you ignoring the automated failovers we implemented last quarter?

### 🎯 Moderator's Final Verdict
1. Run a 2-week diagnostic on backend latency vs. frontend drop-off rates.
2. Implement modular UI updates for the onboarding screen instead of a full rewrite.
3. Establish a fallback monitoring system to catch tech-debt-related outages.
</few_shot_example>
"""

EXPERT_PANEL_ACTIVE_NOTICE = (
    "ℹ️ Режим 'Группа экспертов' активирован. Ваш кастомный System Prompt "
    "временно игнорируется, модель работает в роли Модератора."
)


class ModelSettings(BaseModel):
    """Параметры генерации в рамках сессии Chainlit."""

    provider: str
    model: str
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=1)
    seed: int | None = 42
    top_k: int | None = Field(None, ge=1)  # не слать в API, пока не подтверждён бэкенд (M3)
    system_prompt: str = ""
    stop: list[str] | None = None
    step_by_step: bool = False
    pre_generated_prompt: bool = False
    expert_panel_enabled: bool = False
    experts_config: str = DEFAULT_EXPERTS_CONFIG

    @field_validator("seed", mode="before")
    @classmethod
    def _coerce_seed(cls, value: object) -> int | None:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("seed должен быть целым числом или пустым")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return int(value)
        raise ValueError(f"ожидалось целое число, получено: {value!r}")

    @field_validator("stop", mode="before")
    @classmethod
    def _coerce_stop(cls, value: object) -> list[str] | None:
        if value is None or value == "":
            return None
        if isinstance(value, list):
            parts = [str(item).strip() for item in value]
        else:
            parts = [part.strip() for part in str(value).split(",")]
        cleaned = [part for part in parts if part]
        return cleaned or None

    @field_validator("experts_config", mode="before")
    @classmethod
    def _coerce_experts_config(cls, value: object) -> str:
        if value is None:
            return DEFAULT_EXPERTS_CONFIG
        return str(value)


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
        if settings.stop:
            payload["stop"] = settings.stop
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
