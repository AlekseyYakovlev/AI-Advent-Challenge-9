import chainlit as cl
from pydantic import ValidationError

from app.chainlit.settings_schema import (
    build_chat_settings,
    draft_settings_from_ui,
    exclusive_modes_snapshot,
    resolve_exclusive_modes,
)
from app.config import Settings, get_settings
from app.llm.base import (
    DEFAULT_EXPERTS_CONFIG,
    EXPERT_PANEL_ACTIVE_NOTICE,
    ChatMessage,
    LLMProvider,
    ModelSettings,
)
from app.llm.factory import create_provider
from app.observability.logging import configure_logging
from app.services.agent import AgentService, ContextOverflowError
from app.services.context import truncate_messages

# Важно: при `chainlit run` FastAPI lifespan не выполняется —
# конфигурируем логи здесь (idempotent, дубль с main.py безопасен).
configure_logging(json_logs=get_settings().log_json)

# Компактная EN-обёртка: без отдельного system — экономия токенов.
PROMPT_GEN_USER_TEMPLATE = (
    'Generate a prompt to solve the following question: "{question}". '
    'Return only the prompt.'
)


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
        system_prompt=settings.default_system_prompt,
        step_by_step=False,
        pre_generated_prompt=False,
        expert_panel_enabled=False,
        experts_config=DEFAULT_EXPERTS_CONFIG,
    )
    cl.user_session.set("model_settings", model_settings)
    cl.user_session.set("history", [])
    cl.user_session.set("expert_panel_notice_sent", False)
    cl.user_session.set(
        "exclusive_modes_snapshot",
        exclusive_modes_snapshot(
            step_by_step=False,
            pre_generated_prompt=False,
            expert_panel_enabled=False,
        ),
    )

    await cl.ChatSettings(build_chat_settings(model_settings)).send()
    await cl.Message(
        content=(
            f"Готов. Провайдер: `{model_settings.provider}`, "
            f"модель: `{model_settings.model}`."
        )
    ).send()


def _resolve_modes_against_baseline(
    settings: dict[str, object],
    baseline: ModelSettings,
) -> tuple[bool, bool, bool]:
    return resolve_exclusive_modes(
        current=baseline,
        step_by_step=bool(settings.get("step_by_step", baseline.step_by_step)),
        pre_generated_prompt=bool(
            settings.get("pre_generated_prompt", baseline.pre_generated_prompt)
        ),
        expert_panel_enabled=bool(
            settings.get("expert_panel_enabled", baseline.expert_panel_enabled)
        ),
    )


@cl.on_settings_edit
async def on_settings_edit(settings: dict[str, object]) -> None:
    """Живое обновление UI: гасит конфликтующие Switch сразу при клике."""
    current: ModelSettings = cl.user_session.get("model_settings")
    snapshot = cl.user_session.get("exclusive_modes_snapshot")
    baseline = (
        current.model_copy(update=snapshot)
        if isinstance(snapshot, dict)
        else current
    )

    incoming_step = bool(settings.get("step_by_step", baseline.step_by_step))
    incoming_pre = bool(
        settings.get("pre_generated_prompt", baseline.pre_generated_prompt)
    )
    incoming_expert = bool(
        settings.get("expert_panel_enabled", baseline.expert_panel_enabled)
    )
    step_by_step, pre_generated_prompt, expert_panel_enabled = (
        _resolve_modes_against_baseline(settings, baseline)
    )

    new_snapshot = exclusive_modes_snapshot(
        step_by_step, pre_generated_prompt, expert_panel_enabled
    )
    cl.user_session.set("exclusive_modes_snapshot", new_snapshot)

    # Без изменений — не дергаем refresh (защита от лишних циклов).
    if (step_by_step, pre_generated_prompt, expert_panel_enabled) == (
        incoming_step,
        incoming_pre,
        incoming_expert,
    ):
        return

    try:
        draft = draft_settings_from_ui(
            settings,
            current,
            step_by_step=step_by_step,
            pre_generated_prompt=pre_generated_prompt,
            expert_panel_enabled=expert_panel_enabled,
        )
    except (ValidationError, ValueError, TypeError):
        return

    # refresh() пушит виджеты в открытую панель Settings, не коммитя session.
    await cl.ChatSettings(build_chat_settings(draft)).refresh()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, object]) -> None:
    current: ModelSettings = cl.user_session.get("model_settings")
    step_by_step, pre_generated_prompt, expert_panel_enabled = (
        _resolve_modes_against_baseline(settings, current)
    )

    try:
        updated = draft_settings_from_ui(
            settings,
            current,
            step_by_step=step_by_step,
            pre_generated_prompt=pre_generated_prompt,
            expert_panel_enabled=expert_panel_enabled,
        )
    except (ValidationError, ValueError, TypeError) as exc:
        await cl.Message(content=f"Некорректные настройки: {exc}").send()
        return

    app_settings = get_settings()
    if updated.max_tokens > app_settings.max_allowed_tokens:
        updated = updated.model_copy(
            update={"max_tokens": app_settings.max_allowed_tokens}
        )

    cl.user_session.set("model_settings", updated)
    cl.user_session.set(
        "exclusive_modes_snapshot",
        exclusive_modes_snapshot(
            updated.step_by_step,
            updated.pre_generated_prompt,
            updated.expert_panel_enabled,
        ),
    )
    await cl.ChatSettings(build_chat_settings(updated)).send()

    notes: list[str] = [
        f"Настройки обновлены: {updated.provider}/{updated.model}"
    ]
    if expert_panel_enabled and (
        current.step_by_step or current.pre_generated_prompt
    ):
        notes.append(
            "ℹ️ step-by-step и «Сначала составить промпт…» отключены — "
            "они несовместимы с «Группой экспертов»."
        )
    if (step_by_step or pre_generated_prompt) and current.expert_panel_enabled:
        notes.append(
            "ℹ️ «Группа экспертов» отключена — режим несовместим с "
            "step-by-step / генерацией промпта."
        )
    await cl.Message(content="\n".join(notes)).send()

    if updated.expert_panel_enabled:
        await cl.Message(content=EXPERT_PANEL_ACTIVE_NOTICE).send()
        cl.user_session.set("expert_panel_notice_sent", True)
    else:
        cl.user_session.set("expert_panel_notice_sent", False)


def wrap_user_question_for_prompt_generation(question: str) -> str:
    """Оборачивает исходный вопрос для первого (prompt-gen) запроса к LLM."""
    return PROMPT_GEN_USER_TEMPLATE.format(question=question)


async def generate_improved_prompt(
    history: list[ChatMessage],
    model_settings: ModelSettings,
    provider: LLMProvider,
    app_settings: Settings,
) -> str:
    """Этап 1: non-streaming генерация улучшенного промпта."""
    prompt_gen_settings = model_settings.model_copy(
        update={
            "system_prompt": "",  # только user-обёртка, без лишнего system
            "step_by_step": False,
            "expert_panel_enabled": False,  # этап 1 — обычная генерация промпта
        }
    )
    # В API уходит обёрнутый вопрос, history по-прежнему хранит исходный текст
    # до успешной замены на сгенерированный промпт.
    if not history or history[-1].role != "user":
        raise ValueError("Для генерации промпта нужен последний user-сообщение")
    prompt_history = [
        *history[:-1],
        ChatMessage(
            role="user",
            content=wrap_user_question_for_prompt_generation(history[-1].content),
        ),
    ]
    # provider.chat не инжектит system_prompt — готовим messages как AgentService
    messages = truncate_messages(
        prompt_history,
        max_messages=app_settings.max_history_messages,
        max_chars=app_settings.max_context_chars,
        system_prompt=prompt_gen_settings.system_prompt,
    )
    return (await provider.chat(messages, prompt_gen_settings)).strip()


async def _maybe_notify_expert_panel(model_settings: ModelSettings) -> None:
    """Сообщает пользователю, что кастомный system prompt игнорируется."""
    if not model_settings.expert_panel_enabled:
        return
    if cl.user_session.get("expert_panel_notice_sent"):
        return
    await cl.Message(content=EXPERT_PANEL_ACTIVE_NOTICE).send()
    cl.user_session.set("expert_panel_notice_sent", True)


@cl.on_message
async def on_message(message: cl.Message) -> None:
    model_settings: ModelSettings = cl.user_session.get("model_settings")
    history: list[ChatMessage] = cl.user_session.get("history")
    app_settings = get_settings()

    if model_settings.max_tokens > app_settings.max_allowed_tokens:
        model_settings = model_settings.model_copy(
            update={"max_tokens": app_settings.max_allowed_tokens}
        )
        cl.user_session.set("model_settings", model_settings)

    await _maybe_notify_expert_panel(model_settings)

    provider = create_provider(model_settings.provider, app_settings)
    agent = AgentService(provider, app_settings)

    if model_settings.pre_generated_prompt:
        await _on_message_with_pre_generated_prompt(
            message=message,
            history=history,
            model_settings=model_settings,
            provider=provider,
            agent=agent,
            app_settings=app_settings,
        )
        return

    history.append(ChatMessage(role="user", content=message.content))
    reply = cl.Message(content="")
    await reply.send()

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


async def _on_message_with_pre_generated_prompt(
    *,
    message: cl.Message,
    history: list[ChatMessage],
    model_settings: ModelSettings,
    provider: LLMProvider,
    agent: AgentService,
    app_settings: Settings,
) -> None:
    history.append(ChatMessage(role="user", content=message.content))

    try:
        generated_prompt = await generate_improved_prompt(
            history, model_settings, provider, app_settings
        )
    except ContextOverflowError as exc:
        await cl.Message(content=f"⚠️ {exc}").send()
        cl.user_session.set("history", history)
        return
    except Exception as exc:
        await cl.Message(
            content=f"⚠️ Ошибка генерации промпта: {exc}"
        ).send()
        cl.user_session.set("history", history)
        return

    history[-1] = ChatMessage(role="user", content=generated_prompt)

    reply = cl.Message(
        content=(
            f"🛠️ **Сгенерированный промпт:**\n\n"
            f"```text\n{generated_prompt}\n```\n\n"
            f"🤖 **Ответ:**\n\n"
        )
    )
    await reply.send()

    answer_parts: list[str] = []
    succeeded = False
    try:
        async for token in agent.astream(history, model_settings):
            answer_parts.append(token)
            await reply.stream_token(token)
        succeeded = True
    except ContextOverflowError as exc:
        await reply.stream_token(f"\n\n⚠️ {exc}")
    except Exception as exc:
        await reply.stream_token(f"\n\n⚠️ Ошибка LLM: {exc}")
    finally:
        await reply.update()

    # В history — только текст ответа модели, без UI-обёртки сгенерированного промпта
    if succeeded:
        history.append(
            ChatMessage(role="assistant", content="".join(answer_parts))
        )
    cl.user_session.set("history", history)
