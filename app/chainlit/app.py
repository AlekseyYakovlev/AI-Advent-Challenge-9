import chainlit as cl
from pydantic import ValidationError

from app.chainlit.settings_schema import build_chat_settings
from app.config import Settings, get_settings
from app.llm.base import ChatMessage, LLMProvider, ModelSettings
from app.llm.factory import create_provider
from app.observability.logging import configure_logging
from app.services.agent import AgentService, ContextOverflowError
from app.services.context import truncate_messages

# Важно: при `chainlit run` FastAPI lifespan не выполняется —
# конфигурируем логи здесь (idempotent, дубль с main.py безопасен).
configure_logging(json_logs=get_settings().log_json)

PROMPT_ENGINEERING_SYSTEM = (
    "Ты эксперт по промпт-инжинирингу. Твоя задача — преобразовать запрос "
    "пользователя в детальный, структурированный и эффективный промпт для "
    "языковой модели. Верни ТОЛЬКО текст промпта, без каких-либо "
    "дополнительных комментариев, пояснений или обрамляющих фраз."
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
    )
    cl.user_session.set("model_settings", model_settings)
    cl.user_session.set("history", [])

    await cl.ChatSettings(build_chat_settings(model_settings)).send()
    await cl.Message(
        content=(
            f"Готов. Провайдер: `{model_settings.provider}`, "
            f"модель: `{model_settings.model}`."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, object]) -> None:
    current: ModelSettings = cl.user_session.get("model_settings")
    try:
        updated = ModelSettings(
            provider=str(settings.get("provider", current.provider)),
            model=str(settings.get("model", current.model)),
            temperature=settings.get("temperature", current.temperature),
            top_p=settings.get("top_p", current.top_p),
            max_tokens=settings.get("max_tokens", current.max_tokens),
            seed=settings.get("seed", current.seed),
            top_k=current.top_k,  # M3: виджет скрыт — не читать из UI
            system_prompt=str(
                settings.get("system_prompt", current.system_prompt)
            ),
            stop=settings.get("stop", current.stop),
            step_by_step=bool(
                settings.get("step_by_step", current.step_by_step)
            ),
            pre_generated_prompt=bool(
                settings.get(
                    "pre_generated_prompt", current.pre_generated_prompt
                )
            ),
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
    await cl.Message(
        content=f"Настройки обновлены: {updated.provider}/{updated.model}"
    ).send()


async def generate_improved_prompt(
    history: list[ChatMessage],
    model_settings: ModelSettings,
    provider: LLMProvider,
    app_settings: Settings,
) -> str:
    """Этап 1: non-streaming генерация улучшенного промпта."""
    prompt_gen_settings = model_settings.model_copy(
        update={
            "system_prompt": PROMPT_ENGINEERING_SYSTEM,
            "step_by_step": False,
        }
    )
    # provider.chat не инжектит system_prompt — готовим messages как AgentService
    messages = truncate_messages(
        history,
        max_messages=app_settings.max_history_messages,
        max_chars=app_settings.max_context_chars,
        system_prompt=prompt_gen_settings.system_prompt,
    )
    return (await provider.chat(messages, prompt_gen_settings)).strip()


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
