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
        content=(
            f"Готов. Провайдер: `{model_settings.provider}`, "
            f"модель: `{model_settings.model}`."
        )
    ).send()


@cl.on_settings_update
async def on_settings_update(settings: dict[str, object]) -> None:
    current: ModelSettings = cl.user_session.get("model_settings")
    # M2: ручная валидация OK для MVP; при расширении UI → pydantic.BaseModel
    try:
        seed = _parse_optional_int(settings.get("seed"))
    except ValueError:
        await cl.Message(content="seed должен быть целым числом или пустым").send()
        return

    updated = ModelSettings(
        provider=str(settings.get("provider", current.provider)),
        model=str(settings.get("model", current.model)),
        temperature=float(settings.get("temperature", current.temperature)),  # type: ignore[arg-type]
        top_p=float(settings.get("top_p", current.top_p)),  # type: ignore[arg-type]
        max_tokens=int(settings.get("max_tokens", current.max_tokens)),  # type: ignore[arg-type]
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
