from app.config import Settings
from app.llm.base import LLMProvider
from app.llm.openai_compatible import OpenAICompatibleProvider


def create_provider(
    provider_id: str,
    settings: Settings,
    *,
    timeout: float | None = None,
) -> LLMProvider:
    """timeout=None → settings.request_timeout_sec; для health передавать короткий."""
    mapping = {
        "lmstudio": (settings.lmstudio_base_url, settings.lmstudio_api_key),
        "ollama": (settings.ollama_base_url, settings.ollama_api_key),
        "deepseek": (settings.deepseek_base_url, settings.deepseek_api_key),
    }
    if provider_id not in mapping:
        raise ValueError(f"Неизвестный провайдер: {provider_id}")
    base_url, api_key = mapping[provider_id]
    if provider_id == "deepseek" and not api_key:
        raise ValueError("DEEPSEEK_API_KEY не задан")
    return OpenAICompatibleProvider(
        base_url,
        api_key,
        timeout if timeout is not None else settings.request_timeout_sec,
        allow_top_k=settings.allow_top_k,
    )
