from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    chainlit_host: str = "0.0.0.0"
    chainlit_port: int = 8000

    default_provider: str = "lmstudio"
    default_model: str = "Bionic"

    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_api_key: str = "lm-studio"
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_api_key: str = "ollama"
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_api_key: str = ""

    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_max_tokens: int = 2048
    default_seed: int | None = 42
    request_timeout_sec: float = 120.0
    health_timeout_sec: float = 5.0  # короткий ping в /api/health

    # Лимиты безопасности (бэкенд всегда сильнее UI)
    max_allowed_tokens: int = 4096
    max_history_messages: int = 40
    max_context_chars: int = 24_000
    default_system_prompt: str = ""

    health_verbose: bool = True
    log_json: bool = True
    allow_top_k: bool = False  # M3: False до smoke LM Studio

    @field_validator("default_seed", mode="before")
    @classmethod
    def _empty_seed_to_none(cls, v: object) -> object:
        if v == "":
            return None
        return v

    @field_validator("default_temperature")
    @classmethod
    def _temp(cls, v: float) -> float:
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature должна быть в [0, 2]")
        return v

    @field_validator("default_max_tokens")
    @classmethod
    def _max_tokens_default(cls, v: int) -> int:
        if v < 1:
            raise ValueError("default_max_tokens должен быть ≥ 1")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
