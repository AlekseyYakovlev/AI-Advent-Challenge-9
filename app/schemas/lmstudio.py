from enum import Enum

from pydantic import BaseModel, Field, field_validator


class ModelLoadStatus(str, Enum):
    IDLE = "idle"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"
    UNREACHABLE = "unreachable"
    CIRCUIT_OPEN = "circuit_open"


class ModelLoadRequest(BaseModel):
    # pattern без look-ahead: pydantic-core/Rust regex его не поддерживает.
    # Запрет ".." — в field_validator (+ проверка частей split("/")).
    model_id: str = Field(
        ...,
        min_length=1,
        max_length=256,
        pattern=r"^[a-zA-Z0-9_\-/\.]+$",
    )
    gpu_offload: str = "max"
    context_length: int | None = None
    force_unload_previous: bool = True

    @field_validator("model_id")
    @classmethod
    def _reject_path_traversal(cls, value: str) -> str:
        if ".." in value or ".." in value.split("/"):
            raise ValueError("model_id содержит path traversal ('..')")
        return value


class ModelLoadResult(BaseModel):
    model_config = {"frozen": True}

    status: ModelLoadStatus
    message: str
    model_id: str | None = None
    retry_after_seconds: int | None = None
