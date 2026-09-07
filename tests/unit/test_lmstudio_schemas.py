import pytest
from pydantic import ValidationError

from app.schemas.lmstudio import ModelLoadRequest


@pytest.mark.parametrize(
    "model_id",
    [
        "Bionic",
        "openai/gpt-oss-20b",
        "org/model-name.gguf",
        "a_b-c/d.e",
    ],
)
def test_valid_model_id(model_id: str) -> None:
    req = ModelLoadRequest(model_id=model_id)
    assert req.model_id == model_id


@pytest.mark.parametrize(
    "model_id",
    [
        "..",
        "../secret",
        "../../etc/passwd",
        "models/../secret",
        "foo/../../bar",
        "has space",
        "path\\windows",
        "bad|pipe",
        "",
    ],
)
def test_invalid_model_id_rejected(model_id: str) -> None:
    with pytest.raises(ValidationError):
        ModelLoadRequest(model_id=model_id)
