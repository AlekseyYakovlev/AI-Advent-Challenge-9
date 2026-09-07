from app.chainlit.components import (
    STATUS_LABELS,
    actions_blocked,
    build_switcher_content,
    format_global_warning,
    format_status_label,
)
from app.schemas.lmstudio import ModelLoadStatus
from app.services.lmstudio_state import LMStudioState


def test_status_labels_cover_all() -> None:
    for status in ModelLoadStatus:
        assert status in STATUS_LABELS
        assert format_status_label(status)


def test_actions_blocked_for_loading_and_circuit() -> None:
    assert actions_blocked(LMStudioState(status=ModelLoadStatus.LOADING))
    assert actions_blocked(LMStudioState(status=ModelLoadStatus.CIRCUIT_OPEN))
    assert not actions_blocked(LMStudioState(status=ModelLoadStatus.IDLE))
    assert not actions_blocked(LMStudioState(status=ModelLoadStatus.LOADED))


def test_switcher_hides_actions_for_non_lmstudio() -> None:
    content = build_switcher_content(
        provider="ollama",
        session_model="llama",
        state=None,
    )
    assert "lmstudio" not in content.lower() or "скрыто" in content
    assert "llama" in content


def test_global_warning_present_for_lmstudio() -> None:
    state = LMStudioState(
        status=ModelLoadStatus.LOADED,
        current_model="Bionic",
        available_models=["Bionic"],
    )
    content = build_switcher_content(
        provider="lmstudio",
        session_model="Bionic",
        state=state,
    )
    assert format_global_warning() in content
    assert "🟢" in content
