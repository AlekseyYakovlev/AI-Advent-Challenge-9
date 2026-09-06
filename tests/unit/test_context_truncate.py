from app.llm.base import ChatMessage
from app.services.context import truncate_messages


def test_truncate_keeps_system_and_tail() -> None:
    history = [
        ChatMessage(role="user", content="u1"),
        ChatMessage(role="assistant", content="a1"),
        ChatMessage(role="user", content="u2"),
        ChatMessage(role="assistant", content="a2"),
        ChatMessage(role="user", content="u3"),
        ChatMessage(role="assistant", content="a3"),
    ]
    result = truncate_messages(
        history,
        max_messages=2,
        max_chars=10_000,
        system_prompt="SYS",
    )
    assert result[0].role == "system"
    assert result[0].content == "SYS"
    # хвост из последних max_messages (с поправкой на одинокий assistant)
    assert [m.content for m in result[1:]] == ["u3", "a3"]


def test_truncate_huge_single_pair_trims_content() -> None:
    huge = "x" * 5_000
    history = [
        ChatMessage(role="user", content=huge),
        ChatMessage(role="assistant", content="short"),
    ]
    result = truncate_messages(
        history,
        max_messages=40,
        max_chars=200,
        system_prompt="SYS",
    )
    assert result[0].role == "system"
    assert result[0].content == "SYS"
    total = sum(len(m.content) for m in result)
    assert total <= 200
    # либо выкинули пару, либо обрезали content последнего user
    assert all(m.role != "system" or m.content == "SYS" for m in result)


def test_system_always_present_even_on_empty_history() -> None:
    result = truncate_messages(
        [],
        max_messages=40,
        max_chars=100,
        system_prompt="SYS",
    )
    assert result == [ChatMessage(role="system", content="SYS")]


def test_truncate_does_not_drop_system_when_over_budget() -> None:
    history = [ChatMessage(role="user", content="y" * 10_000)]
    result = truncate_messages(
        history,
        max_messages=40,
        max_chars=100,
        system_prompt="KEEP-ME",
    )
    assert result[0] == ChatMessage(role="system", content="KEEP-ME")
    assert len(result) >= 1
