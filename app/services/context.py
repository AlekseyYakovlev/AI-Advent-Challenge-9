from app.llm.base import ChatMessage


def truncate_messages(
    history: list[ChatMessage],
    *,
    max_messages: int,
    max_chars: int,
    system_prompt: str,
) -> list[ChatMessage]:
    """
    Сохраняет system + хвост диалога.
    MVP: молча обрезать старые пары; при одной огромной паре — усечь content.
    """
    system = ChatMessage(role="system", content=system_prompt)
    tail = [m for m in history if m.role != "system"]

    if len(tail) > max_messages:
        tail = tail[-max_messages:]
        if tail and tail[0].role == "assistant":
            tail = tail[1:]

    def total_chars(msgs: list[ChatMessage]) -> int:
        return sum(len(m.content) for m in msgs)

    # Пока хвост длиннее 2 — выкидываем старые пары
    while len(tail) > 2 and total_chars([system, *tail]) > max_chars:
        drop = 1
        if len(tail) >= 2 and tail[0].role == "user" and tail[1].role == "assistant":
            drop = 2
        tail = tail[drop:]

    # Граничный случай (M4): 1–2 огромных сообщения всё ещё > max_chars
    budget = max_chars - len(system.content)
    while tail and total_chars(tail) > budget:
        if len(tail) >= 2:
            if tail[0].role == "user" and tail[1].role == "assistant":
                tail = tail[2:]
            else:
                tail = tail[1:]
            continue
        # Осталось одно сообщение — жёстко режем content (хвост промпта важнее начала)
        only = tail[0]
        keep = max(0, budget - 64)
        if keep <= 0:
            tail = []
            break
        if len(only.content) > keep:
            tail = [ChatMessage(role=only.role, content=only.content[-keep:])]
        break

    return [system, *tail]
