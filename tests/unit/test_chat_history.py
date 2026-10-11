"""Unit tests for local SQLite chat history and service integration."""

from dataclasses import dataclass

import pytest

from rag_system.chat.history import ChatHistoryStore
from rag_system.chat.service import RAGChatService
from rag_system.utils.exceptions import ChatHistoryError


@dataclass
class FakeResponse:
    answer: str
    verified: bool


class FakePipeline:
    def __init__(self):
        self.calls = []

    def run(self, query, conversation_history=()):
        self.calls.append((query, list(conversation_history)))
        return FakeResponse(
            answer="safe answer" if query != "bad" else "I couldn't verify the answer.",
            verified=query != "bad",
        )


def test_sqlite_history_persists_sessions_messages_and_verification(tmp_path):
    path = tmp_path / "chat.sqlite3"
    store = ChatHistoryStore(path)
    session = store.create_session()
    store.add_message(session.session_id, "user", "first question")
    store.add_message(session.session_id, "assistant", "verified reply", verified=True)
    store.add_message(session.session_id, "assistant", "unverified reply", verified=False)

    reopened = ChatHistoryStore(path)
    messages = reopened.get_messages(session.session_id)

    assert [message.role for message in messages] == ["user", "assistant", "assistant"]
    assert messages[1].verified is True
    assert messages[2].verified is False
    assert reopened.list_sessions()[0].title == "first question"


def test_history_limit_returns_latest_messages_in_original_order(tmp_path):
    store = ChatHistoryStore(tmp_path / "chat.sqlite3")
    session = store.create_session()
    for i in range(4):
        store.add_message(session.session_id, "user", f"message {i}")

    assert [m.content for m in store.get_messages(session.session_id, limit=2)] == [
        "message 2",
        "message 3",
    ]


def test_empty_message_unknown_session_and_invalid_limits_fail(tmp_path):
    store = ChatHistoryStore(tmp_path / "chat.sqlite3")
    session = store.create_session()

    with pytest.raises(ChatHistoryError, match="cannot be empty"):
        store.add_message(session.session_id, "user", "  ")
    with pytest.raises(ChatHistoryError, match="not found"):
        store.get_messages("missing")
    with pytest.raises(ChatHistoryError, match="positive"):
        store.get_messages(session.session_id, limit=0)


def test_chat_service_passes_recent_turns_and_persists_final_answer(tmp_path):
    store = ChatHistoryStore(tmp_path / "chat.sqlite3")
    pipeline = FakePipeline()
    service = RAGChatService(pipeline, store, max_history_messages=10)
    session = service.start_session("test")

    first = service.send(session.session_id, "First question")
    second = service.send(session.session_id, "And what about 2024?")

    assert first.verified and second.verified
    assert pipeline.calls[0] == ("First question", [])
    assert pipeline.calls[1][1] == [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "safe answer"},
    ]
    assert [m.role for m in store.get_messages(session.session_id)] == [
        "user",
        "assistant",
        "user",
        "assistant",
    ]


def test_chat_service_stores_safe_fallback_for_unverified_answer(tmp_path):
    store = ChatHistoryStore(tmp_path / "chat.sqlite3")
    service = RAGChatService(FakePipeline(), store)
    session = service.start_session()

    response = service.send(session.session_id, "bad")
    assistant_message = store.get_messages(session.session_id)[-1]

    assert not response.verified
    assert assistant_message.content == "I couldn't verify the answer."
    assert assistant_message.verified is False
