"""SQLite-backed local persistence for chat sessions and messages."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal

from rag_system.chat.models import ChatMessage, ChatSession
from rag_system.utils.config import get_settings
from rag_system.utils.exceptions import ChatHistoryError

ChatRole = Literal["user", "assistant"]


class ChatHistoryStore:
    """Persist chat history locally with a fresh SQLite connection per operation."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        if database_path is None:
            database_path = Path(get_settings().paths.data_dir) / "chat_history.sqlite3"
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        except sqlite3.Error as exc:
            raise ChatHistoryError("Chat history database operation failed", details={"error": str(exc)}) from exc
        finally:
            connection.close()

    def _initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        title TEXT
                    )"""
                )
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS chat_messages (
                        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
                        role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        verified INTEGER
                    )"""
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, message_id)"
                )
        except OSError as exc:
            raise ChatHistoryError("Could not initialize chat history database", details={"error": str(exc)}) from exc

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

    def create_session(self, title: str | None = None) -> ChatSession:
        session_id = uuid.uuid4().hex
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_sessions(session_id, created_at, updated_at, title) VALUES (?, ?, ?, ?)",
                (session_id, now, now, title.strip() if title and title.strip() else None),
            )
        return ChatSession(session_id, now, now, title.strip() if title and title.strip() else None)

    def add_message(
        self,
        session_id: str,
        role: ChatRole,
        content: str,
        verified: bool | None = None,
    ) -> ChatMessage:
        if role not in {"user", "assistant"}:
            raise ChatHistoryError(f"Unsupported chat role: {role!r}")
        if not isinstance(content, str) or not content.strip():
            raise ChatHistoryError("Chat message content cannot be empty")
        now = self._now()
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                raise ChatHistoryError(f"Chat session not found: {session_id}")
            cursor = connection.execute(
                "INSERT INTO chat_messages(session_id, role, content, created_at, verified) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content.strip(), now, None if verified is None else int(verified)),
            )
            connection.execute(
                """UPDATE chat_sessions SET updated_at = ?,
                   title = COALESCE(title, ?) WHERE session_id = ?""",
                (now, content.strip()[:100] if role == "user" else None, session_id),
            )
            message_id = cursor.lastrowid
        return ChatMessage(message_id, session_id, role, content.strip(), now, verified)

    def get_messages(self, session_id: str, limit: int | None = None) -> list[ChatMessage]:
        if limit is not None and limit < 1:
            raise ChatHistoryError("Message limit must be positive")
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM chat_sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if exists is None:
                raise ChatHistoryError(f"Chat session not found: {session_id}")
            if limit is None:
                rows = connection.execute(
                    "SELECT * FROM chat_messages WHERE session_id = ? ORDER BY message_id",
                    (session_id,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT * FROM (
                           SELECT * FROM chat_messages WHERE session_id = ?
                           ORDER BY message_id DESC LIMIT ?
                       ) ORDER BY message_id""",
                    (session_id, limit),
                ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def list_sessions(self, limit: int | None = None) -> list[ChatSession]:
        if limit is not None and limit < 1:
            raise ChatHistoryError("Session limit must be positive")
        with self._connect() as connection:
            if limit is None:
                rows = connection.execute(
                    "SELECT * FROM chat_sessions ORDER BY updated_at DESC"
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM chat_sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [ChatSession(row["session_id"], row["created_at"], row["updated_at"], row["title"]) for row in rows]

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> ChatMessage:
        verified = row["verified"]
        return ChatMessage(
            message_id=row["message_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            verified=None if verified is None else bool(verified),
        )
