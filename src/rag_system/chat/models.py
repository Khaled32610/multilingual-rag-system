"""Data models for persistent chat sessions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ChatSession:
    session_id: str
    created_at: str
    updated_at: str
    title: str | None = None


@dataclass(frozen=True)
class ChatMessage:
    message_id: int
    session_id: str
    role: Literal["user", "assistant"]
    content: str
    created_at: str
    verified: bool | None = None
