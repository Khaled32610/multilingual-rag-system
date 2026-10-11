"""Chat service connecting persistent history to the RAG pipeline."""

from __future__ import annotations

from rag_system.chat.history import ChatHistoryStore
from rag_system.pipeline.rag_pipeline import RAGPipeline, RAGResponse


class RAGChatService:
    """Manage sessions and pass a bounded dialogue history to the RAG pipeline."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        history: ChatHistoryStore,
        max_history_messages: int = 10,
    ) -> None:
        if max_history_messages < 1:
            raise ValueError("max_history_messages must be positive")
        self.pipeline = pipeline
        self.history = history
        self.max_history_messages = max_history_messages

    def start_session(self, title: str | None = None):
        return self.history.create_session(title=title)

    def send(self, session_id: str, query: str) -> RAGResponse:
        previous = self.history.get_messages(session_id, limit=self.max_history_messages)
        conversation_history = [
            {"role": message.role, "content": message.content} for message in previous
        ]
        self.history.add_message(session_id, "user", query)
        response = self.pipeline.run(query, conversation_history=conversation_history)
        self.history.add_message(
            session_id,
            "assistant",
            response.answer,
            verified=response.verified,
        )
        return response
