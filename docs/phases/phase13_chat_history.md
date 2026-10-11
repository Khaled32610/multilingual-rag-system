# Phase 13 — Chat and history

- Chat sessions and user/assistant messages are stored locally in SQLite (`data/chat_history.sqlite3` by default); the database is ignored by Git through the existing `data/` ignore rules.
- Each assistant turn stores the final user-facing answer and whether verification passed. The unverified draft is not added to conversation history.
- The service supplies a bounded recent history to `RAGPipeline`; prompts mark it as context only and explicitly prevent previous assistant answers from acting as evidence.
- The last user question is combined with a new query only when its opening looks like a follow-up (for example, “And what about…?” / “وماذا عن…”). Independent questions retrieve on their own.
- History operations use fresh SQLite connections and parameterized SQL. Missing sessions, empty messages, and invalid limits raise `ChatHistoryError`.

## Files

- `src/rag_system/chat/models.py`
- `src/rag_system/chat/history.py`
- `src/rag_system/chat/service.py`
- `tests/unit/test_chat_history.py`
