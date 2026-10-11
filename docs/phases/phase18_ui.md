# Phase 18 — Local UI

Adds a Streamlit interface for persistent Arabic/English document chat, conversation selection, PDF upload, incremental indexing, document listing and deletion. The chat UI displays citation excerpts and reports whether verification passed. Chat, generation, retrieval, and document management share the same `HybridRetriever` instance so newly uploaded documents are searchable immediately.

Install with `pip install -e ".[ui]"` and launch with `streamlit run src/rag_system/ui/app.py`. Ollama, the selected model, and existing vector/BM25 indexes must be available.
