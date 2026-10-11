# Phase 19 — Tests and optimization

Added formal unit coverage for PDF validation (missing, empty, corrupt, extension and size), SHA-256 hashing and duplicate grouping, Arabic/English-safe text cleaning, and ingestion orchestration success/failure paths. The new evaluation, experiment, PDF export input, and UI modules also have regression coverage.

The Streamlit service factory is cached for the app session, and the chat and upload flows share one hybrid retriever/index so the app does not load duplicate embedding/retrieval stores. Newly added evaluation, experiment, report, UI, and document-manager modules pass Ruff. The repository-wide Ruff run still reports findings in older scripts and modules outside this phase; those are kept out of this focused maintenance commit.

Run validation with `python -m pytest -q` and targeted lint with `ruff check` on the new phase modules.
