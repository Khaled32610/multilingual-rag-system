# Phase 20 — Docker and Windows/Linux packaging

Uses CPU-only PyTorch wheels to avoid pulling CUDA runtime packages for local embedding/reranking, while Ollama owns local LLM inference. Provides a Python 3.12 Docker image, Docker Compose services for the Streamlit UI and local Ollama, persistent data/model volumes, and native PowerShell/Bash install and start scripts. The compose port binds to `127.0.0.1`; Ollama is reachable only inside the compose network. Pull the configured model once with `docker compose exec ollama ollama pull gemma3:4b`.

`RAG_PROJECT_ROOT` lets installed deployments locate `configs/config.yaml` and data. `RAG_OLLAMA_BASE_URL` and `RAG_LLM_MODEL` override the configured local LLM endpoint/model, for example `http://ollama:11434` inside Compose. PDF reporting and the UI remain optional dependencies, also grouped in the `all` extra.

Native setup supports Python 3.11+. Run the platform's install script once, then its start script. Model and embedding weights are not bundled; they are downloaded or pulled on first use and should be persisted in the configured caches.
