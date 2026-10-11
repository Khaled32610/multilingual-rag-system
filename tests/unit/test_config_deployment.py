from rag_system.utils.config import Settings


def test_settings_resolve_config_and_data_paths_from_deployment_root(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text(
        "project:\n  name: deployment-test\npaths:\n  data_dir: state\nllm:\n  default_model: from-config\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("RAG_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("RAG_OLLAMA_BASE_URL", "http://ollama:11434")
    monkeypatch.setenv("RAG_LLM_MODEL", "gemma3:4b")

    settings = Settings()

    assert settings.project_root == tmp_path
    assert settings.paths.data_dir == str(tmp_path / "state")
    assert settings.llm.ollama_base_url == "http://ollama:11434"
    assert settings.llm.default_model == "gemma3:4b"


def test_settings_load_env_file_from_explicit_root(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "config.yaml").write_text("project:\n  name: test\n", encoding="utf-8")
    (tmp_path / ".env").write_text("RAG_OLLAMA_BASE_URL=http://local-ollama:11434\n", encoding="utf-8")
    monkeypatch.delenv("RAG_PROJECT_ROOT", raising=False)
    monkeypatch.delenv("RAG_OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("RAG_LLM_MODEL", raising=False)

    settings = Settings(project_root=tmp_path)

    assert settings.llm.ollama_base_url == "http://local-ollama:11434"
