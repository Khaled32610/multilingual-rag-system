"""
Centralized configuration system for rag_system.

Loads settings from (in order of precedence, highest first):
1. Environment variables / .env file (secrets, environment-specific overrides)
2. configs/config.yaml (project defaults, non-secret configuration)

Usage:
    from rag_system.utils.config import get_settings
    settings = get_settings()
    print(settings.chunking.chunk_size)
"""

from __future__ import annotations

import functools
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = three levels up from this file:
# src/rag_system/utils/config.py -> src/rag_system/utils -> src/rag_system -> src -> ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    raw_pdf_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    vector_store_dir: str = "data/vector_store"
    bm25_index_dir: str = "data/bm25_index"
    cache_dir: str = "data/cache"
    logs_dir: str = "logs"

    def resolve(self, root: Path) -> PathsConfig:
        """Return a copy with all paths made absolute relative to root."""
        return PathsConfig(
            data_dir=str(root / self.data_dir),
            raw_pdf_dir=str(root / self.raw_pdf_dir),
            processed_dir=str(root / self.processed_dir),
            vector_store_dir=str(root / self.vector_store_dir),
            bm25_index_dir=str(root / self.bm25_index_dir),
            cache_dir=str(root / self.cache_dir),
            logs_dir=str(root / self.logs_dir),
        )


class PDFConfig(BaseModel):
    max_file_size_mb: int = 100
    allowed_extensions: list[str] = Field(default_factory=lambda: [".pdf"])
    enable_table_extraction: bool = True
    # جديد:
    max_table_text_ratio: float = 3.0   # لو نص الجدول أكبر من 3x نص الصفحة العادي -> يتجاهل
    min_table_density: float = 0.15     # لو أقل من 15% من الخلايا مليانة -> يتجاهل كـ"جدول" رسم بياني


class ChunkingConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 64
    strategy: str = "recursive"


class EmbeddingConfig(BaseModel):
    default_model: str = "placeholder-embedding-model"
    batch_size: int = 32
    dimension: int = 768


class RetrievalConfig(BaseModel):
    top_k: int = 10
    hybrid_alpha: float = 0.5
    enable_query_translation: bool = True
    translation_model: str = "Helsinki-NLP/opus-mt-ar-en"


class RerankingConfig(BaseModel):
    top_n: int = 5
    model_name: str = "BAAI/bge-reranker-v2-m3"


class LLMConfig(BaseModel):
    default_model: str = "gemma3:4b"
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 1024
    benchmark_models: list[str] = ["aya:latest", "qwen2.5:3b-instruct", "gemma3:4b", "phi4-mini"]

class LoggingConfig(BaseModel):
    level: str = "INFO"
    file_name: str = "rag_system.log"
    max_bytes: int = 10_485_760
    backup_count: int = 5


class ProjectConfig(BaseModel):
    name: str = "rag_system"
    version: str = "0.1.0"


class YamlConfig(BaseModel):
    """Schema mirroring configs/config.yaml."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    pdf: PDFConfig = Field(default_factory=PDFConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reranking: RerankingConfig = Field(default_factory=RerankingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


class EnvSettings(BaseSettings):
    """Secrets and environment-specific values, loaded from .env / real env vars."""

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_PATH),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = "development"
    log_level: str = "INFO"

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    cohere_api_key: str | None = None
    huggingface_api_token: str | None = None
    rag_project_root: str | None = None
    rag_ollama_base_url: str | None = None
    rag_llm_model: str | None = None


class Settings:
    """
    Unified settings object combining YAML project config and
    environment-based secrets. This is the single object the rest
    of the codebase should depend on.
    """

    def __init__(
        self,
        config_path: Path | None = None,
        project_root: Path | None = None,
    ) -> None:
        environment_root = os.environ.get("RAG_PROJECT_ROOT")
        provisional_root = Path(environment_root or project_root or PROJECT_ROOT).resolve()
        self.env: EnvSettings = EnvSettings(_env_file=provisional_root / ".env")
        root_value = self.env.rag_project_root or environment_root
        self.project_root = Path(root_value or project_root or PROJECT_ROOT).resolve()
        resolved_config = Path(config_path) if config_path else self.project_root / "configs" / "config.yaml"
        yaml_data = _load_yaml_config(resolved_config)
        self.yaml: YamlConfig = YamlConfig(**yaml_data)
        if self.env.rag_ollama_base_url:
            self.yaml.llm.ollama_base_url = self.env.rag_ollama_base_url
        if self.env.rag_llm_model:
            self.yaml.llm.default_model = self.env.rag_llm_model

        # Resolve all relative paths in the YAML config into absolute paths.
        self.paths: PathsConfig = self.yaml.paths.resolve(self.project_root)

        # Convenience direct access to sub-configs.
        self.project = self.yaml.project
        self.pdf = self.yaml.pdf
        self.chunking = self.yaml.chunking
        self.embedding = self.yaml.embedding
        self.retrieval = self.yaml.retrieval
        self.reranking = self.yaml.reranking
        self.llm = self.yaml.llm
        self.logging = self.yaml.logging

    def ensure_directories(self) -> None:
        """Create all configured directories if they don't already exist."""
        for dir_path in (
            self.paths.data_dir,
            self.paths.raw_pdf_dir,
            self.paths.processed_dir,
            self.paths.vector_store_dir,
            self.paths.bm25_index_dir,
            self.paths.cache_dir,
            self.paths.logs_dir,
        ):
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return (
            f"<Settings project={self.project.name!r} "
            f"env={self.env.environment!r} "
            f"root={self.project_root}>"
        )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached, singleton Settings instance.
    Use this everywhere instead of instantiating Settings() directly.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings
