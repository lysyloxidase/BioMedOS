"""Application configuration for BioMedOS."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="BMOS_")

    # LLM
    OLLAMA_HOST: str = "http://localhost:11434"
    MODEL_ROUTER: str = "llama3.2:3b"
    MODEL_REASONER: str = "qwen2.5:14b"
    MODEL_EXTRACTOR: str = "llama3.1:8b"
    MODEL_VERIFIER: str = "phi4:14b"
    MODEL_EMBEDDING: str = "nomic-embed-text"
    FAST_LOCAL_MODE: bool = False

    # Vector Store
    CHROMA_PERSIST_DIR: str = "data/chroma"
    CHROMA_COLLECTION: str = "pubmed_abstracts"

    # Graph
    DEFAULT_GENES: list[str] = Field(
        default_factory=lambda: [
            "LOX",
            "LOXL1",
            "LOXL2",
            "LOXL3",
            "LOXL4",
            "TP53",
            "BRCA1",
            "BRCA2",
            "EGFR",
            "KRAS",
            "BRAF",
            "ALK",
            "MYC",
            "PTEN",
            "RB1",
            "APC",
            "PIK3CA",
            "MTOR",
            "JAK2",
            "FLT3",
        ]
    )
    GRAPH_PERSIST_PATH: str = "data/knowledge_graph.gpickle"

    # ML
    DEVICE: str = "cpu"
    EMBEDDING_DIM: int = 128
    HIDDEN_DIM: int = 256
    GNN_LAYERS: int = 3
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 200
    PATIENCE: int = 20

    # RAG
    RAG_TOP_K: int = 10
    RAG_RERANK_TOP_K: int = 5
    KG_CONTEXT_DEPTH: int = 2
    KG_CONTEXT_MAX_TRIPLES: int = 50

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    RANDOM_SEED: int = 42

    def resolve_path(self, value: str | Path) -> Path:
        """Resolve a project-relative path against the repository root."""

        return resolve_project_path(value)

    def graph_path(self) -> Path:
        """Return the absolute path to the persisted knowledge graph."""

        return self.resolve_path(self.GRAPH_PERSIST_PATH)

    def chroma_path(self) -> Path:
        """Return the absolute path to the Chroma persistence directory."""

        return self.resolve_path(self.CHROMA_PERSIST_DIR)


def project_root() -> Path:
    """Return the BioMedOS repository root."""

    return PROJECT_ROOT


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a path relative to the repository root when needed."""

    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
