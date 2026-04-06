"""Local sentence embedding management with disk-backed caching."""

from __future__ import annotations

import hashlib
import importlib
import json
import sqlite3
from pathlib import Path
from threading import RLock
from typing import Any

from biomedos.config import resolve_project_path

SentenceTransformerModel: Any | None = None


DEFAULT_EMBEDDING_MODELS: dict[str, str] = {
    "all-MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "PubMedBERT": "pritamdeka/S-PubMedBert-MS-MARCO",
}


class EmbeddingManager:
    """Manage lazy-loaded sentence-transformer models and cached embeddings."""

    def __init__(
        self,
        cache_path: str | Path = "data/embeddings.sqlite",
        default_model: str = "all-MiniLM",
    ) -> None:
        """Initialize the embedding manager.

        Args:
            cache_path: SQLite cache file path.
            default_model: Short alias of the default embedding model.
        """

        self.cache_path = resolve_project_path(cache_path)
        self.default_model = default_model
        self._models: dict[str, Any] = {}
        self._lock = RLock()
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.cache_path, check_same_thread=False)
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """Create the SQLite cache schema."""

        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    vector_json TEXT NOT NULL
                )
                """
            )

    def _resolve_model_name(self, model_name: str | None) -> str:
        """Resolve a short model alias to a fully qualified model identifier."""

        alias = model_name or self.default_model
        return DEFAULT_EMBEDDING_MODELS.get(alias, alias)

    def _get_model(self, model_name: str | None = None) -> Any:
        """Lazy-load and cache a sentence-transformer model."""

        resolved_name = self._resolve_model_name(model_name)
        with self._lock:
            if resolved_name not in self._models:
                sentence_transformer_cls = self._load_sentence_transformer()
                if sentence_transformer_cls is None:
                    msg = "sentence-transformers is not installed."
                    raise RuntimeError(msg)
                self._models[resolved_name] = sentence_transformer_cls(resolved_name)
            return self._models[resolved_name]

    def _cache_key(self, text: str, model_name: str) -> str:
        """Build a deterministic cache key for text and model."""

        return hashlib.sha256(f"{model_name}\u0000{text}".encode()).hexdigest()

    def _get_cached_vector(self, cache_key: str) -> list[float] | None:
        """Fetch an embedding vector from the SQLite cache."""

        cursor = self._connection.execute(
            "SELECT vector_json FROM embeddings WHERE cache_key = ?",
            (cache_key,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        payload = json.loads(str(row[0]))
        if not isinstance(payload, list):
            return None
        return [float(value) for value in payload]

    def _set_cached_vector(self, cache_key: str, model_name: str, vector: list[float]) -> None:
        """Persist an embedding vector into the SQLite cache."""

        with self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO embeddings (cache_key, model_name, vector_json)
                VALUES (?, ?, ?)
                """,
                (cache_key, model_name, json.dumps(vector)),
            )

    def encode(
        self,
        texts: list[str],
        *,
        model_name: str | None = None,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        use_cache: bool = True,
    ) -> list[list[float]]:
        """Encode a batch of texts into vectors."""

        if not texts:
            return []

        resolved_name = self._resolve_model_name(model_name)
        cached_results: dict[int, list[float]] = {}
        missing_indices: list[int] = []

        for index, text in enumerate(texts):
            cache_key = self._cache_key(text, resolved_name)
            vector = self._get_cached_vector(cache_key) if use_cache else None
            if vector is None:
                missing_indices.append(index)
            else:
                cached_results[index] = vector

        if missing_indices:
            model = self._get_model(resolved_name)
            missing_texts = [texts[index] for index in missing_indices]
            raw_vectors = model.encode(
                missing_texts,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors = raw_vectors.tolist() if hasattr(raw_vectors, "tolist") else list(raw_vectors)
            for index, vector in zip(missing_indices, vectors, strict=True):
                normalized = [float(value) for value in vector]
                cached_results[index] = normalized
                if use_cache:
                    cache_key = self._cache_key(texts[index], resolved_name)
                    self._set_cached_vector(cache_key, resolved_name, normalized)

        return [cached_results[index] for index in range(len(texts))]

    def encode_one(self, text: str, *, model_name: str | None = None) -> list[float]:
        """Encode a single text string."""

        return self.encode([text], model_name=model_name)[0]

    def clear_cache(self) -> None:
        """Remove all cached vectors from disk."""

        with self._connection:
            self._connection.execute("DELETE FROM embeddings")

    def close(self) -> None:
        """Close the SQLite cache connection."""

        self._connection.close()

    @staticmethod
    def _load_sentence_transformer() -> Any | None:
        """Load the SentenceTransformer class dynamically."""

        global SentenceTransformerModel

        if SentenceTransformerModel is not None:
            return SentenceTransformerModel

        try:
            module = importlib.import_module("sentence_transformers")
        except ImportError:  # pragma: no cover - optional at test time
            return None

        SentenceTransformerModel = getattr(module, "SentenceTransformer", None)
        return SentenceTransformerModel
