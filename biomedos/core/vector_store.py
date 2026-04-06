"""Local ChromaDB-backed vector store with hybrid retrieval."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from biomedos.config import Settings, get_settings, resolve_project_path
from biomedos.core.embedding import EmbeddingManager

try:
    import chromadb
except ImportError:  # pragma: no cover - optional during unit tests
    chromadb = None


class VectorDocument(BaseModel):
    """A document stored inside the vector store."""

    id: str
    text: str
    metadata: dict[str, object] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A document retrieval result."""

    id: str
    text: str
    score: float
    metadata: dict[str, object] = Field(default_factory=dict)


class ChromaVectorStore:
    """Thin wrapper around ChromaDB with a BM25 hybrid search layer."""

    _EMPTY_METADATA_SENTINEL_KEY = "_biomedos_empty_metadata"
    _EMPTY_METADATA_SENTINEL_VALUE = True

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedding_manager: EmbeddingManager | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the vector store."""

        self.settings = settings or get_settings()
        configured_persist_dir = persist_dir or self.settings.CHROMA_PERSIST_DIR
        if configured_persist_dir == ":memory:":
            self.persist_dir = configured_persist_dir
        else:
            self.persist_dir = str(resolve_project_path(Path(configured_persist_dir)))
        self.collection_name = collection_name or self.settings.CHROMA_COLLECTION
        self.embedding_manager = embedding_manager
        self._client: Any | None = None
        self._collection: Any | None = None
        self._documents: dict[str, VectorDocument] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Chroma client if the dependency is available."""

        if chromadb is None:
            return

        if self.persist_dir == ":memory:":
            self._client = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def reset_collection(self) -> None:
        """Delete and recreate the backing collection."""

        self._documents.clear()
        self._embeddings.clear()
        if self._client is None:
            return
        try:
            self._client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._collection = self._client.get_or_create_collection(name=self.collection_name)

    def add_documents(
        self,
        documents: Sequence[VectorDocument],
        *,
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Add or update documents in the vector store."""

        if not documents:
            return

        ids = [document.id for document in documents]
        texts = [document.text for document in documents]
        metadatas = [self._metadata_for_storage(document.metadata) for document in documents]

        vectors: list[list[float]] | None = None
        if embeddings is not None:
            vectors = [[float(value) for value in embedding] for embedding in embeddings]
        elif self.embedding_manager is not None:
            vectors = self.embedding_manager.encode(texts)

        for document in documents:
            self._documents[document.id] = document
        if vectors is not None:
            for document_id, vector in zip(ids, vectors, strict=True):
                self._embeddings[document_id] = vector

        if self._collection is not None:
            kwargs: dict[str, Any] = {
                "ids": ids,
                "documents": texts,
                "metadatas": metadatas,
            }
            if vectors is not None:
                kwargs["embeddings"] = vectors
            self._collection.upsert(**kwargs)

    def search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Run dense vector similarity search."""

        return self._dense_search(query, top_k=top_k)

    def dense_search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Public alias for dense vector search."""

        return self._dense_search(query, top_k=top_k)

    def bm25_search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Public alias for sparse BM25 search."""

        return self._bm25_search(query, top_k=top_k)

    def hybrid_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        alpha: float = 0.6,
    ) -> list[SearchResult]:
        """Combine dense similarity and BM25 scores."""

        dense_results = self._dense_search(query, top_k=max(top_k * 2, top_k))
        bm25_results = self._bm25_search(query, top_k=max(top_k * 2, top_k))

        scores: dict[str, float] = {}
        payloads: dict[str, SearchResult] = {}

        for result in dense_results:
            scores[result.id] = scores.get(result.id, 0.0) + alpha * self._normalize_score(
                result.score
            )
            payloads[result.id] = result

        for result in bm25_results:
            scores[result.id] = scores.get(result.id, 0.0) + (1.0 - alpha) * self._normalize_score(
                result.score
            )
            payloads[result.id] = result

        ranked_ids = sorted(
            scores,
            key=lambda document_id: scores[document_id],
            reverse=True,
        )[:top_k]
        return [
            SearchResult(
                id=payloads[document_id].id,
                text=payloads[document_id].text,
                metadata=payloads[document_id].metadata,
                score=scores[document_id],
            )
            for document_id in ranked_ids
        ]

    def count(self) -> int:
        """Return the number of known documents."""

        return len(self._documents)

    def get_document(self, document_id: str) -> VectorDocument | None:
        """Return a stored document by identifier."""

        return self._documents.get(document_id)

    def _dense_search(self, query: str, *, top_k: int) -> list[SearchResult]:
        """Run dense retrieval against Chroma or the in-memory fallback."""

        if self.embedding_manager is None:
            return self._bm25_search(query, top_k=top_k)

        query_vector = self.embedding_manager.encode_one(query)

        if self._collection is not None and self._embeddings:
            response = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
            )
            ids = response.get("ids", [[]])[0]
            docs = response.get("documents", [[]])[0]
            metadatas = response.get("metadatas", [[]])[0]
            distances = response.get("distances", [[]])[0]
            results: list[SearchResult] = []
            for document_id, document, metadata, distance in zip(
                ids,
                docs,
                metadatas,
                distances,
                strict=True,
            ):
                similarity = 1.0 / (1.0 + float(distance))
                results.append(
                    SearchResult(
                        id=str(document_id),
                        text=str(document),
                        metadata=self._metadata_for_output(metadata),
                        score=similarity,
                    )
                )
            return results

        results = []
        for document_id, vector in self._embeddings.items():
            document = self._documents[document_id]
            results.append(
                SearchResult(
                    id=document.id,
                    text=document.text,
                    metadata=document.metadata,
                    score=self._cosine_similarity(query_vector, vector),
                )
            )
        return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]

    def _bm25_search(self, query: str, *, top_k: int) -> list[SearchResult]:
        """Run a classic BM25 sparse search across all documents."""

        if not self._documents:
            return []

        documents = list(self._documents.values())
        corpus = [self._tokenize(document.text) for document in documents]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(self._tokenize(query))
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                id=documents[int(index)].id,
                text=documents[int(index)].text,
                metadata=documents[int(index)].metadata,
                score=float(scores[int(index)]),
            )
            for index in ranked_indices
        ]

    @staticmethod
    def _sanitize_metadata(metadata: dict[str, object] | object) -> dict[str, object]:
        """Ensure metadata is JSON-serializable and dict-shaped."""

        if not isinstance(metadata, dict):
            return {}
        cleaned: dict[str, object] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                cleaned[str(key)] = value
            else:
                cleaned[str(key)] = str(value)
        return cleaned

    @classmethod
    def _metadata_for_storage(cls, metadata: dict[str, object] | object) -> dict[str, object]:
        """Return Chroma-compatible metadata with a non-empty fallback payload."""

        cleaned = cls._sanitize_metadata(metadata)
        if cleaned:
            return cleaned
        return {cls._EMPTY_METADATA_SENTINEL_KEY: cls._EMPTY_METADATA_SENTINEL_VALUE}

    @classmethod
    def _metadata_for_output(cls, metadata: dict[str, object] | object) -> dict[str, object]:
        """Strip internal placeholder metadata before returning search results."""

        cleaned = cls._sanitize_metadata(metadata)
        if (
            len(cleaned) == 1
            and cleaned.get(cls._EMPTY_METADATA_SENTINEL_KEY) is cls._EMPTY_METADATA_SENTINEL_VALUE
        ):
            return {}
        cleaned.pop(cls._EMPTY_METADATA_SENTINEL_KEY, None)
        return cleaned

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 scoring."""

        return re.findall(r"[A-Za-z0-9_:-]+", text.lower())

    @staticmethod
    def _normalize_score(score: float) -> float:
        """Squash arbitrary scores into a bounded range."""

        if score == 0.0:
            return 0.0
        return score / (1.0 + abs(score))

    @staticmethod
    def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
        """Compute cosine similarity between two dense vectors."""

        if not left or not right:
            return 0.0
        numerator = sum(a * b for a, b in zip(left, right, strict=False))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0
        return numerator / (left_norm * right_norm)
