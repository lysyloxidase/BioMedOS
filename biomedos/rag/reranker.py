"""Cross-encoder reranking utilities."""

from __future__ import annotations

import importlib
import os
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from biomedos.rag.retriever import RetrievedDocument


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers cross-encoder."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        fast_mode: bool = False,
    ) -> None:
        """Initialize the reranker.

        Args:
            model_name: Cross-encoder model identifier.
            fast_mode: Whether to force lightweight heuristic reranking.
        """

        self.model_name = model_name
        self.fast_mode = fast_mode
        self._model: Any | None = None

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """Rerank retrieved documents by query relevance.

        Args:
            query: User query.
            documents: Candidate documents.
            top_k: Number of reranked documents to keep.

        Returns:
            Reranked documents with updated scores.
        """

        if not documents:
            return []

        scores = self._score_documents(query, documents)
        reranked = [
            document.model_copy(update={"score": float(score)})
            for document, score in zip(documents, scores, strict=True)
        ]
        reranked.sort(key=lambda item: item.score, reverse=True)
        return reranked[:top_k]

    def _score_documents(self, query: str, documents: list[RetrievedDocument]) -> list[float]:
        """Score query-document pairs."""

        model = self._load_model()
        if model is not None:
            pairs = [[query, self._document_text(document)] for document in documents]
            predictions = model.predict(pairs)
            return [float(score) for score in predictions]

        return [self._heuristic_score(query, document) for document in documents]

    def _load_model(self) -> Any | None:
        """Load the cross-encoder model lazily."""

        if self._model is not None:
            return self._model
        if self.fast_mode:
            return None
        if os.getenv("BMOS_FAST_LOCAL_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
            return None

        try:
            cross_encoder_module = importlib.import_module("sentence_transformers.cross_encoder")
        except ImportError:  # pragma: no cover - optional model dependency
            return None

        cross_encoder_cls = getattr(cross_encoder_module, "CrossEncoder", None)
        if cross_encoder_cls is None:
            return None

        try:
            self._model = cross_encoder_cls(self.model_name)
        except Exception:  # pragma: no cover - local model may not exist yet
            self._model = None
        return self._model

    def _heuristic_score(self, query: str, document: RetrievedDocument) -> float:
        """Fallback lexical relevance score."""

        query_tokens = set(self._tokenize(query))
        if not query_tokens:
            return document.score

        text_tokens = set(self._tokenize(self._document_text(document)))
        overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        title_tokens = set(self._tokenize(document.title))
        title_bonus = 0.15 if any(token in title_tokens for token in query_tokens) else 0.0
        source_bonus = 0.05 if document.source == "pubmed" else 0.0
        return float(document.score + overlap + title_bonus + source_bonus)

    @staticmethod
    def _document_text(document: RetrievedDocument) -> str:
        """Concatenate title, content, and KG context for scoring."""

        parts = [document.title, document.content]
        if document.kg_context:
            parts.append(document.kg_context)
        return "\n".join(part for part in parts if part)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for heuristic scoring."""

        return re.findall(r"[A-Za-z0-9_:-]+", text.lower())
