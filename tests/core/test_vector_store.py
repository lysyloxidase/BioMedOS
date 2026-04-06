"""Tests for the Chroma vector store wrapper."""

from __future__ import annotations

from biomedos.core.vector_store import ChromaVectorStore, VectorDocument


class FakeEmbeddingManager:
    """Deterministic fake embeddings for unit tests."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode texts deterministically."""

        return [self._vectorize(text) for text in texts]

    def encode_one(self, text: str) -> list[float]:
        """Encode a single text."""

        return self._vectorize(text)

    @staticmethod
    def _vectorize(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(len(lowered.split())),
            float(lowered.count("egfr")),
            float(lowered.count("brca1")),
            float(lowered.count("fibrosis")),
        ]


def test_add_and_search_documents() -> None:
    """Dense search returns the closest stored document."""

    store = ChromaVectorStore(persist_dir=":memory:", embedding_manager=FakeEmbeddingManager())
    store.add_documents(
        [
            VectorDocument(id="1", text="EGFR inhibitor response in lung cancer"),
            VectorDocument(id="2", text="BRCA1 synthetic lethality in breast tumors"),
            VectorDocument(id="3", text="Fibrosis progression and matrix remodeling"),
        ]
    )

    results = store.search("EGFR inhibitor", top_k=2)

    assert store.count() == 3
    assert results[0].id == "1"


def test_hybrid_search_combines_sparse_and_dense_signals() -> None:
    """Hybrid search surfaces relevant documents."""

    store = ChromaVectorStore(persist_dir=":memory:", embedding_manager=FakeEmbeddingManager())
    store.add_documents(
        [
            VectorDocument(id="1", text="EGFR inhibitor response in lung cancer"),
            VectorDocument(id="2", text="BRCA1 synthetic lethality in breast tumors"),
            VectorDocument(id="3", text="Fibrosis progression and matrix remodeling"),
        ]
    )

    results = store.hybrid_search("BRCA1 DNA repair", top_k=2)

    assert results
    assert results[0].id == "2"
