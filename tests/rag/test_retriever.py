"""Tests for the hybrid retriever."""

from __future__ import annotations

from typing import cast

import pytest

from biomedos.core.vector_store import ChromaVectorStore, SearchResult
from biomedos.rag.kg_context import KGContext, KGContextExtractor
from biomedos.rag.reranker import CrossEncoderReranker
from biomedos.rag.retriever import HybridRetriever, RetrievedDocument


class StubVectorStore:
    """Deterministic vector-store stub for retrieval tests."""

    def __init__(self) -> None:
        """Seed sparse and dense retrieval outputs."""

        self.bm25_results = [
            SearchResult(
                id="doc-a",
                text="EGFR inhibitor evidence in lung cancer.",
                score=9.0,
                metadata={"pmid": "1001", "title": "Doc A", "source": "pubmed"},
            ),
            SearchResult(
                id="doc-b",
                text="ALK and EGFR crosstalk in tumors.",
                score=8.0,
                metadata={"pmid": "1002", "title": "Doc B", "source": "pubmed"},
            ),
            SearchResult(
                id="doc-c",
                text="Gefitinib response biomarkers.",
                score=7.0,
                metadata={"pmid": "1003", "title": "Doc C", "source": "pubmed"},
            ),
        ]
        self.dense_results = [
            SearchResult(
                id="doc-c",
                text="Gefitinib response biomarkers.",
                score=0.95,
                metadata={"pmid": "1003", "title": "Doc C", "source": "pubmed"},
            ),
            SearchResult(
                id="doc-b",
                text="ALK and EGFR crosstalk in tumors.",
                score=0.90,
                metadata={"pmid": "1002", "title": "Doc B", "source": "pubmed"},
            ),
            SearchResult(
                id="doc-d",
                text="Background textbook note on resistance.",
                score=0.70,
                metadata={"title": "Doc D", "source": "textbook"},
            ),
        ]

    def bm25_search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return the seeded sparse results."""

        del query
        return self.bm25_results[:top_k]

    def dense_search(self, query: str, *, top_k: int = 5) -> list[SearchResult]:
        """Return the seeded dense results."""

        del query
        return self.dense_results[:top_k]


class StubKGContextExtractor(KGContextExtractor):
    """KG context extractor stub."""

    def __init__(self) -> None:
        """Avoid parent initialization for isolated testing."""

    async def extract_context(
        self,
        query: str,
        max_triples: int = 50,
        depth: int = 2,
    ) -> KGContext:
        """Return a small fixed context payload."""

        del query, max_triples, depth
        return KGContext(
            entities_found=["EGFR", "Lung Cancer"],
            triples=["EGFR gene_associated_with_disease Lung Cancer"],
            context_text=(
                "Relevant knowledge graph context:\n"
                "Matched entities: EGFR, Lung Cancer.\n"
                "- EGFR gene_associated_with_disease Lung Cancer"
            ),
            subgraph_size=2,
        )


class IdentityReranker(CrossEncoderReranker):
    """Reranker that preserves the input order from reciprocal rank fusion."""

    def __init__(self) -> None:
        """Avoid loading external cross-encoder weights."""

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """Return the documents unchanged."""

        del query
        return documents[:top_k]


@pytest.mark.asyncio
async def test_retrieve_uses_reciprocal_rank_fusion() -> None:
    """Hybrid retrieval merges BM25 and dense rankings via RRF."""

    retriever = HybridRetriever(
        cast(ChromaVectorStore, StubVectorStore()),
        kg_context_extractor=StubKGContextExtractor(),
        reranker=IdentityReranker(),
    )

    results = await retriever.retrieve("EGFR lung cancer", top_k=4, include_kg=True)

    assert [document.title for document in results] == ["Doc C", "Doc B", "Doc A", "Doc D"]
    assert all(document.kg_context for document in results)
    assert results[0].pmid == "1003"


@pytest.mark.asyncio
async def test_retrieve_returns_kg_context_when_no_documents_found() -> None:
    """Retriever still returns graph context when text retrieval is empty."""

    class EmptyVectorStore(StubVectorStore):
        def __init__(self) -> None:
            self.bm25_results = []
            self.dense_results = []

    retriever = HybridRetriever(
        cast(ChromaVectorStore, EmptyVectorStore()),
        kg_context_extractor=StubKGContextExtractor(),
        reranker=IdentityReranker(),
    )

    results = await retriever.retrieve("EGFR lung cancer", top_k=3, include_kg=True)

    assert len(results) == 1
    assert results[0].source == "kg_context"
    assert "EGFR gene_associated_with_disease Lung Cancer" in results[0].content
