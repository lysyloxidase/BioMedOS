"""Tests for the literature agent."""

from __future__ import annotations

from typing import cast

import pytest

from biomedos.agents.literature import LiteratureAgent
from biomedos.config import Settings, get_settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.data.pubmed import PubMedArticle, PubMedClient
from biomedos.graph.entity_extractor import BioEntityExtractor, ExtractionResult
from biomedos.orchestration.state import Task, TaskType
from biomedos.rag.pubmed_indexer import PubMedIndexer
from biomedos.rag.reranker import CrossEncoderReranker
from biomedos.rag.retriever import HybridRetriever, RetrievedDocument


class FakePubMedClient:
    """Deterministic PubMed client for agent tests."""

    def __init__(self, articles: list[PubMedArticle]) -> None:
        """Store articles by PMID."""

        self._articles = {article.pmid: article for article in articles}
        self.search_queries: list[str] = []

    async def search(self, query: str, *, max_results: int = 20) -> list[str]:
        """Return the stored PMIDs for any query."""

        del max_results
        self.search_queries.append(query)
        return list(self._articles)

    async def fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        """Return matching stored articles."""

        return [self._articles[pmid] for pmid in pmids if pmid in self._articles]


class FailingPubMedClient(FakePubMedClient):
    """PubMed client stub that fails if remote access is attempted."""

    async def search(self, query: str, *, max_results: int = 20) -> list[str]:
        """Raise if fast-local mode still attempts PubMed search."""

        del query, max_results
        raise AssertionError("Fast local mode should not call PubMed search.")

    async def fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        """Raise if fast-local mode still attempts PubMed fetch."""

        del pmids
        raise AssertionError("Fast local mode should not fetch PubMed abstracts.")


class FakeEntityExtractor:
    """No-op entity extractor used to avoid consuming extra LLM responses."""

    async def extract_triples(self, text: str) -> ExtractionResult:
        """Return an empty extraction result."""

        del text
        return ExtractionResult()


class IdentityReranker(CrossEncoderReranker):
    """Reranker stub that preserves candidate order."""

    def __init__(self) -> None:
        """Avoid loading external model weights."""

    def rerank(
        self,
        query: str,
        documents: list[RetrievedDocument],
        top_k: int = 5,
    ) -> list[RetrievedDocument]:
        """Keep the input order intact."""

        del query
        return documents[:top_k]


@pytest.mark.asyncio
async def test_literature_agent_returns_grounded_citations(
    mock_ollama: object,
    sample_kg: object,
) -> None:
    """Literature agent formats valid author-year citations from PMID-grounded output."""

    articles = [
        PubMedArticle(
            pmid="2001",
            title="EGFR inhibition in lung cancer",
            abstract=(
                "Gefitinib targets EGFR in non-small cell lung cancer and improves "
                "progression-free survival."
            ),
            journal="Clinical Oncology",
            authors=["Brown T", "Green M"],
            year=2022,
        ),
        PubMedArticle(
            pmid="2002",
            title="Mechanisms of resistance to EGFR inhibitors",
            abstract=(
                "Resistance to EGFR inhibitors can emerge through bypass signaling "
                "pathways and adaptive tumor evolution."
            ),
            journal="Cancer Discovery",
            authors=["Miller R", "Chen L"],
            year=2023,
        ),
    ]
    fake_pubmed = FakePubMedClient(articles)
    vector_store = ChromaVectorStore(persist_dir=":memory:")
    indexer = PubMedIndexer(cast(PubMedClient, fake_pubmed), vector_store)
    retriever = HybridRetriever(
        vector_store,
        knowledge_graph=sample_kg,
        reranker=IdentityReranker(),
        settings=get_settings(),
    )

    mock_ollama.responses = [  # type: ignore[attr-defined]
        '{"queries": ["EGFR lung cancer therapy", "gefitinib EGFR NSCLC"]}',
        (
            "Gefitinib-based EGFR inhibition is supported in lung cancer "
            "[PMID:2001]. Resistance mechanisms include bypass signaling pathways "
            "[PMID:2002]."
        ),
    ]
    agent = LiteratureAgent(
        llm_client=cast(OllamaClient, mock_ollama),
        knowledge_graph=sample_kg,
        vector_store=vector_store,
        pubmed_client=cast(PubMedClient, fake_pubmed),
        indexer=indexer,
        retriever=retriever,
        entity_extractor=cast(BioEntityExtractor, FakeEntityExtractor()),
    )
    task = Task(
        id="task-1",
        type=TaskType.LITERATURE,
        description="Summarize EGFR inhibitor evidence in lung cancer.",
        payload={"query": "What evidence supports EGFR inhibitors in lung cancer?"},
    )

    result = await agent.run(task)

    assert "Brown et al., 2022" in result.summary
    assert "Miller et al., 2023" in result.summary
    assert "[PMID:2001]" not in result.summary
    assert result.citations == ["2001", "2002"]
    assert result.output["invalid_citations"] == []
    assert result.output["search_queries"] == [
        "EGFR lung cancer therapy",
        "gefitinib EGFR NSCLC",
    ]
    bibliography = cast(str, result.output["bibliography"])
    assert "## Bibliography" in bibliography
    assert "PMID: 2001" in bibliography
    assert fake_pubmed.search_queries == [
        "EGFR lung cancer therapy",
        "gefitinib EGFR NSCLC",
    ]


@pytest.mark.asyncio
async def test_literature_agent_fast_local_mode_skips_remote_pubmed(sample_kg: object) -> None:
    """Fast local mode should answer from local retrieval without PubMed network calls."""

    vector_store = ChromaVectorStore(persist_dir=":memory:")
    indexer = PubMedIndexer(
        cast(PubMedClient, FailingPubMedClient([])),
        vector_store,
    )
    indexer.index_articles(
        [
            PubMedArticle(
                pmid="3001",
                title="LOX supports extracellular matrix remodeling",
                abstract=(
                    "LOX cross-links extracellular matrix proteins and shapes tissue stiffness."
                ),
                journal="Matrix Reports",
                authors=["Adams P", "Nolan R"],
                year=2024,
            )
        ],
        source="demo",
    )
    agent = LiteratureAgent(
        knowledge_graph=sample_kg,
        vector_store=vector_store,
        settings=Settings(FAST_LOCAL_MODE=True),
        pubmed_client=cast(PubMedClient, FailingPubMedClient([])),
        indexer=indexer,
        entity_extractor=cast(BioEntityExtractor, FakeEntityExtractor()),
    )
    task = Task(
        id="task-fast-lit",
        type=TaskType.LITERATURE,
        description="LOX function",
        payload={"query": "What is the main function of LOX?"},
    )

    result = await agent.run(task)

    assert "LOX" in result.summary
    assert result.output["invalid_citations"] == []
