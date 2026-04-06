"""Tests for the review writer agent."""

from __future__ import annotations

import pytest

from biomedos.agents.review_writer import ReviewWriterAgent
from biomedos.config import Settings
from biomedos.core.vector_store import ChromaVectorStore, VectorDocument
from biomedos.orchestration.state import Task, TaskType


@pytest.mark.asyncio
async def test_review_writer_uses_fast_local_mode_for_lightweight_models(
    sample_kg: object,
) -> None:
    """Review writer should take the deterministic fast path for lightweight local models."""

    vector_store = ChromaVectorStore(persist_dir=":memory:")
    vector_store.add_documents(
        [
            VectorDocument(
                id="pmid:3001",
                text=(
                    "EGFR signaling is central to lung cancer biology and supports "
                    "targeted inhibition strategies."
                ),
                metadata={
                    "pmid": "3001",
                    "title": "EGFR signaling in lung cancer",
                    "authors": "Brown T; Green M",
                    "year": 2024,
                    "source": "pubmed",
                },
            ),
            VectorDocument(
                id="pmid:3002",
                text=(
                    "Adaptive resistance and pathway cross-talk complicate durable "
                    "EGFR inhibition responses."
                ),
                metadata={
                    "pmid": "3002",
                    "title": "Resistance to EGFR inhibition",
                    "authors": "Miller R; Chen L",
                    "year": 2025,
                    "source": "pubmed",
                },
            ),
        ]
    )
    settings = Settings(
        MODEL_ROUTER="llama3.2:1b",
        MODEL_REASONER="llama3.2:1b",
        MODEL_EXTRACTOR="llama3.2:1b",
        MODEL_VERIFIER="llama3.2:1b",
    )
    agent = ReviewWriterAgent(
        knowledge_graph=sample_kg,
        vector_store=vector_store,
        settings=settings,
    )
    task = Task(
        id="review-1",
        type=TaskType.REVIEW_WRITER,
        description="EGFR signaling in lung cancer",
        payload={"query": "EGFR signaling in lung cancer"},
    )

    result = await agent.run(task)

    assert result.output["mode"] == "fast_local"
    assert "Narrative Review of EGFR signaling in lung cancer" in result.summary
    assert "Brown et al., 2024" in result.summary
    assert result.citations
