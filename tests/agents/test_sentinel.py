"""Tests for the sentinel agent."""

from __future__ import annotations

from typing import cast

import pytest

from biomedos.agents.sentinel import SentinelAgent
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore, VectorDocument
from biomedos.orchestration.state import Task, TaskType


@pytest.mark.asyncio
async def test_sentinel_agent_flags_unsupported_claims_and_uses_literature_support(
    mock_ollama: object,
    sample_kg: object,
) -> None:
    """Sentinel uses KG and local literature to distinguish supported from unsupported claims."""

    vector_store = ChromaVectorStore(persist_dir=":memory:")
    vector_store.add_documents(
        [
            VectorDocument(
                id="lit-1",
                text="EGFR has also been implicated in fibrosis through cross-talk pathways.",
                metadata={"pmid": "4001", "title": "EGFR fibrosis crossover"},
            )
        ]
    )
    mock_ollama.responses = ['{"issues": ["Potential unsupported extrapolation."]}']  # type: ignore[attr-defined]
    agent = SentinelAgent(
        llm_client=cast(OllamaClient, mock_ollama),
        knowledge_graph=sample_kg,
        vector_store=vector_store,
    )
    task = Task(
        id="sentinel-1",
        type=TaskType.SENTINEL,
        description="Verify recent outputs.",
        payload={
            "results": [
                {
                    "summary": (
                        "EGFR is associated with Lung Cancer. "
                        "EGFR is implicated in Fibrosis. "
                        "BRCA1 cures diabetes."
                    ),
                    "output": {"invalid_citations": ["9999"]},
                }
            ]
        },
    )

    result = await agent.run(task)

    issues = result.output["issues"]
    assert result.output["supported_claims"] >= 2
    assert any("BRCA1 cures diabetes" in issue for issue in issues)
    assert any("Invalid citation: 9999" == issue for issue in issues)
    assert any("Potential unsupported extrapolation." == issue for issue in issues)
