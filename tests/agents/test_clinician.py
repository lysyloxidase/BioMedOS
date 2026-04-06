"""Tests for the clinician agent."""

from __future__ import annotations

from typing import cast

import pytest

from biomedos.agents.clinician import ClinicianAgent
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore, VectorDocument
from biomedos.orchestration.state import Task, TaskType


@pytest.mark.asyncio
async def test_clinician_agent_returns_ranked_differential(
    mock_ollama: object,
    sample_kg: object,
) -> None:
    """Clinician agent combines HPO matching, KG support, and local literature."""

    vector_store = ChromaVectorStore(persist_dir=":memory:")
    vector_store.add_documents(
        [
            VectorDocument(
                id="doc-1",
                text="Lung cancer often presents with chronic cough and dyspnea.",
                metadata={"pmid": "3001", "title": "Lung cancer symptom review"},
            )
        ]
    )
    mock_ollama.responses = ['{"terms": ["chronic cough", "dyspnea"]}']  # type: ignore[attr-defined]
    agent = ClinicianAgent(
        llm_client=cast(OllamaClient, mock_ollama),
        knowledge_graph=sample_kg,
        vector_store=vector_store,
    )
    task = Task(
        id="clin-1",
        type=TaskType.CLINICIAN,
        description="Evaluate persistent cough and breathlessness.",
        payload={"symptoms": "persistent cough and breathlessness"},
    )

    result = await agent.run(task)

    diagnoses = result.output["diagnoses"]
    assert diagnoses[0]["disease_name"] == "Lung Cancer"
    assert result.output["mapped_hpo_terms"]
    assert result.output["literature_evidence"]
    assert {"1001", "3001"} & set(result.citations)
    assert "Differential diagnosis for: persistent cough and breathlessness" in result.summary
