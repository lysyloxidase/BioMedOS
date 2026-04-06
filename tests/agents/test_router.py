"""Tests for the router agent."""

from __future__ import annotations

from typing import cast

import pytest

from biomedos.agents.router import RouterAgent
from biomedos.core.llm_client import OllamaClient
from biomedos.orchestration.state import TaskType


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Search PubMed and then analyze the graph for EGFR.", TaskType.ROUTER),
        ("Find PubMed papers on EGFR resistance.", TaskType.LITERATURE),
        ("Show the graph path between EGFR and lung cancer.", TaskType.GRAPH_EXPLORER),
        ("Predict a novel link between BRCA1 and fibrosis.", TaskType.LINK_PREDICTOR),
        ("Repurpose existing drugs for pulmonary fibrosis.", TaskType.DRUG_REPURPOSER),
        ("Assess gene tractability and mutation patterns for ALK.", TaskType.GENETICIST),
        ("Check DDI and pharmacokinetic issues for gefitinib.", TaskType.PHARMACOLOGIST),
        ("Generate a differential diagnosis for chronic cough.", TaskType.CLINICIAN),
        ("Perform pathway enrichment for EGFR and ALK.", TaskType.PATHWAY_ANALYST),
        ("Generate a novel hypothesis around LOXL2.", TaskType.HYPOTHESIS_GENERATOR),
        ("Draft narrative review on BRCA1 synthetic lethality.", TaskType.REVIEW_WRITER),
        ("Verify this answer and fact-check possible hallucinations.", TaskType.SENTINEL),
    ],
)
async def test_router_keyword_classification(query: str, expected: TaskType) -> None:
    """Router fallback classification covers all task types."""

    router = RouterAgent()
    task_type = await router.classify_task(query)
    assert task_type == expected


@pytest.mark.asyncio
async def test_router_uses_gene_entity_hints_for_function_questions(sample_kg: object) -> None:
    """Generic function questions about known genes should route to the geneticist."""

    router = RouterAgent(knowledge_graph=sample_kg)

    task_type = await router.classify_task("What is main function of TP53?")

    assert task_type == TaskType.GENETICIST


@pytest.mark.asyncio
async def test_router_prefers_llm_output(mock_ollama: object) -> None:
    """LLM classification overrides keyword fallback when valid JSON is returned."""

    mock_ollama.responses = ['{"task_type": "sentinel"}']  # type: ignore[attr-defined]
    router = RouterAgent(llm_client=cast(OllamaClient, mock_ollama))

    task_type = await router.classify_task("Find PubMed papers on EGFR.")

    assert task_type == TaskType.SENTINEL
