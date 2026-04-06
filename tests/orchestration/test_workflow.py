"""Tests for the biomedical workflow."""

from __future__ import annotations

from typing import cast

import pytest

from biomedos.agents.base import BaseAgent
from biomedos.core.llm_client import OllamaClient
from biomedos.orchestration.state import AgentResult, Task, TaskType, WorkflowState
from biomedos.orchestration.workflow import BiomedicalWorkflow


class DummyAgent(BaseAgent):
    """Simple deterministic agent for workflow tests."""

    def __init__(self, name: str, summary: str, citations: list[str] | None = None) -> None:
        """Initialize the dummy agent."""

        super().__init__(name=name)
        self._summary = summary
        self._citations = citations or []

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Return a deterministic result."""

        del state
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=self._summary,
            citations=self._citations,
            confidence=0.8,
        )


@pytest.mark.asyncio
async def test_biomedical_workflow_routes_executes_and_aggregates(
    mock_ollama: object,
    sample_kg: object,
) -> None:
    """Workflow runs router, specialists, sentinel, and aggregation in order."""

    workflow = BiomedicalWorkflow(
        llm_client=cast(OllamaClient, mock_ollama),
        knowledge_graph=sample_kg,
        agents={
            TaskType.LITERATURE: DummyAgent("literature", "Literature summary", ["2001"]),
            TaskType.PHARMACOLOGIST: DummyAgent("pharmacologist", "Pharmacology summary"),
            TaskType.SENTINEL: DummyAgent("sentinel", "Sentinel confidence: 0.90", ["2001"]),
        },
    )

    state = await workflow.run(
        "Find PubMed papers on EGFR and check DDI between Gefitinib and Olaparib."
    )

    assert state.final_response is not None
    assert "## Literature" in state.final_response
    assert "## Pharmacologist" in state.final_response
    assert "## Verification" in state.final_response
    assert "2001" in state.citations


@pytest.mark.asyncio
async def test_workflow_build_returns_async_invokable_app(
    mock_ollama: object,
    sample_kg: object,
) -> None:
    """Workflow build returns an async app interface even without LangGraph installed."""

    workflow = BiomedicalWorkflow(
        llm_client=cast(OllamaClient, mock_ollama),
        knowledge_graph=sample_kg,
        agents={
            TaskType.LITERATURE: DummyAgent("literature", "Literature summary"),
            TaskType.SENTINEL: DummyAgent("sentinel", "Sentinel confidence: 0.90"),
        },
    )

    app = workflow.build()
    result = await app.ainvoke({"query": "Find PubMed papers on EGFR."})  # type: ignore[attr-defined]

    assert isinstance(result, WorkflowState)
    assert result.final_response is not None
