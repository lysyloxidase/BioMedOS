"""Tests for the geneticist agent."""

from __future__ import annotations

import pytest

from biomedos.agents.geneticist import GeneticistAgent
from biomedos.orchestration.state import Task, TaskType


@pytest.mark.asyncio
async def test_geneticist_agent_builds_gene_report(sample_kg: object) -> None:
    """Geneticist agent summarizes diseases, drugs, and pathways."""

    agent = GeneticistAgent(knowledge_graph=sample_kg)
    task = Task(
        id="gene-1",
        type=TaskType.GENETICIST,
        description="Profile EGFR",
        payload={"gene": "EGFR"},
    )

    result = await agent.run(task)

    assert "Gene report for EGFR" in result.summary
    assert "Lung Cancer" in result.summary
    assert "Gefitinib" in result.summary
