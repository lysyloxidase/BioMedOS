"""Tests for the drug repurposer agent."""

from __future__ import annotations

import pytest

from biomedos.agents.drug_repurposer import DrugRepurposerAgent
from biomedos.orchestration.state import Task, TaskType


@pytest.mark.asyncio
async def test_drug_repurposer_ranks_candidates(sample_kg: object) -> None:
    """Repurposer returns ranked candidates for a disease."""

    agent = DrugRepurposerAgent(knowledge_graph=sample_kg)
    task = Task(
        id="repurpose-1",
        type=TaskType.DRUG_REPURPOSER,
        description="Repurpose drugs for Lung Cancer",
        payload={"disease": "Lung Cancer", "model": "graphsage", "epochs": 3},
    )

    result = await agent.run(task)

    assert "Repurposing candidates for Lung Cancer" in result.summary
    candidates = result.output["candidates"]
    assert candidates
    assert candidates[0]["drug_name"] == "Olaparib"
