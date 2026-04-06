"""Tests for the link predictor agent."""

from __future__ import annotations

import pytest

from biomedos.agents.link_predictor import LinkPredictorAgent
from biomedos.orchestration.state import Task, TaskType


@pytest.mark.asyncio
async def test_link_predictor_agent_returns_ranked_predictions(sample_kg: object) -> None:
    """Link predictor agent returns novel candidate links."""

    agent = LinkPredictorAgent(knowledge_graph=sample_kg)
    task = Task(
        id="predict-1",
        type=TaskType.LINK_PREDICTOR,
        description="Predict novel drug-disease links",
        payload={
            "source_id": "drug:olaparib",
            "target_type": "Disease",
            "edge_type": "drug_treats_disease",
            "epochs": 3,
            "top_k": 3,
        },
    )

    result = await agent.run(task)

    assert "Top novel drug_treats_disease candidates" in result.summary
    assert result.output["predictions"]
