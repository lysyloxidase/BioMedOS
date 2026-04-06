"""Tests for the graph explorer agent."""

from __future__ import annotations

import pytest

from biomedos.agents.graph_explorer import GraphExplorerAgent
from biomedos.orchestration.state import Task, TaskType


@pytest.mark.asyncio
async def test_graph_explorer_resolves_path_from_natural_language(sample_kg: object) -> None:
    """Graph explorer should infer source and target nodes from a path question."""

    agent = GraphExplorerAgent(knowledge_graph=sample_kg)
    task = Task(
        id="graph-1",
        type=TaskType.GRAPH_EXPLORER,
        description="Show the graph path between EGFR and Lung Cancer.",
        payload={"query": "Show the graph path between EGFR and Lung Cancer."},
    )

    result = await agent.run(task)

    assert "Shortest graph path" in result.summary
    assert result.output["path"] == ["gene:egfr", "disease:lung_cancer"]
