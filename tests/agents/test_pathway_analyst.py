"""Tests for the pathway analyst agent."""

from __future__ import annotations

import pytest

from biomedos.agents.pathway_analyst import PathwayAnalystAgent
from biomedos.orchestration.state import Task, TaskType


@pytest.mark.asyncio
async def test_pathway_analyst_agent_reports_enrichment(sample_kg: object) -> None:
    """Pathway analyst returns enrichment and crosstalk summaries."""

    agent = PathwayAnalystAgent(knowledge_graph=sample_kg)
    task = Task(
        id="pathway-1",
        type=TaskType.PATHWAY_ANALYST,
        description="Analyze EGFR and ALK pathways",
        payload={"genes": ["EGFR", "ALK"]},
    )

    result = await agent.run(task)

    assert "Pathway analysis for genes: EGFR, ALK" in result.summary
    assert result.output["pathway_enrichment"]
