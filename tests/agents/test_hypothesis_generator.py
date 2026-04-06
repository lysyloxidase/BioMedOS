"""Tests for the hypothesis generator agent."""

from __future__ import annotations

import pytest

from biomedos.agents.hypothesis_generator import HypothesisGeneratorAgent
from biomedos.orchestration.state import Task, TaskType


@pytest.mark.asyncio
async def test_hypothesis_generator_finds_structural_holes(sample_kg: object) -> None:
    """Hypothesis generator returns ranked indirect associations."""

    agent = HypothesisGeneratorAgent(knowledge_graph=sample_kg)
    task = Task(
        id="hypothesis-1",
        type=TaskType.HYPOTHESIS_GENERATOR,
        description="Generate hypotheses around BRCA1",
        payload={"source": "BRCA1", "target_type": "Disease", "epochs": 3},
    )

    result = await agent.run(task)

    assert "Hypotheses for BRCA1" in result.summary
    assert result.output["hypotheses"]
