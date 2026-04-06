"""Shared workflow state models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    """Task categories aligned to the 12-agent system."""

    ROUTER = "router"
    LITERATURE = "literature"
    GRAPH_EXPLORER = "graph_explorer"
    LINK_PREDICTOR = "link_predictor"
    DRUG_REPURPOSER = "drug_repurposer"
    GENETICIST = "geneticist"
    PHARMACOLOGIST = "pharmacologist"
    CLINICIAN = "clinician"
    PATHWAY_ANALYST = "pathway_analyst"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    REVIEW_WRITER = "review_writer"
    SENTINEL = "sentinel"


class Task(BaseModel):
    """A unit of work routed to an agent."""

    id: str
    type: TaskType
    description: str
    payload: dict[str, object] = Field(default_factory=dict)
    priority: int = 1
    dependencies: list[str] = Field(default_factory=list)


class AgentResult(BaseModel):
    """Structured output from an agent execution."""

    agent_name: str
    task_id: str
    summary: str
    output: dict[str, object] = Field(default_factory=dict)
    citations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    node_ids: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class WorkflowState(BaseModel):
    """State shared across a multi-agent workflow."""

    query: str
    tasks: list[Task] = Field(default_factory=list)
    results: dict[str, AgentResult] = Field(default_factory=dict)
    messages: list[str] = Field(default_factory=list)
    visited_agents: list[str] = Field(default_factory=list)
    final_response: str | None = None
    citations: list[str] = Field(default_factory=list)
