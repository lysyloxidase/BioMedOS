"""LangGraph workflow orchestration."""

from __future__ import annotations

import asyncio
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict

from biomedos.agents.base import BaseAgent
from biomedos.agents.clinician import ClinicianAgent
from biomedos.agents.drug_repurposer import DrugRepurposerAgent
from biomedos.agents.geneticist import GeneticistAgent
from biomedos.agents.graph_explorer import GraphExplorerAgent
from biomedos.agents.hypothesis_generator import HypothesisGeneratorAgent
from biomedos.agents.link_predictor import LinkPredictorAgent
from biomedos.agents.literature import LiteratureAgent
from biomedos.agents.pathway_analyst import PathwayAnalystAgent
from biomedos.agents.pharmacologist import PharmacologistAgent
from biomedos.agents.review_writer import ReviewWriterAgent
from biomedos.agents.router import RouterAgent
from biomedos.agents.sentinel import SentinelAgent
from biomedos.config import Settings, get_settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.orchestration.decomposer import QueryDecomposer
from biomedos.orchestration.state import AgentResult, Task, TaskType, WorkflowState

ImportedStateGraph: Any | None = None
try:  # pragma: no cover - optional dependency at runtime
    _langgraph_graph = importlib.import_module("langgraph.graph")
except ImportError:  # pragma: no cover - exercised by unit tests
    END = "__end__"
    START = "__start__"
else:
    END = str(_langgraph_graph.END)
    START = str(_langgraph_graph.START)
    ImportedStateGraph = _langgraph_graph.StateGraph


class LangGraphState(TypedDict, total=False):
    """Minimal state payload used by the LangGraph workflow."""

    query: str
    tasks: list[dict[str, object]]
    results: dict[str, object]
    final_response: str


@dataclass(slots=True)
class _LocalWorkflowApp:
    """Tiny async wrapper mirroring a compiled workflow app."""

    workflow: BiomedicalWorkflow

    async def ainvoke(self, input_state: WorkflowState | LangGraphState) -> WorkflowState:
        """Invoke the workflow with a query or a prepared state."""

        if isinstance(input_state, WorkflowState):
            return await self.workflow.run(input_state.query)
        query = str(input_state.get("query", ""))
        return await self.workflow.run(query)


@dataclass(slots=True)
class _CompiledWorkflowApp:
    """Typed adapter for compiled LangGraph applications."""

    workflow: BiomedicalWorkflow
    app: Any

    async def ainvoke(self, input_state: WorkflowState | LangGraphState) -> WorkflowState:
        """Invoke the compiled workflow and normalize its output."""

        if isinstance(input_state, WorkflowState):
            payload: LangGraphState = {"query": input_state.query}
        else:
            payload = input_state
        result = await self.app.ainvoke(payload)
        if isinstance(result, WorkflowState):
            return result
        if isinstance(result, dict):
            return self.workflow._state_from_langgraph(result)
        query = str(payload.get("query", ""))
        return await self.workflow.run(query)


class BiomedicalWorkflow:
    """Coordinate multi-agent biomedical workflows."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
        router: RouterAgent | None = None,
        decomposer: QueryDecomposer | None = None,
        agents: Mapping[TaskType, BaseAgent] | None = None,
    ) -> None:
        """Initialize the orchestration workflow."""

        self.settings = settings or get_settings()
        self.llm_client = llm_client
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.vector_store = vector_store
        self.router = router or RouterAgent(
            llm_client=llm_client,
            knowledge_graph=self.knowledge_graph,
            vector_store=vector_store,
            settings=self.settings,
        )
        self.decomposer = decomposer or QueryDecomposer(
            router=self.router,
            knowledge_graph=self.knowledge_graph,
        )
        self._agents = dict(agents or {})

    def build(self) -> object:
        """Build the orchestration graph.

        Returns:
            A workflow graph object.
        """

        if ImportedStateGraph is None:
            return _LocalWorkflowApp(self)

        graph = ImportedStateGraph(LangGraphState)
        graph.add_node("route", self._langgraph_route)
        graph.add_node("execute", self._langgraph_execute)
        graph.add_node("sentinel", self._langgraph_sentinel)
        graph.add_node("aggregate", self._langgraph_aggregate)
        graph.add_edge(START, "route")
        graph.add_edge("route", "execute")
        graph.add_edge("execute", "sentinel")
        graph.add_edge("sentinel", "aggregate")
        graph.add_edge("aggregate", END)
        return _CompiledWorkflowApp(self, graph.compile())

    async def run(self, query: str) -> WorkflowState:
        """Execute the full workflow for a query."""

        state = WorkflowState(query=query)
        router_task = Task(
            id="task-router",
            type=TaskType.ROUTER,
            description=query,
            payload={"query": query},
            priority=1,
        )
        router_result = await self.router.run(router_task, state)
        state.results[router_task.id] = router_result
        state.visited_agents.append(self.router.name)

        tasks = self.decomposer.decompose(query)
        state.tasks = tasks
        specialist_results = await asyncio.gather(*(self._run_task(task, state) for task in tasks))
        for result in specialist_results:
            state.results[result.task_id] = result
            state.visited_agents.append(result.agent_name)

        sentinel_task = Task(
            id="task-sentinel",
            type=TaskType.SENTINEL,
            description=f"Verify results for: {query}",
            payload={
                "results": [result.model_dump(mode="python") for result in specialist_results]
            },
            priority=3,
            dependencies=[task.id for task in tasks],
        )
        sentinel_result = await self._get_agent(TaskType.SENTINEL).run(sentinel_task, state)
        state.results[sentinel_task.id] = sentinel_result
        state.visited_agents.append(sentinel_result.agent_name)

        state.citations = sorted(
            {citation for result in state.results.values() for citation in result.citations}
        )
        state.final_response = self._aggregate(state, tasks)
        return state

    async def _run_task(self, task: Task, state: WorkflowState) -> AgentResult:
        """Run one specialist task."""

        agent = self._get_agent(task.type)
        return await agent.run(task, state)

    def _get_agent(self, task_type: TaskType) -> BaseAgent:
        """Resolve or lazily construct an agent for a task type."""

        if task_type in self._agents:
            return self._agents[task_type]

        registry: dict[TaskType, BaseAgent] = {
            TaskType.LITERATURE: LiteratureAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
            TaskType.GRAPH_EXPLORER: GraphExplorerAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
            TaskType.LINK_PREDICTOR: LinkPredictorAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
            TaskType.DRUG_REPURPOSER: DrugRepurposerAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
            TaskType.GENETICIST: GeneticistAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
            TaskType.PHARMACOLOGIST: PharmacologistAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
            TaskType.CLINICIAN: ClinicianAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
            TaskType.PATHWAY_ANALYST: PathwayAnalystAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
            TaskType.HYPOTHESIS_GENERATOR: HypothesisGeneratorAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
            TaskType.REVIEW_WRITER: ReviewWriterAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
            TaskType.SENTINEL: SentinelAgent(
                llm_client=self.llm_client,
                knowledge_graph=self.knowledge_graph,
                vector_store=self.vector_store,
                settings=self.settings,
            ),
        }
        if task_type not in registry:
            msg = f"No agent registered for task type: {task_type.value}"
            raise KeyError(msg)
        self._agents[task_type] = registry[task_type]
        return self._agents[task_type]

    def _aggregate(self, state: WorkflowState, tasks: list[Task]) -> str:
        """Aggregate specialist outputs into final Markdown."""

        sections = [f"# BioMedOS response\n\nQuery: {state.query}"]
        for task in tasks:
            result = state.results.get(task.id)
            if result is None:
                continue
            title = task.type.value.replace("_", " ").title()
            sections.append(f"## {title}\n{result.summary}")
        sentinel = state.results.get("task-sentinel")
        if sentinel is not None:
            issues = sentinel.output.get("issues", [])
            issue_text = (
                "\n".join(f"- {issue}" for issue in issues)
                if isinstance(issues, list) and issues
                else "- No major issues detected."
            )
            sections.append(f"## Verification\n{sentinel.summary}\n{issue_text}")
        if state.citations:
            sections.append("## Citations\n" + ", ".join(state.citations))
        return "\n\n".join(sections)

    def _state_from_langgraph(self, payload: Mapping[str, object]) -> WorkflowState:
        """Convert a LangGraph payload back into the public workflow model."""

        query = str(payload.get("query", ""))
        workflow_state = WorkflowState(query=query)
        raw_tasks = payload.get("tasks", [])
        raw_results = payload.get("results", {})

        if isinstance(raw_tasks, list):
            workflow_state.tasks = [
                Task.model_validate(task_payload)
                for task_payload in raw_tasks
                if isinstance(task_payload, dict)
            ]
        if isinstance(raw_results, dict):
            workflow_state.results = {
                str(task_id): AgentResult.model_validate(result_payload)
                for task_id, result_payload in raw_results.items()
                if isinstance(result_payload, dict)
            }
        visited_task_ids = [
            "task-router",
            *[task.id for task in workflow_state.tasks],
            "task-sentinel",
        ]
        workflow_state.visited_agents = [
            workflow_state.results[task_id].agent_name
            for task_id in visited_task_ids
            if task_id in workflow_state.results
        ]
        workflow_state.citations = sorted(
            {
                citation
                for result in workflow_state.results.values()
                for citation in result.citations
            }
        )
        final_response = payload.get("final_response")
        workflow_state.final_response = (
            str(final_response)
            if isinstance(final_response, str)
            else self._aggregate(workflow_state, workflow_state.tasks)
        )
        return workflow_state

    async def _langgraph_route(self, state: LangGraphState) -> LangGraphState:
        """LangGraph node: route and decompose tasks."""

        query = str(state.get("query", ""))
        workflow_state = WorkflowState(query=query)
        router_task = Task(
            id="task-router",
            type=TaskType.ROUTER,
            description=query,
            payload={"query": query},
        )
        result = await self.router.run(router_task, workflow_state)
        tasks = self.decomposer.decompose(query)
        return {
            **state,
            "tasks": [task.model_dump(mode="python") for task in tasks],
            "results": {"task-router": result.model_dump(mode="python")},
        }

    async def _langgraph_execute(self, state: LangGraphState) -> LangGraphState:
        """LangGraph node: execute specialist tasks."""

        query = str(state.get("query", ""))
        workflow_state = WorkflowState(query=query)
        raw_tasks = state.get("tasks", [])
        if not isinstance(raw_tasks, list):
            return state
        tasks = [Task.model_validate(task) for task in raw_tasks if isinstance(task, dict)]
        results = await asyncio.gather(*(self._run_task(task, workflow_state) for task in tasks))
        existing_results = state.get("results", {})
        normalized_results = dict(existing_results) if isinstance(existing_results, dict) else {}
        for result in results:
            normalized_results[result.task_id] = result.model_dump(mode="python")
        return {**state, "results": normalized_results}

    async def _langgraph_sentinel(self, state: LangGraphState) -> LangGraphState:
        """LangGraph node: run sentinel verification."""

        query = str(state.get("query", ""))
        raw_results = state.get("results", {})
        result_payloads = list(raw_results.values()) if isinstance(raw_results, dict) else []
        workflow_state = WorkflowState(query=query)
        sentinel_task = Task(
            id="task-sentinel",
            type=TaskType.SENTINEL,
            description=f"Verify results for: {query}",
            payload={"results": result_payloads},
        )
        sentinel_agent = self._get_agent(TaskType.SENTINEL)
        sentinel_result = await sentinel_agent.run(sentinel_task, workflow_state)
        normalized_results = dict(raw_results) if isinstance(raw_results, dict) else {}
        normalized_results[sentinel_task.id] = sentinel_result.model_dump(mode="python")
        return {**state, "results": normalized_results}

    async def _langgraph_aggregate(self, state: LangGraphState) -> LangGraphState:
        """LangGraph node: aggregate final output."""

        query = str(state.get("query", ""))
        raw_tasks = state.get("tasks", [])
        raw_results = state.get("results", {})
        workflow_state = WorkflowState(query=query)
        task_payloads = raw_tasks if isinstance(raw_tasks, list) else []
        tasks = [Task.model_validate(task) for task in task_payloads if isinstance(task, dict)]
        if isinstance(raw_results, dict):
            workflow_state.results = {
                str(task_id): AgentResult.model_validate(payload)
                for task_id, payload in raw_results.items()
                if isinstance(payload, dict)
            }
        workflow_state.citations = sorted(
            {
                citation
                for result in workflow_state.results.values()
                for citation in result.citations
            }
        )
        final_response = self._aggregate(workflow_state, tasks)
        return {**state, "final_response": final_response}
