"""Router agent for task classification and decomposition."""

from __future__ import annotations

import json
import re
from typing import Any

from biomedos.agents.base import BaseAgent
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import NodeType
from biomedos.orchestration.state import AgentResult, Task, TaskType, WorkflowState


class RouterAgent(BaseAgent):
    """Classify user requests into the appropriate BioMedOS task type."""

    KEYWORDS: dict[TaskType, tuple[str, ...]] = {
        TaskType.LITERATURE: ("pubmed", "paper", "publication", "literature", "citation", "review"),
        TaskType.GRAPH_EXPLORER: ("graph", "network", "path", "subgraph", "neighbor", "traverse"),
        TaskType.LINK_PREDICTOR: ("predict link", "novel link", "missing edge", "link prediction"),
        TaskType.DRUG_REPURPOSER: ("repurpose", "reposition", "new indication", "candidate drug"),
        TaskType.GENETICIST: ("gene", "variant", "mutation", "biomarker", "druggability"),
        TaskType.PHARMACOLOGIST: (
            "ddi",
            "interaction",
            "pharmacokinetic",
            "admet",
            "contraindication",
            "metabolism",
        ),
        TaskType.CLINICIAN: ("symptom", "diagnosis", "patient", "phenotype", "hpo", "differential"),
        TaskType.PATHWAY_ANALYST: ("pathway", "enrichment", "signaling", "crosstalk"),
        TaskType.HYPOTHESIS_GENERATOR: ("hypothesis", "novel mechanism", "structural hole"),
        TaskType.REVIEW_WRITER: ("write review", "narrative review", "draft review"),
        TaskType.SENTINEL: ("verify", "validate", "hallucination", "fact-check"),
    }

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the router agent."""

        super().__init__(
            name="router",
            llm_client=llm_client,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            settings=settings,
        )

    async def classify_task(self, query: str) -> TaskType:
        """Classify a query using LLM routing with keyword fallback."""

        if self.should_use_fast_path():
            return self._keyword_classify(query)
        llm_choice = await self._llm_classify(query)
        if llm_choice is not None:
            return llm_choice
        return self._keyword_classify(query)

    async def decompose(self, query: str) -> list[Task]:
        """Decompose a complex query into one or more tasks."""

        task_types = self._detect_multiple_task_types(query)
        if not task_types:
            task_types = [await self.classify_task(query)]

        tasks: list[Task] = []
        for index, task_type in enumerate(task_types, start=1):
            tasks.append(
                Task(
                    id=f"task-{index}",
                    type=task_type,
                    description=query,
                    payload={"query": query},
                    priority=1,
                )
            )
        return tasks

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Classify a workflow task and emit decomposition metadata."""

        task_type = await self.classify_task(task.description)
        subtasks = await self.decompose(task.description)
        visited_agents = state.visited_agents if state is not None else []
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=f"Routed request to `{task_type.value}`.",
            output={
                "task_type": task_type.value,
                "subtasks": [subtask.model_dump(mode="python") for subtask in subtasks],
                "visited_agents": visited_agents,
            },
            confidence=0.85,
        )

    async def _llm_classify(self, query: str) -> TaskType | None:
        """Ask the router model to classify the task."""

        if self.llm_client is None or self.should_use_fast_path():
            return None

        prompt = (
            "Classify the following biomedical request into one task type.\n"
            f"Allowed task types: {', '.join(task_type.value for task_type in TaskType)}.\n"
            'Return JSON: {"task_type": "..."}.\n'
            f"Request: {query}"
        )
        try:
            response = await self.llm_client.generate(
                prompt,
                model=self.settings.MODEL_ROUTER,
                system="Return only valid JSON.",
            )
            payload = self._parse_json(response)
            value = payload.get("task_type")
            if value is None:
                return None
            return TaskType(str(value))
        except (ValueError, json.JSONDecodeError):
            return None

    def _keyword_classify(self, query: str) -> TaskType:
        """Classify a task by weighted keyword matching."""

        normalized = query.lower()
        best_type = TaskType.ROUTER
        best_score = 0

        if self._looks_complex(normalized):
            return TaskType.ROUTER

        for task_type, keywords in self.KEYWORDS.items():
            score = sum(
                len(keyword.split())
                for keyword in keywords
                if self._keyword_present(keyword, normalized)
            )
            if score > best_score:
                best_type = task_type
                best_score = score

        if best_score == 0:
            entity_choice = self._entity_hint_classify(query)
            if entity_choice is not None:
                return entity_choice
        return best_type if best_score > 0 else TaskType.ROUTER

    def _detect_multiple_task_types(self, query: str) -> list[TaskType]:
        """Infer multiple task types from the request."""

        normalized = query.lower()
        detected = [
            task_type
            for task_type, keywords in self.KEYWORDS.items()
            if any(self._keyword_present(keyword, normalized) for keyword in keywords)
        ]
        if self._looks_complex(normalized) and detected:
            ordered: list[TaskType] = [TaskType.ROUTER]
            for task_type in detected:
                if task_type not in ordered:
                    ordered.append(task_type)
            return ordered
        return detected[:1]

    @staticmethod
    def _looks_complex(query: str) -> bool:
        """Heuristic for multi-step or orchestration-heavy queries."""

        triggers = (" then ", " compare ", " plus ", "multi-agent", "workflow")
        return any(trigger in query for trigger in triggers)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse the first JSON object from a model response."""

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, dict):
                return parsed

        msg = "Router response did not contain a valid JSON object."
        raise ValueError(msg)

    @staticmethod
    def _keyword_present(keyword: str, query: str) -> bool:
        """Return whether a keyword or phrase appears as a word-level match."""

        pattern = r"\b" + re.escape(keyword).replace(r"\ ", r"\s+") + r"\b"
        return re.search(pattern, query) is not None

    def _entity_hint_classify(self, query: str) -> TaskType | None:
        """Infer a task type from entity mentions when keyword routing is inconclusive."""

        normalized = query.lower()
        nodes = self.extract_nodes_from_text(query, limit=3)
        if not nodes:
            return None
        first = nodes[0]
        if first.node_type == NodeType.GENE:
            if "drug" in normalized and "target" in normalized:
                return TaskType.GRAPH_EXPLORER
            if any(token in normalized for token in ("function", "role", "what is", "what does")):
                return TaskType.GENETICIST
            return TaskType.GENETICIST
        if first.node_type == NodeType.DRUG:
            return TaskType.PHARMACOLOGIST
        if first.node_type == NodeType.DISEASE:
            return TaskType.LITERATURE
        if first.node_type == NodeType.PATHWAY:
            return TaskType.PATHWAY_ANALYST
        return None
