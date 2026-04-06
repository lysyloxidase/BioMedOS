"""Complex task decomposition helpers."""

from __future__ import annotations

import re

from biomedos.agents.router import RouterAgent
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import BioNode, NodeType
from biomedos.orchestration.state import Task, TaskType

QUERY_SPLIT_PATTERN = re.compile(
    r"(?:;\s*|\?\s*|,\s*(?=(?:what|which|are there|is there|find|show|check|analyze|analyse|"
    r"generate|draft|verify|list)\b)|\band\b(?=\s+(?:what|which|are there|is there|find|show|"
    r"check|analyze|analyse|generate|draft|verify|list)\b))",
    flags=re.IGNORECASE,
)


class QueryDecomposer:
    """Split a complex biomedical query into tasks."""

    def __init__(
        self,
        router: RouterAgent | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> None:
        """Initialize the decomposer."""

        self.router = router or RouterAgent(knowledge_graph=knowledge_graph)
        self.knowledge_graph = knowledge_graph

    def decompose(self, query: str) -> list[Task]:
        """Create subtasks from a user request.

        Args:
            query: Original user request.

        Returns:
            Decomposed workflow tasks.
        """

        clauses = self._split_query(query)
        if not clauses:
            clauses = [query.strip()]

        tasks: list[Task] = []
        for index, clause in enumerate(clauses, start=1):
            task_type = self.router._keyword_classify(clause)
            if task_type == TaskType.ROUTER:
                task_type = self._fallback_task_type(clause)
            tasks.append(
                Task(
                    id=f"subtask-{index}",
                    type=task_type,
                    description=clause,
                    payload=self._build_payload(clause, task_type),
                    priority=self._priority_for(task_type),
                )
            )
        return self._deduplicate(tasks)

    def _split_query(self, query: str) -> list[str]:
        """Split a complex query into clause-like subtasks."""

        normalized = re.sub(r"\s+", " ", query).strip()
        if not normalized:
            return []
        return [
            part.strip(" ,.") for part in QUERY_SPLIT_PATTERN.split(normalized) if part.strip(" ,.")
        ]

    def _build_payload(self, clause: str, task_type: TaskType) -> dict[str, object]:
        """Construct a task payload from a clause."""

        payload: dict[str, object] = {"query": clause}
        resolved_nodes = self._extract_nodes(clause, limit=4)
        resolved_genes = self._extract_entities(clause, NodeType.GENE)
        resolved_drugs = self._extract_entities(clause, NodeType.DRUG)
        resolved_diseases = self._extract_entities(clause, NodeType.DISEASE)

        if task_type == TaskType.PHARMACOLOGIST and resolved_drugs:
            payload["drugs"] = resolved_drugs
        if task_type == TaskType.PATHWAY_ANALYST and resolved_genes:
            payload["genes"] = resolved_genes
        if task_type == TaskType.GENETICIST and resolved_genes:
            payload["gene"] = resolved_genes[0]
        if task_type == TaskType.DRUG_REPURPOSER and resolved_diseases:
            payload["disease"] = resolved_diseases[0]
        if task_type == TaskType.CLINICIAN:
            payload["symptoms"] = clause
        if task_type == TaskType.LINK_PREDICTOR and resolved_genes:
            payload["source"] = resolved_genes[0]
            payload["target_type"] = NodeType.DISEASE.value
        if task_type == TaskType.GRAPH_EXPLORER and len(resolved_nodes) >= 2:
            payload["source_id"] = resolved_nodes[0].id
            payload["target_id"] = resolved_nodes[1].id
        return payload

    def _extract_entities(self, clause: str, node_type: NodeType) -> list[str]:
        """Extract entity names from the knowledge graph when possible."""

        if self.knowledge_graph is None:
            return []
        normalized = clause.lower()
        matches = []
        for node_id, payload in self.knowledge_graph.graph.nodes(data=True):
            if payload.get("node_type") != node_type:
                continue
            name = str(payload.get("name", ""))
            if name and name.lower() in normalized:
                node = self.knowledge_graph.get_node(str(node_id))
                if node is not None:
                    matches.append(node.name)
        return matches[:5]

    def _extract_nodes(self, clause: str, *, limit: int = 5) -> list[BioNode]:
        """Extract ordered node mentions from a natural-language clause."""

        if self.knowledge_graph is None:
            return []
        normalized = clause.lower()
        matches: list[tuple[int, int, BioNode]] = []
        for node_id, payload in self.knowledge_graph.graph.nodes(data=True):
            name = str(payload.get("name", "")).strip()
            if not name:
                continue
            pattern = r"\b" + re.escape(name.lower()).replace(r"\ ", r"\s+") + r"\b"
            match = re.search(pattern, normalized)
            if match is None:
                continue
            node = self.knowledge_graph.get_node(str(node_id))
            if node is None:
                continue
            publication_penalty = 1 if node.node_type == NodeType.PUBLICATION else 0
            matches.append((match.start(), publication_penalty, node))

        ordered: list[BioNode] = []
        seen_ids: set[str] = set()
        for _, _, node in sorted(
            matches,
            key=lambda item: (item[0], item[1], -len(item[2].name)),
        ):
            if node.id in seen_ids:
                continue
            seen_ids.add(node.id)
            ordered.append(node)
            if len(ordered) >= limit:
                break
        return ordered

    def _fallback_task_type(self, clause: str) -> TaskType:
        """Choose a specialist when keyword routing is inconclusive."""

        normalized = clause.lower()
        gene_hits = self._extract_entities(clause, NodeType.GENE)
        drug_hits = self._extract_entities(clause, NodeType.DRUG)
        if "trial" in normalized or "target" in normalized or "path" in normalized:
            return TaskType.GRAPH_EXPLORER
        if gene_hits and any(
            token in normalized for token in ("function", "role", "what is", "what does")
        ):
            return TaskType.GENETICIST
        if drug_hits:
            return TaskType.PHARMACOLOGIST
        if "paper" in normalized or "literature" in normalized or "pubmed" in normalized:
            return TaskType.LITERATURE
        return TaskType.LITERATURE

    def _priority_for(self, task_type: TaskType) -> int:
        """Return a simple execution priority."""

        if task_type in {TaskType.CLINICIAN, TaskType.PHARMACOLOGIST, TaskType.SENTINEL}:
            return 3
        if task_type in {TaskType.LITERATURE, TaskType.GRAPH_EXPLORER, TaskType.GENETICIST}:
            return 2
        return 1

    def _deduplicate(self, tasks: list[Task]) -> list[Task]:
        """Deduplicate near-identical task clauses while preserving order."""

        seen: set[tuple[TaskType, str]] = set()
        deduplicated: list[Task] = []
        for task in tasks:
            key = (task.type, task.description.lower())
            if key in seen:
                continue
            seen.add(key)
            deduplicated.append(task)
        return deduplicated
