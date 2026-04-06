"""Base abstractions for BioMedOS agents."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence

from biomedos.config import Settings, get_settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import BioEdge, BioNode, EdgeType, NodeType
from biomedos.orchestration.state import AgentResult, Task, WorkflowState


class BaseAgent(ABC):
    """Abstract base class for all BioMedOS agents."""

    def __init__(
        self,
        *,
        name: str,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize shared agent dependencies."""

        self.name = name
        self.settings = settings or get_settings()
        self.llm_client = llm_client
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.vector_store = vector_store

    @abstractmethod
    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the agent for the supplied task."""

    def query_kg(
        self,
        query: str,
        *,
        node_type: NodeType | None = None,
        limit: int = 10,
    ) -> list[BioNode]:
        """Search the knowledge graph."""

        return self.knowledge_graph.search_nodes(query, node_type=node_type, limit=limit)

    def update_kg(
        self,
        *,
        nodes: Sequence[BioNode] | None = None,
        edges: Sequence[BioEdge] | None = None,
    ) -> None:
        """Insert nodes and edges into the knowledge graph."""

        for node in nodes or []:
            self.knowledge_graph.merge_node(node)
        for edge in edges or []:
            self.knowledge_graph.merge_edge(edge)

    def resolve_node(self, query: str, *, node_type: NodeType | None = None) -> BioNode | None:
        """Resolve a node by identifier or fuzzy name match."""

        direct = self.knowledge_graph.get_node(query)
        if direct is not None and (node_type is None or direct.node_type == node_type):
            return direct
        matches = self.query_kg(query, node_type=node_type, limit=1)
        return matches[0] if matches else None

    def infer_edge_type(
        self,
        source_type: NodeType,
        target_type: NodeType,
    ) -> EdgeType | None:
        """Infer a likely relation type from source and target node types."""

        relation_map = {
            (NodeType.GENE, NodeType.DISEASE): EdgeType.GENE_DISEASE,
            (NodeType.GENE, NodeType.PATHWAY): EdgeType.GENE_PATHWAY,
            (NodeType.GENE, NodeType.GENE): EdgeType.GENE_GENE,
            (NodeType.DRUG, NodeType.GENE): EdgeType.DRUG_TARGET,
            (NodeType.DRUG, NodeType.DISEASE): EdgeType.DRUG_DISEASE,
            (NodeType.DRUG, NodeType.DRUG): EdgeType.DRUG_DRUG,
            (NodeType.DISEASE, NodeType.PHENOTYPE): EdgeType.DISEASE_PHENOTYPE,
            (NodeType.DISEASE, NodeType.PATHWAY): EdgeType.DISEASE_PATHWAY,
            (NodeType.PATHWAY, NodeType.PATHWAY): EdgeType.PATHWAY_PATHWAY,
        }
        return relation_map.get((source_type, target_type))

    @staticmethod
    def coerce_int(value: object, default: int) -> int:
        """Safely coerce a task payload value to an integer."""

        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default

    @staticmethod
    def is_lightweight_model(model_name: str) -> bool:
        """Return whether a configured local model is lightweight enough for fast paths."""

        normalized = model_name.lower()
        return any(token in normalized for token in ("1b", "2b", "3b", "7b"))

    def should_use_fast_path(self) -> bool:
        """Return whether the current model configuration should prefer fast local fallbacks."""

        return self.settings.FAST_LOCAL_MODE

    def extract_nodes_from_text(
        self,
        text: str,
        *,
        limit: int = 5,
        node_types: Sequence[NodeType] | None = None,
    ) -> list[BioNode]:
        """Resolve ordered entity mentions from free text against the local graph."""

        normalized = text.lower()
        allowed_types = set(node_types or NodeType)
        matches: list[tuple[int, int, BioNode]] = []
        for node_id, payload in self.knowledge_graph.graph.nodes(data=True):
            raw_type = payload.get("node_type")
            if raw_type not in allowed_types:
                continue
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
