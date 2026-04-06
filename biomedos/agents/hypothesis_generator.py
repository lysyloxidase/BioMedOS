"""Hypothesis generation agent."""

from __future__ import annotations

import networkx as nx
from pydantic import BaseModel, Field

from biomedos.agents.base import BaseAgent
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType
from biomedos.ml.link_prediction import LinkPredictionPipeline
from biomedos.orchestration.state import AgentResult, Task, WorkflowState


class HypothesisCandidate(BaseModel):
    """A ranked biomedical hypothesis."""

    source_id: str
    target_id: str
    edge_type: EdgeType
    score: float
    indirect_path: list[str] = Field(default_factory=list)
    literature_support: list[str] = Field(default_factory=list)
    statement: str


class HypothesisGeneratorAgent(BaseAgent):
    """Generate novel biomedical hypotheses."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
        pipeline: LinkPredictionPipeline | None = None,
    ) -> None:
        """Initialize the hypothesis generator."""

        super().__init__(
            name="hypothesis_generator",
            llm_client=llm_client,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            settings=settings,
        )
        self.pipeline = pipeline

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the hypothesis generator agent."""

        del state
        source_query = str(task.payload.get("source", task.payload.get("query", task.description)))
        source_node = self.resolve_node(source_query)
        if source_node is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.id,
                summary=f"Could not resolve source node: {source_query}",
                errors=[f"Unknown source node: {source_query}"],
            )

        target_type = NodeType(str(task.payload.get("target_type", NodeType.DISEASE.value)))
        edge_type = (
            self.infer_edge_type(source_node.node_type, target_type) or EdgeType.GENE_DISEASE
        )
        pipeline = self.pipeline or LinkPredictionPipeline(
            self.knowledge_graph,
            model_name=str(task.payload.get("model", "graphsage")),
            edge_type=edge_type,
            settings=self.settings,
        )
        epochs = self.coerce_int(task.payload.get("epochs"), min(self.settings.EPOCHS, 25))
        pipeline.train(epochs=epochs)

        candidates = self._find_structural_holes(
            source_node.id,
            target_type=target_type,
            edge_type=edge_type,
            pipeline=pipeline,
            top_k=self.coerce_int(task.payload.get("top_k"), 10),
        )
        summary = self._format_summary(source_node.name, candidates)
        citations = sorted(
            {pmid for candidate in candidates for pmid in candidate.literature_support}
        )
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=summary,
            output={
                "hypotheses": [candidate.model_dump(mode="python") for candidate in candidates]
            },
            citations=citations,
            confidence=candidates[0].score if candidates else 0.2,
        )

    def _find_structural_holes(
        self,
        source_id: str,
        *,
        target_type: NodeType,
        edge_type: EdgeType,
        pipeline: LinkPredictionPipeline,
        top_k: int,
    ) -> list[HypothesisCandidate]:
        """Find indirect but unconnected targets with high link scores."""

        graph = self.knowledge_graph.graph.to_undirected()
        source_node = self.knowledge_graph.get_node(source_id)
        if source_node is None:
            return []
        direct_targets = {
            edge.target_id
            for edge in self.knowledge_graph.get_edges(source_id=source_id, edge_type=edge_type)
        }
        candidates: list[HypothesisCandidate] = []
        for node_id, payload in self.knowledge_graph.graph.nodes(data=True):
            if (
                payload.get("node_type") != target_type
                or node_id == source_id
                or node_id in direct_targets
            ):
                continue
            try:
                path = nx.shortest_path(graph, source_id, node_id)
            except nx.NetworkXNoPath:
                continue
            if len(path) < 3 or len(path) > 4:
                continue
            score = pipeline.score_link(source_id, str(node_id), edge_type=edge_type)
            support = self._literature_support(source_id, str(node_id))
            statement = self._hypothesis_statement(source_id, str(node_id), edge_type, path, score)
            candidates.append(
                HypothesisCandidate(
                    source_id=source_id,
                    target_id=str(node_id),
                    edge_type=edge_type,
                    score=float(0.7 * score + 0.3 * min(len(support), 3) / 3.0),
                    indirect_path=path,
                    literature_support=support,
                    statement=statement,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def _literature_support(self, source_id: str, target_id: str) -> list[str]:
        """Collect publication support bridging source and target."""

        source_support = set()
        target_support = set()
        source_node = self.knowledge_graph.get_node(source_id)
        target_node = self.knowledge_graph.get_node(target_id)
        if source_node is None or target_node is None:
            return []

        if source_node.node_type == NodeType.GENE:
            source_support = {
                edge.source_id
                for edge in self.knowledge_graph.get_edges(
                    target_id=source_id,
                    edge_type=EdgeType.PUBLICATION_GENE,
                )
            }
        if source_node.node_type == NodeType.DISEASE:
            source_support = {
                edge.source_id
                for edge in self.knowledge_graph.get_edges(
                    target_id=source_id,
                    edge_type=EdgeType.PUBLICATION_DISEASE,
                )
            }
        if target_node.node_type == NodeType.GENE:
            target_support = {
                edge.source_id
                for edge in self.knowledge_graph.get_edges(
                    target_id=target_id,
                    edge_type=EdgeType.PUBLICATION_GENE,
                )
            }
        if target_node.node_type == NodeType.DISEASE:
            target_support = {
                edge.source_id
                for edge in self.knowledge_graph.get_edges(
                    target_id=target_id,
                    edge_type=EdgeType.PUBLICATION_DISEASE,
                )
            }
        supporting_publications = source_support & target_support
        pmids: list[str] = []
        for publication_id in supporting_publications:
            publication = self.knowledge_graph.get_node(publication_id)
            publication_pmid = getattr(publication, "pmid", None)
            if publication_pmid is not None:
                pmids.append(str(publication_pmid))
        return sorted(pmids)

    def _hypothesis_statement(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        path: list[str],
        score: float,
    ) -> str:
        """Build a hypothesis statement for one candidate."""

        source_node = self.knowledge_graph.get_node(source_id)
        target_node = self.knowledge_graph.get_node(target_id)
        source_name = source_node.name if source_node is not None else source_id
        target_name = target_node.name if target_node is not None else target_id
        label_parts: list[str] = []
        for node_id in path:
            node = self.knowledge_graph.get_node(node_id)
            label_parts.append(node.name if node is not None else node_id)
        path_labels = " -> ".join(label_parts)
        return (
            f"{source_name} may participate in a novel {edge_type.value} relationship with "
            f"{target_name} (score={score:.3f}) via the indirect chain {path_labels}."
        )

    def _format_summary(
        self,
        source_name: str,
        candidates: list[HypothesisCandidate],
    ) -> str:
        """Format ranked hypotheses."""

        if not candidates:
            return f"No ranked structural-hole hypotheses were identified for {source_name}."
        lines = [f"Hypotheses for {source_name}:"]
        lines.extend(f"- {candidate.statement}" for candidate in candidates[:10])
        return "\n".join(lines)
