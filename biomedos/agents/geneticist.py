"""Genetics specialist agent."""

from __future__ import annotations

from biomedos.agents.base import BaseAgent
from biomedos.analysis.centrality import DrugTargetRanker
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType
from biomedos.orchestration.state import AgentResult, Task, WorkflowState


class GeneticistAgent(BaseAgent):
    """Profile genes, variants, and tractability evidence."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
        ranker: DrugTargetRanker | None = None,
    ) -> None:
        """Initialize the geneticist agent."""

        super().__init__(
            name="geneticist",
            llm_client=llm_client,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            settings=settings,
        )
        self.ranker = ranker or DrugTargetRanker(self.knowledge_graph)

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the genetics agent."""

        del state
        gene_query = str(task.payload.get("gene", task.payload.get("query", task.description)))
        gene = self.resolve_node(gene_query, node_type=NodeType.GENE)
        if gene is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.id,
                summary=f"Could not resolve gene: {gene_query}",
                errors=[f"Unknown gene: {gene_query}"],
            )

        diseases = self._named_targets(gene.id, EdgeType.GENE_DISEASE, source=True)
        drugs = self._named_targets(gene.id, EdgeType.DRUG_TARGET, source=False)
        pathways = self._named_targets(gene.id, EdgeType.GENE_PATHWAY, source=True)
        publications = self._recent_publications(gene.id)
        ranking = self.ranker.rank_targets([gene.id], top_k=1)
        target_profile = ranking[0] if ranking else None

        lines = [
            f"Gene report for {gene.name}",
            f"Overview: chromosome={getattr(gene, 'chromosome', None) or 'unknown'}, "
            f"description={getattr(gene, 'description', None) or 'not available'}.",
            f"Diseases: {', '.join(diseases) if diseases else 'none identified'}.",
            f"Drug targets: {', '.join(drugs) if drugs else 'none identified'}.",
            f"Pathways: {', '.join(pathways) if pathways else 'none identified'}.",
        ]
        if target_profile is not None:
            lines.append(
                f"Network position: combined={target_profile.combined_score:.3f}, "
                f"druggability={target_profile.druggability_score:.3f}."
            )
        if publications:
            lines.append(
                "Recent papers: "
                + ", ".join(f"{paper['title']} ({paper['year']})" for paper in publications)
                + "."
            )
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary="\n".join(lines),
            output={
                "gene": gene.model_dump(mode="python"),
                "diseases": diseases,
                "drugs": drugs,
                "pathways": pathways,
                "network_profile": (
                    target_profile.model_dump(mode="python") if target_profile else {}
                ),
                "recent_publications": publications,
            },
            citations=[paper["pmid"] for paper in publications if paper["pmid"]],
            confidence=target_profile.druggability_score if target_profile is not None else 0.5,
        )

    def _named_targets(self, node_id: str, edge_type: EdgeType, *, source: bool) -> list[str]:
        """Resolve connected node names for one relation type."""

        kwargs = {"source_id": node_id} if source else {"target_id": node_id}
        names: list[str] = []
        for edge in self.knowledge_graph.get_edges(edge_type=edge_type, **kwargs):
            neighbor_id = edge.target_id if source else edge.source_id
            node = self.knowledge_graph.get_node(neighbor_id)
            if node is not None:
                names.append(node.name)
        return names

    def _recent_publications(self, gene_id: str) -> list[dict[str, object]]:
        """Return recent publications connected to a gene."""

        publications: list[dict[str, object]] = []
        for edge in self.knowledge_graph.get_edges(
            target_id=gene_id,
            edge_type=EdgeType.PUBLICATION_GENE,
        ):
            publication = self.knowledge_graph.get_node(edge.source_id)
            if publication is None:
                continue
            publications.append(
                {
                    "pmid": getattr(publication, "pmid", None),
                    "title": getattr(publication, "title", publication.name),
                    "year": getattr(publication, "year", None),
                }
            )
        publications.sort(
            key=lambda item: self.coerce_int(item.get("year"), 0),
            reverse=True,
        )
        return publications[:5]
