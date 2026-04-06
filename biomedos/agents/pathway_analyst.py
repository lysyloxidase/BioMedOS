"""Pathway analysis agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from biomedos.agents.base import BaseAgent
from biomedos.analysis.enrichment import EnrichmentResult, GeneSetEnrichment
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType
from biomedos.orchestration.state import AgentResult, Task, WorkflowState


class PathwayCrosstalk(BaseModel):
    """A pathway crosstalk relationship."""

    source_pathway: str
    target_pathway: str
    relation_score: float
    shared_genes: list[str] = Field(default_factory=list)


class PathwayAnalystAgent(BaseAgent):
    """Analyze pathway enrichment and crosstalk."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
        enrichment: GeneSetEnrichment | None = None,
    ) -> None:
        """Initialize the pathway analyst agent."""

        super().__init__(
            name="pathway_analyst",
            llm_client=llm_client,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            settings=settings,
        )
        self.enrichment = enrichment or GeneSetEnrichment(self.knowledge_graph)

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the pathway analysis agent."""

        del state
        genes = self._resolve_genes(task)
        pathway_hits = self.enrichment.pathway_enrichment(genes)
        disease_hits = self.enrichment.disease_enrichment(genes)
        crosstalk = self._pathway_crosstalk(pathway_hits)

        summary_lines = [
            f"Pathway analysis for genes: {', '.join(genes)}",
            "Top enriched pathways: "
            + (", ".join(hit.term_name for hit in pathway_hits[:5]) if pathway_hits else "none"),
            "Top disease enrichments: "
            + (", ".join(hit.term_name for hit in disease_hits[:5]) if disease_hits else "none"),
        ]
        if crosstalk:
            summary_lines.append(
                "Pathway crosstalk: "
                + ", ".join(
                    f"{item.source_pathway}->{item.target_pathway} ({item.relation_score:.2f})"
                    for item in crosstalk[:5]
                )
            )
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary="\n".join(summary_lines),
            output={
                "genes": genes,
                "pathway_enrichment": [hit.model_dump(mode="python") for hit in pathway_hits],
                "disease_enrichment": [hit.model_dump(mode="python") for hit in disease_hits],
                "crosstalk": [item.model_dump(mode="python") for item in crosstalk],
            },
            confidence=0.75 if pathway_hits else 0.3,
        )

    def _resolve_genes(self, task: Task) -> list[str]:
        """Resolve input genes from the task payload or description."""

        raw_genes = task.payload.get("genes")
        if isinstance(raw_genes, list):
            resolved = [self.resolve_node(str(gene), node_type=NodeType.GENE) for gene in raw_genes]
            return [gene.name for gene in resolved if gene is not None]
        matches = self.query_kg(
            str(task.payload.get("query", task.description)),
            node_type=NodeType.GENE,
            limit=10,
        )
        return [match.name for match in matches]

    def _pathway_crosstalk(
        self,
        pathway_hits: list[EnrichmentResult],
    ) -> list[PathwayCrosstalk]:
        """Characterize interactions among enriched pathways."""

        pathway_ids = [hit.term_id for hit in pathway_hits[:10]]
        results: list[PathwayCrosstalk] = []
        for source_id in pathway_ids:
            source_node = self.knowledge_graph.get_node(source_id)
            if source_node is None:
                continue
            source_genes = {
                edge.source_id
                for edge in self.knowledge_graph.get_edges(
                    target_id=source_id,
                    edge_type=EdgeType.GENE_PATHWAY,
                )
            }
            for edge in self.knowledge_graph.get_edges(
                source_id=source_id,
                edge_type=EdgeType.PATHWAY_PATHWAY,
            ):
                if edge.target_id not in pathway_ids:
                    continue
                target_node = self.knowledge_graph.get_node(edge.target_id)
                if target_node is None:
                    continue
                target_genes = {
                    item.source_id
                    for item in self.knowledge_graph.get_edges(
                        target_id=edge.target_id,
                        edge_type=EdgeType.GENE_PATHWAY,
                    )
                }
                shared_genes: list[str] = []
                for gene_id in source_genes & target_genes:
                    gene_node = self.knowledge_graph.get_node(gene_id)
                    if gene_node is not None:
                        shared_genes.append(gene_node.name)
                results.append(
                    PathwayCrosstalk(
                        source_pathway=source_node.name,
                        target_pathway=target_node.name,
                        relation_score=float(edge.score),
                        shared_genes=sorted(shared_genes),
                    )
                )
        results.sort(key=lambda item: item.relation_score, reverse=True)
        return results
