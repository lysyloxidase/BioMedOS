"""Drug repurposing agent."""

from __future__ import annotations

from pydantic import BaseModel, Field

from biomedos.agents.base import BaseAgent
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.queries import GraphQueryEngine
from biomedos.graph.schema import EdgeType, NodeType
from biomedos.ml.link_prediction import LinkPredictionPipeline
from biomedos.orchestration.state import AgentResult, Task, WorkflowState


class RepurposingReport(BaseModel):
    """A ranked drug repurposing candidate."""

    drug_id: str
    drug_name: str
    score: float
    shared_genes: list[str] = Field(default_factory=list)
    gnn_score: float = 0.0
    evidence_count: int = 0
    confidence_level: str = "low"
    rationale: list[str] = Field(default_factory=list)


class DrugRepurposerAgent(BaseAgent):
    """Find candidate drugs for new disease indications."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
        pipeline: LinkPredictionPipeline | None = None,
    ) -> None:
        """Initialize the drug repurposer agent."""

        super().__init__(
            name="drug_repurposer",
            llm_client=llm_client,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            settings=settings,
        )
        self.pipeline = pipeline

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the drug repurposing agent."""

        del state
        disease_query = str(
            task.payload.get("disease", task.payload.get("query", task.description))
        )
        disease = self.resolve_node(disease_query, node_type=NodeType.DISEASE)
        if disease is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.id,
                summary=f"Could not resolve disease: {disease_query}",
                errors=[f"Unknown disease: {disease_query}"],
            )

        query_engine = GraphQueryEngine(self.knowledge_graph)
        candidates = query_engine.find_repurposing_candidates(disease.id)
        pipeline = self.pipeline or LinkPredictionPipeline(
            self.knowledge_graph,
            model_name=str(task.payload.get("model", "graphsage")),
            edge_type=EdgeType.DRUG_DISEASE,
            settings=self.settings,
        )
        epochs = self.coerce_int(task.payload.get("epochs"), min(self.settings.EPOCHS, 25))
        pipeline.train(epochs=epochs)

        reports: list[RepurposingReport] = []
        citation_pmids: set[str] = set()
        max_base_score = max((candidate.score for candidate in candidates), default=1.0)
        for candidate in candidates:
            gnn_score = pipeline.score_link(
                candidate.drug_id,
                disease.id,
                edge_type=EdgeType.DRUG_DISEASE,
            )
            evidence_pmids = self._supporting_publications(disease.id, candidate.target_genes)
            citation_pmids.update(evidence_pmids)
            normalized_base = candidate.score / max(max_base_score, 1.0)
            gene_overlap_score = min(len(set(candidate.target_genes)), 5) / 5.0
            evidence_score = min(len(evidence_pmids), 5) / 5.0
            final_score = (
                0.35 * normalized_base
                + 0.30 * gnn_score
                + 0.20 * evidence_score
                + 0.15 * gene_overlap_score
            )
            reports.append(
                RepurposingReport(
                    drug_id=candidate.drug_id,
                    drug_name=candidate.drug_name,
                    score=float(final_score),
                    shared_genes=sorted(set(candidate.target_genes)),
                    gnn_score=float(gnn_score),
                    evidence_count=len(evidence_pmids),
                    confidence_level=self._confidence_level(final_score),
                    rationale=candidate.rationale,
                )
            )

        reports.sort(key=lambda item: item.score, reverse=True)
        summary = self._format_summary(disease.name, reports[:10])
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=summary,
            output={"candidates": [report.model_dump(mode="python") for report in reports[:10]]},
            citations=sorted(citation_pmids),
            confidence=reports[0].score if reports else 0.1,
        )

    def _supporting_publications(self, disease_id: str, target_genes: list[str]) -> set[str]:
        """Collect publications co-supporting disease and target genes."""

        disease_publications = {
            edge.source_id
            for edge in self.knowledge_graph.get_edges(
                target_id=disease_id,
                edge_type=EdgeType.PUBLICATION_DISEASE,
            )
        }
        supporting: set[str] = set()
        for gene_name in target_genes:
            gene = self.resolve_node(gene_name, node_type=NodeType.GENE)
            if gene is None:
                continue
            gene_publications = {
                edge.source_id
                for edge in self.knowledge_graph.get_edges(
                    target_id=gene.id,
                    edge_type=EdgeType.PUBLICATION_GENE,
                )
            }
            supporting.update(disease_publications & gene_publications)
        pmids: set[str] = set()
        for publication_id in supporting:
            publication = self.knowledge_graph.get_node(publication_id)
            publication_pmid = getattr(publication, "pmid", None)
            if publication_pmid is not None:
                pmids.add(str(publication_pmid))
        return pmids

    def _confidence_level(self, score: float) -> str:
        """Map a repurposing score onto a qualitative confidence level."""

        if score >= 0.75:
            return "high"
        if score >= 0.5:
            return "medium"
        return "low"

    def _format_summary(self, disease_name: str, reports: list[RepurposingReport]) -> str:
        """Format ranked repurposing results."""

        if not reports:
            return f"No repurposing candidates were identified for {disease_name}."
        lines = [f"Repurposing candidates for {disease_name}:"]
        for report in reports:
            shared_genes = ", ".join(report.shared_genes)
            lines.append(
                f"- {report.drug_name}: score={report.score:.3f}, "
                f"confidence={report.confidence_level}, shared genes={shared_genes}"
            )
        return "\n".join(lines)
