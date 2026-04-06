"""Clinical reasoning agent."""

from __future__ import annotations

from biomedos.agents.base import BaseAgent
from biomedos.clinical.differential_diagnosis import DiagnosisCandidate, DifferentialDiagnosis
from biomedos.clinical.phenotype_matcher import PhenotypeMatcher
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType
from biomedos.orchestration.state import AgentResult, Task, WorkflowState
from biomedos.rag.citation_tracker import CitationTracker


class ClinicianAgent(BaseAgent):
    """Support phenotype reasoning and differential diagnosis."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
        phenotype_matcher: PhenotypeMatcher | None = None,
        diagnosis_engine: DifferentialDiagnosis | None = None,
        citation_tracker: CitationTracker | None = None,
    ) -> None:
        """Initialize the clinician agent."""

        super().__init__(
            name="clinician",
            llm_client=llm_client,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            settings=settings,
        )
        self.phenotype_matcher = phenotype_matcher or PhenotypeMatcher(
            llm_client=llm_client,
            knowledge_graph=self.knowledge_graph,
            settings=self.settings,
        )
        self.diagnosis_engine = diagnosis_engine or DifferentialDiagnosis(
            self.phenotype_matcher,
            knowledge_graph=self.knowledge_graph,
        )
        self.citation_tracker = citation_tracker or CitationTracker()

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the clinical reasoning agent."""

        del state
        findings = task.payload.get("symptoms", task.payload.get("query", task.description))
        mapped_terms = await self.phenotype_matcher.map_to_hpo(str(findings))
        diagnoses = await self.diagnosis_engine.rank(str(findings), top_k=5)
        literature = self._literature_evidence(str(findings), diagnoses)
        summary = self._format_summary(str(findings), diagnoses, literature)
        citations = self._citations_for_diagnoses(diagnoses, literature)
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=summary,
            output={
                "findings": findings,
                "mapped_hpo_terms": [item.model_dump(mode="python") for item in mapped_terms],
                "diagnoses": [item.model_dump(mode="python") for item in diagnoses],
                "literature_evidence": literature,
            },
            citations=citations,
            confidence=self._confidence(diagnoses, citations),
        )

    def _format_summary(
        self,
        findings: str,
        diagnoses: list[DiagnosisCandidate],
        literature: list[dict[str, object]],
    ) -> str:
        """Build a ranked differential-diagnosis summary."""

        lines = [f"Differential diagnosis for: {findings}"]
        if not diagnoses:
            lines.append("No high-confidence diagnoses were identified.")
            return "\n".join(lines)
        for diagnosis in diagnoses:
            matching_papers = [
                str(item["title"])
                for item in literature
                if str(item.get("disease_id", "")) == diagnosis.disease_id
            ]
            lines.append(
                f"- {diagnosis.disease_name}: score={diagnosis.score:.3f}; "
                f"phenotypes={', '.join(diagnosis.matched_phenotypes) or 'none'}; "
                f"kg support={', '.join(diagnosis.kg_support) or 'none'}; "
                f"literature={', '.join(matching_papers[:2]) or 'none'}."
            )
        return "\n".join(lines)

    def _citations_for_diagnoses(
        self,
        diagnoses: list[DiagnosisCandidate],
        literature: list[dict[str, object]],
    ) -> list[str]:
        """Collect local publication citations for the ranked diagnoses."""

        citations: set[str] = set()
        for diagnosis in diagnoses:
            for edge in self.knowledge_graph.get_edges(
                target_id=diagnosis.disease_id,
                edge_type=EdgeType.PUBLICATION_DISEASE,
            ):
                publication = self.knowledge_graph.get_node(edge.source_id)
                publication_pmid = getattr(publication, "pmid", None)
                if publication_pmid is not None:
                    citations.add(str(publication_pmid))
        for item in literature:
            pmid = item.get("pmid")
            if pmid is not None:
                citations.add(str(pmid))
        return sorted(citations)

    def _literature_evidence(
        self,
        findings: str,
        diagnoses: list[DiagnosisCandidate],
    ) -> list[dict[str, object]]:
        """Collect local literature evidence for the ranked diagnoses."""

        evidence: list[dict[str, object]] = []
        seen_pmids: set[str] = set()

        if self.vector_store is not None and self.vector_store.count() > 0:
            for diagnosis in diagnoses[:3]:
                for result in self.vector_store.hybrid_search(
                    f"{diagnosis.disease_name} {findings}",
                    top_k=2,
                ):
                    pmid = result.metadata.get("pmid")
                    pmid_key = str(pmid) if pmid is not None else None
                    if pmid_key is not None and pmid_key in seen_pmids:
                        continue
                    if pmid_key is not None:
                        seen_pmids.add(pmid_key)
                    evidence.append(
                        {
                            "disease_id": diagnosis.disease_id,
                            "pmid": pmid_key,
                            "title": str(result.metadata.get("title", result.id)),
                            "snippet": result.text[:240],
                        }
                    )

        for diagnosis in diagnoses[:3]:
            for edge in self.knowledge_graph.get_edges(
                target_id=diagnosis.disease_id,
                edge_type=EdgeType.PUBLICATION_DISEASE,
            ):
                publication = self.knowledge_graph.get_node(edge.source_id)
                if publication is None:
                    continue
                pmid = getattr(publication, "pmid", None)
                pmid_key = str(pmid) if pmid is not None else None
                if pmid_key is not None and pmid_key in seen_pmids:
                    continue
                if pmid_key is not None:
                    seen_pmids.add(pmid_key)
                evidence.append(
                    {
                        "disease_id": diagnosis.disease_id,
                        "pmid": pmid_key,
                        "title": getattr(publication, "title", publication.name),
                        "snippet": getattr(publication, "abstract", "") or publication.name,
                    }
                )
        return evidence[:8]

    def _confidence(self, diagnoses: list[DiagnosisCandidate], citations: list[str]) -> float:
        """Estimate confidence for the ranked diagnosis list."""

        if not diagnoses:
            return 0.2
        evidence_bonus = min(len(citations), 5) / 25.0
        return max(0.0, min(0.95, diagnoses[0].score + evidence_bonus))
