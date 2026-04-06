"""Differential diagnosis helpers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from biomedos.clinical.phenotype_matcher import PhenotypeMatcher
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType


class DiagnosisCandidate(BaseModel):
    """A ranked differential diagnosis candidate."""

    disease_id: str
    disease_name: str
    score: float
    matched_phenotypes: list[str] = Field(default_factory=list)
    kg_support: list[str] = Field(default_factory=list)
    literature_evidence_count: int = 0


class DifferentialDiagnosis:
    """Rank differential diagnoses from symptoms and phenotypes."""

    def __init__(
        self,
        phenotype_matcher: PhenotypeMatcher,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> None:
        """Initialize the diagnosis engine."""

        self.phenotype_matcher = phenotype_matcher
        self.knowledge_graph = knowledge_graph

    async def rank(
        self,
        findings: str | list[str],
        *,
        top_k: int = 10,
    ) -> list[DiagnosisCandidate]:
        """Rank likely diagnoses."""

        phenotype_matches = await self.phenotype_matcher.match(
            findings,
            top_k=max(top_k * 2, top_k),
        )
        candidates: list[DiagnosisCandidate] = []
        for match in phenotype_matches:
            kg_support = self._kg_support(match.disease_id, match.matched_hpo_terms)
            literature_count = self._literature_support(match.disease_id)
            kg_score = min(len(kg_support), 5) / 5.0
            literature_score = min(literature_count, 5) / 5.0
            final_score = 0.55 * match.score + 0.25 * kg_score + 0.20 * literature_score
            candidates.append(
                DiagnosisCandidate(
                    disease_id=match.disease_id,
                    disease_name=match.disease_name,
                    score=float(final_score),
                    matched_phenotypes=match.matched_hpo_terms,
                    kg_support=kg_support,
                    literature_evidence_count=literature_count,
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[:top_k]

    def _kg_support(self, disease_id: str, matched_terms: list[str]) -> list[str]:
        """Collect KG phenotype support for a disease."""

        if self.knowledge_graph is None:
            return []
        support: list[str] = []
        for edge in self.knowledge_graph.get_edges(
            source_id=disease_id,
            edge_type=EdgeType.DISEASE_PHENOTYPE,
        ):
            phenotype = self.knowledge_graph.get_node(edge.target_id)
            if phenotype is None or phenotype.node_type != NodeType.PHENOTYPE:
                continue
            if any(
                term.lower() in phenotype.name.lower() or phenotype.name.lower() in term.lower()
                for term in matched_terms
            ):
                support.append(phenotype.name)
        return sorted(set(support))

    def _literature_support(self, disease_id: str) -> int:
        """Count local publication evidence for a disease."""

        if self.knowledge_graph is None:
            return 0
        return len(
            self.knowledge_graph.get_edges(
                target_id=disease_id,
                edge_type=EdgeType.PUBLICATION_DISEASE,
            )
        )


DifferentialDiagnosisEngine = DifferentialDiagnosis
