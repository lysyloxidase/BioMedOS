"""Phenotype-to-disease matching."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from pydantic import BaseModel, Field

from biomedos.config import Settings, get_settings
from biomedos.core.llm_client import OllamaClient
from biomedos.data.hpo import HPOClient
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType


class MatchedPhenotype(BaseModel):
    """A resolved phenotype term."""

    hpo_id: str
    name: str
    score: float
    input_term: str


class DiseaseMatch(BaseModel):
    """A ranked phenotype-to-disease match."""

    disease_id: str
    disease_name: str
    score: float
    matched_hpo_terms: list[str] = Field(default_factory=list)
    supporting_terms: list[str] = Field(default_factory=list)


class PhenotypeMatcher:
    """Match HPO-like phenotypes to likely diseases."""

    def __init__(
        self,
        hpo_client: HPOClient | None = None,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the phenotype matcher."""

        self.hpo_client = hpo_client or HPOClient()
        self.llm_client = llm_client
        self.knowledge_graph = knowledge_graph
        self.settings = settings or get_settings()

    async def map_to_hpo(self, findings: str | list[str]) -> list[MatchedPhenotype]:
        """Map raw findings onto HPO terms."""

        symptom_terms = self._normalize_findings(findings)
        llm_terms = await self._extract_terms_with_llm(symptom_terms)
        search_terms = llm_terms or symptom_terms

        matches: list[MatchedPhenotype] = []
        seen_ids: set[str] = set()
        for term in search_terms:
            search_hits = await self.hpo_client.search(term, limit=3)
            for hit in search_hits:
                hpo_id = str(hit.get("id", ""))
                if not hpo_id or hpo_id in seen_ids:
                    continue
                seen_ids.add(hpo_id)
                matches.append(
                    MatchedPhenotype(
                        hpo_id=hpo_id,
                        name=str(hit.get("name", term)),
                        score=self._coerce_float(hit.get("score", 1.0)),
                        input_term=term,
                    )
                )
        return matches

    async def match(self, phenotypes: str | list[str], *, top_k: int = 10) -> list[DiseaseMatch]:
        """Rank diseases for a phenotype profile."""

        matched_terms = await self.map_to_hpo(phenotypes)
        disease_scores: dict[str, float] = {}
        disease_names: dict[str, str] = {}
        supporting_terms: dict[str, set[str]] = {}
        matched_hpo_terms: dict[str, set[str]] = {}

        for matched in matched_terms:
            associations = await self.hpo_client.get_disease_associations(matched.hpo_id)
            for association in associations:
                disease_id = str(association.get("disease_id", ""))
                if not disease_id:
                    continue
                association_score = self._coerce_float(association.get("score", 0.0))
                disease_scores[disease_id] = (
                    disease_scores.get(disease_id, 0.0) + association_score * matched.score
                )
                disease_names[disease_id] = str(association.get("disease_name", disease_id))
                supporting_terms.setdefault(disease_id, set()).add(matched.input_term)
                matched_hpo_terms.setdefault(disease_id, set()).add(matched.name)

        if self.knowledge_graph is not None:
            kg_scores = self._match_against_kg(matched_terms)
            for disease_id, kg_score in kg_scores.items():
                disease_scores[disease_id] = disease_scores.get(disease_id, 0.0) + kg_score
                disease = self.knowledge_graph.get_node(disease_id)
                if disease is not None:
                    disease_names[disease_id] = disease.name

        ranked = sorted(disease_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        return [
            DiseaseMatch(
                disease_id=disease_id,
                disease_name=disease_names.get(disease_id, disease_id),
                score=score,
                matched_hpo_terms=sorted(matched_hpo_terms.get(disease_id, set())),
                supporting_terms=sorted(supporting_terms.get(disease_id, set())),
            )
            for disease_id, score in ranked
        ]

    def _match_against_kg(self, matched_terms: list[MatchedPhenotype]) -> dict[str, float]:
        """Score diseases using knowledge-graph phenotype edges."""

        if self.knowledge_graph is None:
            return {}
        scores: dict[str, float] = {}
        for matched in matched_terms:
            phenotype_nodes = self.knowledge_graph.search_nodes(
                matched.name,
                node_type=NodeType.PHENOTYPE,
                limit=3,
            )
            for phenotype in phenotype_nodes:
                for edge in self.knowledge_graph.get_edges(
                    target_id=phenotype.id,
                    edge_type=EdgeType.DISEASE_PHENOTYPE,
                ):
                    scores[edge.source_id] = (
                        scores.get(edge.source_id, 0.0) + edge.score * matched.score
                    )
        return scores

    async def _extract_terms_with_llm(self, symptom_terms: list[str]) -> list[str]:
        """Use the LLM to extract normalized symptom phrases."""

        if self.llm_client is None or not symptom_terms or self.settings.FAST_LOCAL_MODE:
            return []
        prompt = (
            "Normalize the patient findings into concise phenotype terms.\n"
            'Return JSON with the schema {"terms": ["..."]}.\n'
            f"Findings: {', '.join(symptom_terms)}"
        )
        try:
            response = await self.llm_client.generate(
                prompt,
                model=self.settings.MODEL_ROUTER,
                system="Return only valid JSON.",
            )
            payload = self._parse_json(response)
        except ValueError:
            return []
        terms = payload.get("terms", [])
        if not isinstance(terms, list):
            return []
        normalized_terms: list[str] = []
        for term in terms:
            normalized = self._normalize_llm_term(term)
            if normalized:
                normalized_terms.append(normalized)
        return normalized_terms

    def _normalize_findings(self, findings: str | list[str]) -> list[str]:
        """Normalize raw findings into a list of symptom phrases."""

        if isinstance(findings, list):
            return [str(item).strip() for item in findings if str(item).strip()]
        parts = re.split(r",|;|\band\b|\n", findings)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _normalize_llm_term(term: object) -> str:
        """Normalize a possibly structured LLM term payload into plain text."""

        if isinstance(term, str):
            return term.strip()
        if isinstance(term, dict):
            for key in ("term", "name", "label", "code"):
                value = term.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return ""

    @staticmethod
    def _coerce_float(value: object) -> float:
        """Safely coerce a score-like object into a float."""

        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse the first JSON object from a model response."""

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)
        except json.JSONDecodeError:
            pass
        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)
        msg = "Failed to parse JSON phenotype normalization response."
        raise ValueError(msg)
