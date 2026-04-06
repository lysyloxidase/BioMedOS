"""Human Phenotype Ontology client."""

from __future__ import annotations

from collections import defaultdict

import httpx

from biomedos.data.base_client import AsyncAPIClient

LOCAL_HPO_TERMS: list[dict[str, object]] = [
    {"id": "HP:0012735", "name": "Chronic cough", "synonyms": ["cough", "persistent cough"]},
    {"id": "HP:0002094", "name": "Dyspnea", "synonyms": ["shortness of breath", "breathlessness"]},
    {"id": "HP:0002205", "name": "Pulmonary fibrosis", "synonyms": ["fibrotic lung disease"]},
    {"id": "HP:0002615", "name": "Hypoxemia", "synonyms": ["low oxygen"]},
    {"id": "HP:0001824", "name": "Weight loss", "synonyms": ["weight loss", "loss of weight"]},
]

LOCAL_HPO_DISEASES: dict[str, list[dict[str, object]]] = {
    "HP:0012735": [
        {"disease_id": "disease:lung_cancer", "disease_name": "Lung Cancer", "score": 0.82},
        {"disease_id": "disease:fibrosis", "disease_name": "Fibrosis", "score": 0.65},
    ],
    "HP:0002094": [
        {"disease_id": "disease:fibrosis", "disease_name": "Fibrosis", "score": 0.78},
        {"disease_id": "disease:lung_cancer", "disease_name": "Lung Cancer", "score": 0.69},
    ],
    "HP:0002205": [
        {"disease_id": "disease:fibrosis", "disease_name": "Fibrosis", "score": 0.95},
    ],
    "HP:0001824": [
        {"disease_id": "disease:lung_cancer", "disease_name": "Lung Cancer", "score": 0.74},
        {"disease_id": "disease:glioblastoma", "disease_name": "Glioblastoma", "score": 0.42},
    ],
}


class HPOClient(AsyncAPIClient):
    """Search and retrieve HPO terms."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Initialize the HPO client."""

        super().__init__(
            "https://ontology.jax.org/api/hpo",
            requests_per_second=5.0,
            max_concurrency=3,
            client=client,
        )

    async def search(self, query: str, *, limit: int = 10) -> list[dict[str, object]]:
        """Search HPO terms."""

        local_hits = self._local_search(query, limit=limit)
        if local_hits:
            return local_hits

        try:
            payload = await self._request_json(
                "GET",
                "/search",
                params={"q": query, "limit": limit},
            )
        except httpx.HTTPError:
            return []
        terms = payload.get("terms", [])
        if not isinstance(terms, list):
            return []
        return [term for term in terms if isinstance(term, dict)][:limit]

    async def get_disease_associations(self, hpo_id: str) -> list[dict[str, object]]:
        """Return diseases associated with an HPO phenotype."""

        if hpo_id in LOCAL_HPO_DISEASES:
            return LOCAL_HPO_DISEASES[hpo_id]

        try:
            payload = await self._request_json("GET", f"/term/{hpo_id}/diseases")
        except httpx.HTTPError:
            return []
        associations = payload.get("diseases", [])
        if not isinstance(associations, list):
            return []
        return [association for association in associations if isinstance(association, dict)]

    async def phenotype_similarity(
        self,
        left_hpo_ids: list[str],
        right_hpo_ids: list[str],
    ) -> float:
        """Compute a simple phenotype similarity score."""

        if not left_hpo_ids or not right_hpo_ids:
            return 0.0
        left = set(left_hpo_ids)
        right = set(right_hpo_ids)
        return len(left & right) / len(left | right)

    def _local_search(self, query: str, *, limit: int) -> list[dict[str, object]]:
        """Search the bundled local phenotype dictionary."""

        lowered = query.lower().strip()
        scores: dict[str, float] = defaultdict(float)
        payloads: dict[str, dict[str, object]] = {}
        for term in LOCAL_HPO_TERMS:
            name = str(term["name"]).lower()
            raw_synonyms = term.get("synonyms", [])
            synonyms = (
                [str(value).lower() for value in raw_synonyms]
                if isinstance(raw_synonyms, list)
                else []
            )
            if lowered == name:
                scores[str(term["id"])] += 2.0
            if lowered in name:
                scores[str(term["id"])] += 1.0
            if any(lowered in synonym or synonym in lowered for synonym in synonyms):
                scores[str(term["id"])] += 1.0
            if scores.get(str(term["id"]), 0.0) > 0.0:
                payloads[str(term["id"])] = {
                    "id": term["id"],
                    "name": term["name"],
                    "synonyms": list(raw_synonyms) if isinstance(raw_synonyms, list) else [],
                    "score": scores[str(term["id"])],
                }
        ranked = sorted(
            payloads.values(),
            key=lambda item: self._coerce_score(item.get("score")),
            reverse=True,
        )
        return ranked[:limit]

    @staticmethod
    def _coerce_score(value: object) -> float:
        """Safely coerce a score-like value into a float."""

        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0
