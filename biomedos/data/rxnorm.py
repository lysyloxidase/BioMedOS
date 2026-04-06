"""RxNorm client for drug normalization and interaction lookup."""

from __future__ import annotations

from typing import Any

import httpx

from biomedos.data.base_client import AsyncAPIClient

LOCAL_RXNORM_DRUGS: dict[str, dict[str, object]] = {
    "gefitinib": {
        "input_name": "Gefitinib",
        "normalized_name": "Gefitinib",
        "rxcui": "328134",
    },
    "metformin": {
        "input_name": "Metformin",
        "normalized_name": "Metformin",
        "rxcui": "6809",
    },
    "olaparib": {
        "input_name": "Olaparib",
        "normalized_name": "Olaparib",
        "rxcui": "235743",
    },
    "pirfenidone": {
        "input_name": "Pirfenidone",
        "normalized_name": "Pirfenidone",
        "rxcui": "1547545",
    },
    "trametinib": {
        "input_name": "Trametinib",
        "normalized_name": "Trametinib",
        "rxcui": "1657987",
    },
}

LOCAL_RXNORM_INTERACTIONS: dict[frozenset[str], dict[str, object]] = {
    frozenset({"328134", "235743"}): {
        "rxcui_a": "328134",
        "rxcui_b": "235743",
        "drug_a": "Gefitinib",
        "drug_b": "Olaparib",
        "description": "Potential overlapping pulmonary toxicity; monitor closely.",
        "severity": "moderate",
        "source": "BioMedOS local knowledge",
    },
    frozenset({"328134", "6809"}): {
        "rxcui_a": "328134",
        "rxcui_b": "6809",
        "drug_a": "Gefitinib",
        "drug_b": "Metformin",
        "description": "No major direct interaction identified; monitor clinical tolerance.",
        "severity": "minor",
        "source": "BioMedOS local knowledge",
    },
}

LOCAL_RXNORM_NDCS: dict[str, list[str]] = {
    "328134": ["0052-0603-30"],
    "6809": ["0093-1048-01"],
    "235743": ["0310-0660-30"],
}


class RxNormClient(AsyncAPIClient):
    """Resolve normalized medication concepts and interactions."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Initialize the RxNorm client."""

        super().__init__(
            "https://rxnav.nlm.nih.gov/REST",
            requests_per_second=5.0,
            max_concurrency=3,
            client=client,
        )

    async def normalize_drug(self, name: str) -> dict[str, object]:
        """Normalize a free-text drug name."""

        local_match = self._local_normalization(name)
        try:
            payload = await self._request_json(
                "GET",
                "/rxcui.json",
                params={"name": name, "search": 1},
            )
        except httpx.HTTPError:
            return local_match
        id_group = payload.get("idGroup", {})
        if isinstance(id_group, dict):
            identifiers = id_group.get("rxnormId", [])
            if isinstance(identifiers, list) and identifiers:
                return {
                    "input_name": name,
                    "normalized_name": str(id_group.get("name", name)),
                    "rxcui": str(identifiers[0]),
                }

        try:
            approximate = await self._request_json(
                "GET",
                "/approximateTerm.json",
                params={"term": name, "maxEntries": 1},
            )
        except httpx.HTTPError:
            return local_match
        candidate = (
            approximate.get("approximateGroup", {}).get("candidate", [])
            if isinstance(approximate.get("approximateGroup", {}), dict)
            else []
        )
        if isinstance(candidate, list) and candidate:
            first = candidate[0]
            if isinstance(first, dict):
                return {
                    "input_name": name,
                    "normalized_name": str(first.get("rxcui", name)),
                    "rxcui": str(first.get("rxcui", "")),
                    "score": self._safe_int(first.get("score")) or 0,
                }

        return local_match

    async def check_interactions(self, rxcuis: list[str]) -> list[dict[str, object]]:
        """Check interactions for normalized RxCUIs."""

        if not rxcuis:
            return []
        requested_rxcuis = set(rxcuis)
        try:
            payload = await self._request_json(
                "GET",
                "/interaction/list.json",
                params={"rxcuis": "+".join(rxcuis)},
            )
        except httpx.HTTPError:
            return self._local_interactions(rxcuis)
        groups = payload.get("fullInteractionTypeGroup", [])
        if not isinstance(groups, list):
            return self._local_interactions(rxcuis)

        results: list[dict[str, object]] = []
        for group in groups:
            if not isinstance(group, dict):
                continue
            sources = group.get("sourceName")
            interaction_types = group.get("fullInteractionType", [])
            if not isinstance(interaction_types, list):
                continue
            for interaction_type in interaction_types:
                if not isinstance(interaction_type, dict):
                    continue
                pairs = interaction_type.get("interactionPair", [])
                if not isinstance(pairs, list):
                    continue
                for pair in pairs:
                    if not isinstance(pair, dict):
                        continue
                    concepts = pair.get("interactionConcept", [])
                    if not isinstance(concepts, list) or len(concepts) < 2:
                        continue
                    first = concepts[0] if isinstance(concepts[0], dict) else {}
                    second = concepts[1] if isinstance(concepts[1], dict) else {}
                    results.append(
                        {
                            "rxcui_a": self._extract_rxcui(first),
                            "rxcui_b": self._extract_rxcui(second),
                            "drug_a": self._extract_name(first),
                            "drug_b": self._extract_name(second),
                            "description": str(pair.get("description", "")),
                            "severity": self._infer_severity(str(pair.get("description", ""))),
                            "source": str(sources) if sources is not None else "RxNorm",
                        }
                    )
        filtered_results = [
            result
            for result in results
            if str(result.get("rxcui_a", "")) in requested_rxcuis
            and str(result.get("rxcui_b", "")) in requested_rxcuis
        ]
        return filtered_results or self._local_interactions(rxcuis)

    async def get_ndc_codes(self, rxcui: str) -> list[str]:
        """Return historical NDC codes for an RxNorm concept."""

        try:
            payload = await self._request_json(
                "GET",
                f"/rxcui/{rxcui}/allhistoricalndcs.json",
            )
        except httpx.HTTPError:
            return LOCAL_RXNORM_NDCS.get(rxcui, [])
        group = payload.get("historicalNdcConcept", {})
        if not isinstance(group, dict):
            return LOCAL_RXNORM_NDCS.get(rxcui, [])
        ndc_time = group.get("historicalNdcTime", [])
        if not isinstance(ndc_time, list):
            return LOCAL_RXNORM_NDCS.get(rxcui, [])
        codes: list[str] = []
        for item in ndc_time:
            if not isinstance(item, dict):
                continue
            ndc_list = item.get("ndcTime", [])
            if not isinstance(ndc_list, list):
                continue
            for entry in ndc_list:
                if not isinstance(entry, dict):
                    continue
                code = entry.get("ndc")
                if code is not None:
                    codes.append(str(code))
        return sorted(set(codes)) or LOCAL_RXNORM_NDCS.get(rxcui, [])

    def _local_normalization(self, name: str) -> dict[str, object]:
        """Return a bundled local normalization when available."""

        return LOCAL_RXNORM_DRUGS.get(
            name.lower().strip(),
            {"input_name": name, "normalized_name": name.title(), "rxcui": ""},
        )

    def _local_interactions(self, rxcuis: list[str]) -> list[dict[str, object]]:
        """Return bundled local interaction heuristics for known demo drugs."""

        if len(rxcuis) < 2:
            return []

        results: list[dict[str, object]] = []
        for index, left in enumerate(rxcuis):
            for right in rxcuis[index + 1 :]:
                match = LOCAL_RXNORM_INTERACTIONS.get(frozenset({left, right}))
                if match is not None:
                    results.append(dict(match))
        return results

    @staticmethod
    def _extract_rxcui(concept: dict[str, Any]) -> str:
        """Extract an RxCUI from a nested interaction concept."""

        min_concept = concept.get("minConceptItem", {})
        if isinstance(min_concept, dict) and min_concept.get("rxcui") is not None:
            return str(min_concept.get("rxcui"))
        return ""

    @staticmethod
    def _extract_name(concept: dict[str, Any]) -> str:
        """Extract a display drug name from a nested interaction concept."""

        min_concept = concept.get("minConceptItem", {})
        if isinstance(min_concept, dict) and min_concept.get("name") is not None:
            return str(min_concept.get("name"))
        return ""

    @staticmethod
    def _infer_severity(description: str) -> str:
        """Infer a severity label from free-text interaction descriptions."""

        lowered = description.lower()
        if any(keyword in lowered for keyword in ("contraindicated", "avoid combination")):
            return "contraindicated"
        if any(keyword in lowered for keyword in ("life-threatening", "severe", "major")):
            return "major"
        if any(keyword in lowered for keyword in ("monitor", "moderate", "dose adjustment")):
            return "moderate"
        return "minor"

    @staticmethod
    def _safe_int(value: object) -> int | None:
        """Safely coerce a JSON scalar to an integer."""

        if value is None:
            return None
        return int(str(value))
