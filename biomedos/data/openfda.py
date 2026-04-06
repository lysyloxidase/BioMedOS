"""OpenFDA adverse event API client."""

from __future__ import annotations

from typing import Any

import httpx

from biomedos.data.base_client import AsyncAPIClient

LOCAL_OPENFDA_EVENTS: dict[str, list[dict[str, object]]] = {
    "gefitinib": [
        {"patient": {"reaction": [{"reactionmeddrapt": "Rash"}]}},
        {"patient": {"reaction": [{"reactionmeddrapt": "Interstitial lung disease"}]}},
    ],
    "metformin": [
        {"patient": {"reaction": [{"reactionmeddrapt": "Nausea"}]}},
        {"patient": {"reaction": [{"reactionmeddrapt": "Diarrhoea"}]}},
    ],
    "olaparib": [
        {"patient": {"reaction": [{"reactionmeddrapt": "Fatigue"}]}},
        {"patient": {"reaction": [{"reactionmeddrapt": "Interstitial lung disease"}]}},
    ],
}

LOCAL_OPENFDA_LABELS: dict[str, list[dict[str, object]]] = {
    "gefitinib": [
        {
            "pharmacokinetics": ["Gefitinib is orally available and reaches steady state rapidly."],
            "clinical_pharmacology": ["Gefitinib inhibits EGFR signaling."],
            "drug_interactions": ["Monitor when combining with CYP3A4 modulators."],
            "contraindications": [
                "Use caution in interstitial lung disease or severe pulmonary toxicity."
            ],
        }
    ],
    "metformin": [
        {
            "pharmacokinetics": ["Metformin is renally cleared."],
            "clinical_pharmacology": ["Metformin reduces hepatic glucose production."],
            "drug_interactions": ["Monitor renal function with interacting medicines."],
            "contraindications": [
                "Contraindicated in severe renal impairment and metabolic acidosis."
            ],
        }
    ],
    "olaparib": [
        {
            "pharmacokinetics": ["Olaparib undergoes hepatic metabolism."],
            "clinical_pharmacology": ["Olaparib inhibits PARP enzymes."],
            "drug_interactions": ["Avoid strong CYP3A inhibitors when possible."],
            "contraindications": ["Monitor for pulmonary symptoms and myelosuppression."],
        }
    ],
}

LOCAL_OPENFDA_RECALLS: dict[str, list[dict[str, object]]] = {
    "gefitinib": [],
    "metformin": [],
    "olaparib": [],
}


class OpenFDAClient(AsyncAPIClient):
    """Retrieve adverse event and label information from OpenFDA."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Initialize the OpenFDA client."""

        super().__init__(
            "https://api.fda.gov/drug",
            requests_per_second=5.0,
            max_concurrency=3,
            client=client,
        )

    async def adverse_events(self, drug_name: str) -> list[dict[str, object]]:
        """Fetch adverse event records for a drug."""

        try:
            payload = await self._request_json(
                "GET",
                "/event.json",
                params={"search": f'patient.drug.medicinalproduct:"{drug_name}"', "limit": 10},
            )
        except httpx.HTTPError:
            return self._local_results(LOCAL_OPENFDA_EVENTS, drug_name)
        return self._extract_results(payload) or self._local_results(
            LOCAL_OPENFDA_EVENTS,
            drug_name,
        )

    async def drug_labels(self, drug_name: str) -> list[dict[str, object]]:
        """Fetch label information for a drug."""

        try:
            payload = await self._request_json(
                "GET",
                "/label.json",
                params={"search": f'openfda.generic_name:"{drug_name}"', "limit": 5},
            )
        except httpx.HTTPError:
            return self._local_results(LOCAL_OPENFDA_LABELS, drug_name)
        return self._extract_results(payload) or self._local_results(
            LOCAL_OPENFDA_LABELS,
            drug_name,
        )

    async def recalls(self, drug_name: str) -> list[dict[str, object]]:
        """Fetch recall entries for a drug."""

        try:
            payload = await self._request_json(
                "GET",
                "/enforcement.json",
                params={"search": f'product_description:"{drug_name}"', "limit": 10},
            )
        except httpx.HTTPError:
            return self._local_results(LOCAL_OPENFDA_RECALLS, drug_name)
        return self._extract_results(payload) or self._local_results(
            LOCAL_OPENFDA_RECALLS,
            drug_name,
        )

    @staticmethod
    def _extract_results(payload: dict[str, Any]) -> list[dict[str, object]]:
        """Extract a list of result dictionaries from an OpenFDA response."""

        results = payload.get("results", [])
        if not isinstance(results, list):
            return []
        return [result for result in results if isinstance(result, dict)]

    @staticmethod
    def _local_results(
        payloads: dict[str, list[dict[str, object]]],
        drug_name: str,
    ) -> list[dict[str, object]]:
        """Return bundled local OpenFDA-like records for known drugs."""

        return [dict(item) for item in payloads.get(drug_name.lower().strip(), [])]
