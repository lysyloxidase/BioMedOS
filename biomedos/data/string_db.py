"""STRING DB protein-protein interaction client."""

from __future__ import annotations

import httpx
from pydantic import BaseModel

from biomedos.data.base_client import AsyncAPIClient


class ProteinInteraction(BaseModel):
    """A STRING DB interaction record."""

    preferred_name_a: str
    preferred_name_b: str
    score: float
    evidence: str | None = None


class StringDBClient(AsyncAPIClient):
    """Client for the STRING DB network API."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Initialize the STRING DB client."""

        super().__init__(
            "https://string-db.org/api/json",
            requests_per_second=5.0,
            client=client,
        )

    async def get_network(
        self,
        genes: list[str],
        *,
        species: int = 9606,
        required_score: int = 400,
    ) -> list[ProteinInteraction]:
        """Retrieve a protein interaction network for a list of genes."""

        if not genes:
            return []

        response = await self._request(
            "GET",
            "/network",
            params={
                "identifiers": "%0d".join(genes),
                "species": species,
                "required_score": required_score,
            },
        )
        payload = response.json()
        rows = payload if isinstance(payload, list) else payload.get("network", [])
        if not isinstance(rows, list):
            return []
        interactions: list[ProteinInteraction] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            interactions.append(
                ProteinInteraction(
                    preferred_name_a=str(row.get("preferredName_A", "")),
                    preferred_name_b=str(row.get("preferredName_B", "")),
                    score=float(row.get("score", 0.0)),
                    evidence=str(row.get("annotation"))
                    if row.get("annotation") is not None
                    else None,
                )
            )
        return interactions
