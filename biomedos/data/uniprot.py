"""UniProt REST client."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from biomedos.data.base_client import AsyncAPIClient


class UniProtProtein(BaseModel):
    """A concise UniProt protein record."""

    accession: str
    protein_name: str | None = None
    gene_names: list[str] = Field(default_factory=list)
    organism: str | None = None
    length: int | None = None


class UniProtClient(AsyncAPIClient):
    """Client for searching the UniProt REST API."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Initialize the UniProt client."""

        super().__init__(
            "https://rest.uniprot.org",
            requests_per_second=5.0,
            client=client,
        )

    async def get_protein(self, accession: str) -> UniProtProtein | None:
        """Fetch a single UniProt entry."""

        payload = await self._request_json("GET", f"/uniprotkb/{accession}.json")
        if not payload:
            return None
        return self._parse_entry(payload)

    async def search_by_gene(
        self,
        gene_symbol: str,
        *,
        organism_id: int = 9606,
        limit: int = 10,
    ) -> list[UniProtProtein]:
        """Search UniProt entries by gene symbol."""

        payload = await self._request_json(
            "GET",
            "/uniprotkb/search",
            params={
                "query": f"gene:{gene_symbol} AND organism_id:{organism_id}",
                "format": "json",
                "size": limit,
            },
        )
        results = payload.get("results", [])
        if not isinstance(results, list):
            return []
        return [self._parse_entry(entry) for entry in results if isinstance(entry, dict)]

    def _parse_entry(self, payload: dict[str, object]) -> UniProtProtein:
        """Parse a UniProt JSON entry."""

        primary_accession = str(payload.get("primaryAccession", ""))
        protein_description = payload.get("proteinDescription", {})
        recommended_name = (
            protein_description.get("recommendedName", {})
            if isinstance(protein_description, dict)
            else {}
        )
        full_name = (
            recommended_name.get("fullName", {}) if isinstance(recommended_name, dict) else {}
        )

        genes = payload.get("genes", [])
        gene_names: list[str] = []
        if isinstance(genes, list):
            for gene in genes:
                if not isinstance(gene, dict):
                    continue
                gene_name = gene.get("geneName", {})
                if isinstance(gene_name, dict) and gene_name.get("value") is not None:
                    gene_names.append(str(gene_name.get("value")))

        organism = payload.get("organism", {})
        sequence = payload.get("sequence", {})
        return UniProtProtein(
            accession=primary_accession,
            protein_name=str(full_name.get("value")) if isinstance(full_name, dict) else None,
            gene_names=gene_names,
            organism=str(organism.get("scientificName"))
            if isinstance(organism, dict) and organism.get("scientificName") is not None
            else None,
            length=self._safe_int(sequence.get("length")) if isinstance(sequence, dict) else None,
        )

    @staticmethod
    def _safe_int(value: object) -> int | None:
        """Safely coerce a JSON scalar to an integer."""

        if value is None:
            return None
        return int(str(value))
