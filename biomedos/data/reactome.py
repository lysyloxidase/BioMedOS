"""Reactome pathway API client."""

from __future__ import annotations


class ReactomeClient:
    """Retrieve pathway context from Reactome."""

    async def pathways_for_gene(self, gene_symbol: str) -> list[dict[str, object]]:
        """Fetch pathways associated with a gene.

        Args:
            gene_symbol: HGNC gene symbol.

        Returns:
            Pathway records.
        """

        # TODO: Implement Reactome pathway lookups and identifier mapping.
        raise NotImplementedError("ReactomeClient.pathways_for_gene is not implemented yet.")
