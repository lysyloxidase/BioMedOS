"""DisGeNET gene-disease association client."""

from __future__ import annotations


class DisGeNETClient:
    """Retrieve gene-disease associations from DisGeNET."""

    async def gene_disease_associations(self, gene_symbol: str) -> list[dict[str, object]]:
        """Fetch disease associations for a gene.

        Args:
            gene_symbol: HGNC gene symbol.

        Returns:
            Association records.
        """

        # TODO: Implement authenticated DisGeNET requests and response normalization.
        raise NotImplementedError(
            "DisGeNETClient.gene_disease_associations is not implemented yet."
        )
