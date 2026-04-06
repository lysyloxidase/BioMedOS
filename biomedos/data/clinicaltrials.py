"""ClinicalTrials.gov API v2 client."""

from __future__ import annotations


class ClinicalTrialsClient:
    """Search ClinicalTrials.gov for interventional studies."""

    async def search(self, query: str, *, limit: int = 20) -> list[dict[str, object]]:
        """Search trials for a biomedical query.

        Args:
            query: Search term or disease/drug string.
            limit: Maximum number of trials to return.

        Returns:
            Trial summaries.
        """

        # TODO: Implement ClinicalTrials.gov v2 search and normalization.
        raise NotImplementedError("ClinicalTrialsClient.search is not implemented yet.")
