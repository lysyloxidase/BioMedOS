"""Bulk graph importers for public biomedical knowledge graphs."""

from __future__ import annotations

from biomedos.graph.builder import KnowledgeGraph


class PrimeKGImporter:
    """Import PrimeKG into the local graph."""

    def load(self, path: str) -> KnowledgeGraph:
        """Load PrimeKG from a local file path.

        Args:
            path: Source data path.

        Returns:
            Imported knowledge graph.
        """

        # TODO: Parse PrimeKG tables and map them to the BioMedOS schema.
        raise NotImplementedError("PrimeKGImporter.load is not implemented yet.")


class HetionetImporter:
    """Import Hetionet into the local graph."""

    def load(self, path: str) -> KnowledgeGraph:
        """Load Hetionet from a local file path."""

        # TODO: Parse Hetionet artifacts and map them to the BioMedOS schema.
        raise NotImplementedError("HetionetImporter.load is not implemented yet.")


class DRKGImporter:
    """Import DRKG into the local graph."""

    def load(self, path: str) -> KnowledgeGraph:
        """Load DRKG from a local file path."""

        # TODO: Parse DRKG edge lists and map them to the BioMedOS schema.
        raise NotImplementedError("DRKGImporter.load is not implemented yet.")
