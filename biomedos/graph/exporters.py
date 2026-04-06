"""Graph export utilities."""

from __future__ import annotations

from biomedos.graph.builder import KnowledgeGraph


class Neo4jExporter:
    """Export a knowledge graph into Neo4j import format."""

    def export(self, knowledge_graph: KnowledgeGraph, output_dir: str) -> None:
        """Export nodes and edges into Neo4j CSV files.

        Args:
            knowledge_graph: Source graph.
            output_dir: Target output directory.
        """

        # TODO: Write node and edge CSV files compatible with Neo4j bulk import.
        raise NotImplementedError("Neo4jExporter.export is not implemented yet.")


class GraphMLExporter:
    """Export a knowledge graph as GraphML."""

    def export(self, knowledge_graph: KnowledgeGraph, path: str) -> None:
        """Export the graph to GraphML.

        Args:
            knowledge_graph: Source graph.
            path: Output file path.
        """

        # TODO: Serialize graph attributes into GraphML-safe fields.
        raise NotImplementedError("GraphMLExporter.export is not implemented yet.")
