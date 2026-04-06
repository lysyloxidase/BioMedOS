"""Graph statistics and topology analysis."""

from __future__ import annotations

import networkx as nx

from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import NodeType


class GraphStatistics:
    """Convenience methods for network statistics."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        """Initialize the statistics helper."""

        self.knowledge_graph = knowledge_graph

    def degree_centrality(self) -> dict[str, float]:
        """Compute degree centrality on the graph."""

        return dict(nx.degree_centrality(self.knowledge_graph.graph.to_undirected()))

    def betweenness(self) -> dict[str, float]:
        """Compute betweenness centrality."""

        return dict(nx.betweenness_centrality(self.knowledge_graph.graph.to_undirected()))

    def pagerank(self) -> dict[str, float]:
        """Compute PageRank scores."""

        return dict(nx.pagerank(self.knowledge_graph.graph))

    def connected_components(self) -> list[list[str]]:
        """Return connected components of the undirected graph."""

        components = nx.connected_components(self.knowledge_graph.graph.to_undirected())
        return [sorted(component) for component in components]

    def hub_genes(self, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Return the highest-degree gene nodes."""

        scores = self.degree_centrality()
        ranked: list[tuple[str, float]] = []
        for node_id, score in scores.items():
            node = self.knowledge_graph.get_node(node_id)
            if node is not None and node.node_type == NodeType.GENE:
                ranked.append((node.name, score))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
