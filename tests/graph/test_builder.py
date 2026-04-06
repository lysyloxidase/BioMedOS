"""Tests for the knowledge graph builder."""

from __future__ import annotations

from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import BioEdge, EdgeType, GeneNode


def test_builder_add_merge_save_and_load(tmp_path: object) -> None:
    """Nodes and edges can be added, merged, persisted, and restored."""

    graph = KnowledgeGraph()
    graph.add_node(GeneNode(id="gene:egfr", name="EGFR", symbol="EGFR"))
    graph.merge_node(GeneNode(id="gene:egfr", name="EGFR", symbol="EGFR", description="Kinase"))
    graph.add_node(GeneNode(id="gene:alk", name="ALK", symbol="ALK"))
    graph.add_edge(
        BioEdge(
            source_id="gene:egfr", target_id="gene:alk", edge_type=EdgeType.GENE_GENE, score=0.7
        )
    )
    graph.merge_edge(
        BioEdge(
            source_id="gene:egfr", target_id="gene:alk", edge_type=EdgeType.GENE_GENE, score=0.9
        )
    )

    node = graph.get_node("gene:egfr")
    assert node is not None
    assert node.description == "Kinase"
    assert graph.get_edges(edge_type=EdgeType.GENE_GENE)[0].score == 0.9

    output = tmp_path / "graph.pkl"
    graph.save(output)
    loaded = KnowledgeGraph.load(output)

    assert loaded.graph.number_of_nodes() == 2
    assert loaded.graph.number_of_edges() == 1
    assert loaded.stats()["node_types"]["Gene"] == 2
