"""Tests for the graph schema models."""

from __future__ import annotations

from biomedos.graph.schema import BioEdge, EdgeType, GeneNode, NodeType


def test_schema_enumerations_and_models() -> None:
    """All expected node and edge types are available."""

    assert len(NodeType) == 12
    assert len(EdgeType) == 18

    node = GeneNode(id="gene:egfr", name="EGFR", symbol="EGFR")
    edge = BioEdge(source_id="gene:egfr", target_id="disease:lung", edge_type=EdgeType.GENE_DISEASE)

    assert node.node_type == NodeType.GENE
    assert edge.edge_type == EdgeType.GENE_DISEASE
