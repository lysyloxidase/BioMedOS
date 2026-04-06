"""Tests for graph dataset conversion."""

from __future__ import annotations

from biomedos.graph.schema import EdgeType, NodeType
from biomedos.ml.dataset import GraphDatasetBuilder


def test_build_dataset_from_sample_graph(sample_kg: object) -> None:
    """Dataset builder produces typed splits and features."""

    builder = GraphDatasetBuilder()
    dataset = builder.build_from_kg(
        sample_kg,
        edge_type=EdgeType.DRUG_DISEASE,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    assert dataset.num_nodes == 30
    assert dataset.heldout_edge_type == EdgeType.DRUG_DISEASE
    assert dataset.source_node_type == NodeType.DRUG
    assert dataset.target_node_type == NodeType.DISEASE
    assert dataset.feature_dim >= len(NodeType)
    assert dataset.train_split.labels.numel() > 0
    assert dataset.edge_index.shape[0] == 2
    assert dataset.candidate_indices_for_type(NodeType.DISEASE)
