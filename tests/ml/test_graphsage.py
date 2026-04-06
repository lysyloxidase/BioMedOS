"""Tests for GraphSAGE and R-GCN models."""

from __future__ import annotations

from biomedos.graph.schema import EdgeType
from biomedos.ml.dataset import GraphDatasetBuilder
from biomedos.ml.models.graphsage import HeteroGraphSAGE
from biomedos.ml.models.rgcn import RGCNModel


def test_graph_models_produce_link_logits(sample_kg: object) -> None:
    """GraphSAGE and R-GCN encode nodes and score candidate links."""

    dataset = GraphDatasetBuilder().build_from_kg(sample_kg, edge_type=EdgeType.DRUG_DISEASE)
    graphsage = HeteroGraphSAGE(
        input_dim=dataset.feature_dim,
        num_relations=dataset.num_relations,
    )
    rgcn = RGCNModel(
        input_dim=dataset.feature_dim,
        num_relations=dataset.num_relations,
    )

    sage_logits = graphsage.score_pairs(
        dataset.global_x,
        dataset.edge_index,
        dataset.relation_index,
        dataset.train_split.source_index,
        dataset.train_split.target_index,
        dataset.train_split.relation_index,
    )
    rgcn_logits = rgcn.score_pairs(
        dataset.global_x,
        dataset.edge_index,
        dataset.relation_index,
        dataset.train_split.source_index,
        dataset.train_split.target_index,
        dataset.train_split.relation_index,
    )

    assert sage_logits.shape[0] == dataset.train_split.labels.shape[0]
    assert rgcn_logits.shape[0] == dataset.train_split.labels.shape[0]
