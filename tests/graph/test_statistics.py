"""Tests for graph statistics."""

from __future__ import annotations

from biomedos.graph.statistics import GraphStatistics


def test_statistics_methods(sample_kg: object) -> None:
    """Centrality and PageRank metrics return sensible values."""

    statistics = GraphStatistics(sample_kg)
    degree = statistics.degree_centrality()
    pagerank = statistics.pagerank()
    hubs = statistics.hub_genes(top_k=3)

    assert "gene:egfr" in degree
    assert abs(sum(pagerank.values()) - 1.0) < 1e-6
    assert hubs
    assert hubs[0][0] in {"EGFR", "TP53", "BRCA1"}
