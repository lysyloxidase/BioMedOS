"""Tests for centrality analysis."""

from __future__ import annotations

from biomedos.analysis.centrality import DrugTargetRanker


def test_drug_target_ranker_scores_genes(sample_kg: object) -> None:
    """Centrality-based target ranking returns normalized scores."""

    ranker = DrugTargetRanker(sample_kg)
    rankings = ranker.rank_targets(top_k=5)

    assert rankings
    assert 0.0 <= rankings[0].druggability_score <= 1.0
    assert rankings[0].gene_name in {"EGFR", "BRCA1", "TP53", "ALK", "LOXL2"}
