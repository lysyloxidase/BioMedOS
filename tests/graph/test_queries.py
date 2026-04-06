"""Tests for graph query helpers."""

from __future__ import annotations

from biomedos.graph.queries import GraphQueryEngine


def test_find_drug_targets(sample_kg: object) -> None:
    """Drug target queries return the expected gene."""

    query_engine = GraphQueryEngine(sample_kg)
    targets = query_engine.find_drug_targets("Gefitinib")

    assert [target.name for target in targets] == ["EGFR"]


def test_find_repurposing_candidates(sample_kg: object) -> None:
    """Repurposing candidates are ranked for diseases."""

    query_engine = GraphQueryEngine(sample_kg)
    candidates = query_engine.find_repurposing_candidates("Lung Cancer")

    assert candidates
    assert candidates[0].drug_name == "Olaparib"
    assert "ALK" in candidates[0].target_genes


def test_shortest_path(sample_kg: object) -> None:
    """Shortest paths are returned when a path exists."""

    query_engine = GraphQueryEngine(sample_kg)
    path = query_engine.shortest_path("drug:olaparib", "disease:lung_cancer")

    assert path == ["drug:olaparib", "gene:alk", "disease:lung_cancer"]
