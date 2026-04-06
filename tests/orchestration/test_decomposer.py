"""Tests for query decomposition."""

from __future__ import annotations

from biomedos.orchestration.decomposer import QueryDecomposer
from biomedos.orchestration.state import TaskType


def test_query_decomposer_splits_multi_part_requests(sample_kg: object) -> None:
    """Complex queries are split into targeted specialist tasks."""

    decomposer = QueryDecomposer(knowledge_graph=sample_kg)

    tasks = decomposer.decompose(
        "What drugs target EGFR, "
        "check DDI between Gefitinib and Olaparib, "
        "and what clinical trials are running?"
    )

    assert len(tasks) == 3
    assert tasks[0].type == TaskType.GRAPH_EXPLORER
    assert tasks[1].type == TaskType.PHARMACOLOGIST
    assert tasks[2].type == TaskType.GRAPH_EXPLORER


def test_query_decomposer_extracts_path_endpoints_for_graph_queries(sample_kg: object) -> None:
    """Path-like graph queries should capture source and target node identifiers."""

    decomposer = QueryDecomposer(knowledge_graph=sample_kg)

    tasks = decomposer.decompose("Show the graph path between EGFR and Lung Cancer.")

    assert len(tasks) == 1
    assert tasks[0].type == TaskType.GRAPH_EXPLORER
    assert tasks[0].payload["source_id"] == "gene:egfr"
    assert tasks[0].payload["target_id"] == "disease:lung_cancer"


def test_query_decomposer_routes_generic_gene_questions_to_geneticist(sample_kg: object) -> None:
    """Gene-centric questions without explicit keywords should prefer the geneticist."""

    decomposer = QueryDecomposer(knowledge_graph=sample_kg)

    tasks = decomposer.decompose("What is main function of TP53?")

    assert len(tasks) == 1
    assert tasks[0].type == TaskType.GENETICIST
    assert tasks[0].payload["gene"] == "TP53"
