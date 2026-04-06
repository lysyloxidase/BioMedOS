"""Tests for knowledge-graph context generation."""

from __future__ import annotations

import pytest

from biomedos.rag.kg_context import KGContextExtractor


@pytest.mark.asyncio
async def test_extract_context_builds_triples_from_sample_graph(sample_kg: object) -> None:
    """KG context extraction returns grounded triples and a summary."""

    extractor = KGContextExtractor(sample_kg)

    context = await extractor.extract_context(
        "How is EGFR involved in lung cancer?",
        max_triples=5,
        depth=1,
    )

    assert "EGFR" in context.entities_found
    assert "Lung Cancer" in context.entities_found
    assert context.subgraph_size >= 2
    assert any(
        triple == "EGFR gene_associated_with_disease Lung Cancer" for triple in context.triples
    )
    assert "Relevant knowledge graph context:" in context.context_text
    assert "Matched entities:" in context.context_text
    assert "EGFR" in context.context_text
    assert "Lung Cancer" in context.context_text
