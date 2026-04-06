"""Tests for enrichment analysis."""

from __future__ import annotations

from biomedos.analysis.enrichment import GeneSetEnrichment


def test_pathway_enrichment_identifies_egfr_signaling(sample_kg: object) -> None:
    """Pathway enrichment ranks the matching pathway for EGFR/ALK genes."""

    analyzer = GeneSetEnrichment(sample_kg)
    results = analyzer.pathway_enrichment(["EGFR", "ALK"])

    assert results
    assert results[0].term_name == "EGFR Signaling"
    assert results[0].overlap >= 1
    assert results[0].fdr <= 1.0
