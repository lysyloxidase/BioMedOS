"""Tests for similarity analysis."""

from __future__ import annotations

import numpy as np

from biomedos.analysis.similarity import BioSimilarity
from biomedos.ml.embeddings import EmbeddingStore


def test_similarity_uses_embedding_store(sample_kg: object) -> None:
    """Similarity search returns same-type neighbors from stored embeddings."""

    store = EmbeddingStore(
        embeddings={
            "gene:egfr": np.array([1.0, 0.0, 0.0]),
            "gene:alk": np.array([0.9, 0.1, 0.0]),
            "drug:gefitinib": np.array([1.0, 0.0, 0.1]),
        },
        metadata={
            "gene:egfr": {"node_type": "Gene"},
            "gene:alk": {"node_type": "Gene"},
            "drug:gefitinib": {"node_type": "Drug"},
        },
    )
    analyzer = BioSimilarity(sample_kg, embedding_store=store)

    hits = analyzer.gene_similarity("gene:egfr", top_k=2)

    assert hits
    assert hits[0].node_id == "gene:alk"
