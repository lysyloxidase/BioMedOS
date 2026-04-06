"""Tests for link prediction training."""

from __future__ import annotations

from pathlib import Path

from biomedos.graph.schema import EdgeType, NodeType
from biomedos.ml.evaluation import compare_models_table
from biomedos.ml.link_prediction import LinkPredictionPipeline


def test_link_prediction_pipeline_trains_and_exports_embeddings(
    sample_kg: object,
    tmp_path: Path,
) -> None:
    """Pipeline trains, evaluates, predicts, and saves embeddings."""

    pipeline = LinkPredictionPipeline(
        sample_kg,
        model_name="graphsage",
        edge_type=EdgeType.DRUG_DISEASE,
    )

    summary = pipeline.train(epochs=5)
    predictions = pipeline.predict_novel_links("drug:olaparib", NodeType.DISEASE, top_k=5)
    store = pipeline.embedding_store
    embedding_path = tmp_path / "embeddings.json"
    pipeline.save_embeddings(embedding_path)
    reloaded = store.load(embedding_path)
    metrics_table = compare_models_table({"graphsage": summary.metrics})

    assert 0.0 <= summary.metrics.auroc <= 1.0
    assert 0.0 <= summary.metrics.auprc <= 1.0
    assert predictions
    assert reloaded.most_similar("gene:egfr", top_k=3)
    assert "| Model | AUROC |" in metrics_table
