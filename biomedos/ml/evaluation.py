"""Evaluation metrics for graph learning."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import average_precision_score, roc_auc_score


class LinkPredictionMetrics(BaseModel):
    """Standard metrics for link prediction."""

    auroc: float
    auprc: float
    hits_at_10: float
    hits_at_50: float
    mrr: float


def compute_auroc(labels: list[int], scores: list[float]) -> float:
    """Compute AUROC for binary classification."""

    if len(set(labels)) < 2:
        return 0.5
    return float(roc_auc_score(labels, scores))


def compute_auprc(labels: list[int], scores: list[float]) -> float:
    """Compute AUPRC for binary classification."""

    if not labels:
        return 0.0
    return float(average_precision_score(labels, scores))


def compute_hits_at_k(labels: list[int], scores: list[float], *, k: int) -> float:
    """Compute Hits@K over positive edge ranks."""

    ranks = _positive_ranks(labels, scores)
    if not ranks:
        return 0.0
    return float(sum(1.0 for rank in ranks if rank <= k) / len(ranks))


def compute_mrr(labels: list[int], scores: list[float]) -> float:
    """Compute mean reciprocal rank."""

    ranks = _positive_ranks(labels, scores)
    if not ranks:
        return 0.0
    return float(sum(1.0 / rank for rank in ranks) / len(ranks))


def evaluate_predictions(labels: list[int], scores: list[float]) -> LinkPredictionMetrics:
    """Evaluate binary link predictions with ranking metrics."""

    return LinkPredictionMetrics(
        auroc=compute_auroc(labels, scores),
        auprc=compute_auprc(labels, scores),
        hits_at_10=compute_hits_at_k(labels, scores, k=10),
        hits_at_50=compute_hits_at_k(labels, scores, k=50),
        mrr=compute_mrr(labels, scores),
    )


def compare_models_table(metrics_by_model: dict[str, LinkPredictionMetrics]) -> str:
    """Render a Markdown comparison table for multiple models."""

    header = "| Model | AUROC | AUPRC | Hits@10 | Hits@50 | MRR |"
    divider = "| --- | ---: | ---: | ---: | ---: | ---: |"
    rows = [
        (
            f"| {model_name} | {metrics.auroc:.3f} | {metrics.auprc:.3f} | "
            f"{metrics.hits_at_10:.3f} | {metrics.hits_at_50:.3f} | {metrics.mrr:.3f} |"
        )
        for model_name, metrics in metrics_by_model.items()
    ]
    return "\n".join([header, divider, *rows])


def _positive_ranks(labels: Iterable[int], scores: Iterable[float]) -> list[int]:
    """Return one-based ranks for each positive example."""

    paired = list(zip(labels, scores, strict=True))
    if not paired:
        return []
    order = np.argsort([score for _, score in paired])[::-1]
    ranks: list[int] = []
    for position, index in enumerate(order, start=1):
        if paired[int(index)][0] == 1:
            ranks.append(position)
    return ranks
