"""Train BioMedOS graph machine learning models."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biomedos.config import get_settings, resolve_project_path
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType
from biomedos.ml.evaluation import compare_models_table
from biomedos.ml.link_prediction import LinkPredictionPipeline


def main() -> int:
    """Script entrypoint for graph model training."""

    parser = argparse.ArgumentParser(description="Train BioMedOS graph ML models.")
    parser.add_argument(
        "--model",
        choices=["graphsage", "rgcn", "node2vec", "all"],
        default="graphsage",
        help="Model family to train.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Optional epoch override for trainable GNN models.",
    )
    parser.add_argument(
        "--edge-type",
        type=str,
        default=EdgeType.DRUG_DISEASE.value,
        help="Edge type to predict.",
    )
    parser.add_argument(
        "--graph",
        type=str,
        default=None,
        help="Path to a persisted knowledge graph.",
    )
    args = parser.parse_args()

    settings = get_settings()
    graph_path = resolve_project_path(args.graph or settings.GRAPH_PERSIST_PATH)
    knowledge_graph = KnowledgeGraph.load(graph_path)
    edge_type = EdgeType(args.edge_type)

    model_names = ["graphsage", "rgcn", "node2vec"] if args.model == "all" else [args.model]
    metrics_by_model = {}
    for model_name in model_names:
        pipeline = LinkPredictionPipeline(
            knowledge_graph,
            model_name=model_name,
            edge_type=edge_type,
            settings=settings,
        )
        summary = pipeline.train(
            epochs=args.epochs,
            checkpoint_path=f"data/checkpoints/{model_name}_{edge_type.value}.pt",
        )
        metrics_by_model[model_name] = summary.metrics

    print(compare_models_table(metrics_by_model))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
