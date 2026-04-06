"""Link prediction training and inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from pydantic import BaseModel, Field

from biomedos.config import Settings, get_settings
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType
from biomedos.ml.dataset import EdgeSplit, GraphDatasetBuilder, GraphMLDataset
from biomedos.ml.embeddings import EmbeddingStore
from biomedos.ml.evaluation import LinkPredictionMetrics, evaluate_predictions
from biomedos.ml.models.base import BaseGNNModel
from biomedos.ml.models.graphsage import HeteroGraphSAGE
from biomedos.ml.models.node2vec import Node2VecBaseline
from biomedos.ml.models.rgcn import RGCNModel
from biomedos.ml.training import Trainer, TrainingConfig, TrainingResult


class PredictedLink(BaseModel):
    """A scored novel-link candidate."""

    source_id: str
    source_name: str
    target_id: str
    target_name: str
    edge_type: EdgeType
    score: float
    metadata: dict[str, object] = Field(default_factory=dict)


@dataclass(slots=True)
class PipelineTrainingSummary:
    """Summary of a trained link prediction pipeline."""

    model_name: str
    edge_type: EdgeType
    training_result: TrainingResult | None
    metrics: LinkPredictionMetrics


class LinkPredictionPipeline:
    """Train and evaluate knowledge graph link prediction models."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        *,
        model_name: str = "graphsage",
        edge_type: EdgeType | None = None,
        settings: Settings | None = None,
        dataset_builder: GraphDatasetBuilder | None = None,
    ) -> None:
        """Initialize the link prediction pipeline."""

        self.knowledge_graph = knowledge_graph
        self.model_name = model_name.lower()
        self.settings = settings or get_settings()
        self.edge_type = edge_type
        self.dataset_builder = dataset_builder or GraphDatasetBuilder(self.settings)
        self.dataset: GraphMLDataset | None = None
        self.model: BaseGNNModel | None = None
        self.trainer: Trainer | None = None
        self.embedding_store = EmbeddingStore()
        self.training_result: TrainingResult | None = None

    def prepare_data(
        self,
        *,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        negative_ratio: float = 1.0,
    ) -> GraphMLDataset:
        """Prepare a train/validation/test dataset."""

        self.dataset = self.dataset_builder.build_from_kg(
            self.knowledge_graph,
            edge_type=self.edge_type,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            negative_ratio=negative_ratio,
            seed=self.settings.RANDOM_SEED,
        )
        self.edge_type = self.dataset.heldout_edge_type
        return self.dataset

    def train(
        self,
        *,
        epochs: int | None = None,
        checkpoint_path: str | None = None,
    ) -> PipelineTrainingSummary:
        """Train the configured link prediction model."""

        dataset = self.dataset or self.prepare_data()
        self.model = self._build_model(dataset)

        if isinstance(self.model, Node2VecBaseline):
            self.model.fit(dataset)
            self.training_result = None
        else:
            self.trainer = Trainer(self.model, device=self.settings.DEVICE)
            config = TrainingConfig(
                epochs=epochs or self.settings.EPOCHS,
                learning_rate=self.settings.LEARNING_RATE,
                patience=self.settings.PATIENCE,
                checkpoint_path=checkpoint_path,
            )
            self.training_result = self.trainer.fit(dataset, config)

        self._refresh_embedding_store(dataset)
        metrics = self.evaluate(split="test")
        return PipelineTrainingSummary(
            model_name=self.model_name,
            edge_type=dataset.heldout_edge_type,
            training_result=self.training_result,
            metrics=metrics,
        )

    def evaluate(self, *, split: str = "test") -> LinkPredictionMetrics:
        """Evaluate the trained model on a split."""

        dataset = self._require_dataset()
        model = self._require_model()
        split_object = {
            "train": dataset.train_split,
            "val": dataset.val_split,
            "test": dataset.test_split,
        }.get(split, dataset.test_split)
        if split_object.labels.numel() == 0:
            split_object = dataset.train_split
        scores = self._score_split(model, dataset, split_object)
        labels = [int(value) for value in split_object.labels.tolist()]
        return evaluate_predictions(labels, [float(score) for score in scores])

    def predict_novel_links(
        self,
        source_id: str,
        target_type: str | NodeType,
        *,
        top_k: int = 10,
        edge_type: EdgeType | None = None,
    ) -> list[PredictedLink]:
        """Predict likely new links."""

        dataset = self._require_dataset()
        self._require_model()
        resolved_target_type = (
            target_type if isinstance(target_type, NodeType) else NodeType(str(target_type))
        )
        relation = edge_type or dataset.heldout_edge_type
        if source_id not in dataset.node_index:
            return []

        source_node = self.knowledge_graph.get_node(source_id)
        if source_node is None:
            return []

        candidate_indices = dataset.candidate_indices_for_type(resolved_target_type)
        existing_targets = {
            edge.target_id
            for edge in self.knowledge_graph.get_edges(source_id=source_id, edge_type=relation)
        }

        predictions: list[PredictedLink] = []
        for candidate_index in candidate_indices:
            target_id = dataset.node_ids[candidate_index]
            if target_id == source_id or target_id in existing_targets:
                continue
            score = self.score_link(source_id, target_id, edge_type=relation)
            target_node = self.knowledge_graph.get_node(target_id)
            if target_node is None:
                continue
            predictions.append(
                PredictedLink(
                    source_id=source_id,
                    source_name=source_node.name,
                    target_id=target_id,
                    target_name=target_node.name,
                    edge_type=relation,
                    score=score,
                    metadata={"target_type": target_node.node_type.value},
                )
            )
        predictions.sort(key=lambda item: item.score, reverse=True)
        return predictions[:top_k]

    def score_link(
        self,
        source_id: str,
        target_id: str,
        *,
        edge_type: EdgeType | None = None,
    ) -> float:
        """Score a specific candidate edge."""

        dataset = self._require_dataset()
        model = self._require_model()
        if source_id not in dataset.node_index or target_id not in dataset.node_index:
            return 0.0
        relation = edge_type or dataset.heldout_edge_type
        relation_tensor = torch.tensor([dataset.relation_id(relation)], dtype=torch.long)
        probabilities = model.predict(
            dataset.global_x,
            dataset.edge_index,
            dataset.relation_index,
            torch.tensor([dataset.node_index[source_id]], dtype=torch.long),
            torch.tensor([dataset.node_index[target_id]], dtype=torch.long),
            relation_tensor,
        )
        return float(probabilities[0].item())

    def save_embeddings(self, path: str | Path) -> None:
        """Persist current node embeddings."""

        self.embedding_store.save(path)

    def _score_split(
        self,
        model: BaseGNNModel,
        dataset: GraphMLDataset,
        split_object: EdgeSplit,
    ) -> np.ndarray:
        """Score a prepared split."""

        probabilities = model.predict(
            dataset.global_x,
            dataset.edge_index,
            dataset.relation_index,
            split_object.source_index,
            split_object.target_index,
            split_object.relation_index,
        )
        scores = np.asarray(probabilities.detach().cpu().numpy(), dtype=np.float32)
        return cast(np.ndarray[tuple[int], np.dtype[np.float32]], scores)

    def _build_model(self, dataset: GraphMLDataset) -> BaseGNNModel:
        """Instantiate the configured model."""

        if self.model_name == "graphsage":
            return HeteroGraphSAGE(
                input_dim=dataset.feature_dim,
                hidden_dim=self.settings.HIDDEN_DIM,
                output_dim=self.settings.EMBEDDING_DIM,
                num_relations=dataset.num_relations,
                num_layers=self.settings.GNN_LAYERS,
            )
        if self.model_name == "rgcn":
            return RGCNModel(
                input_dim=dataset.feature_dim,
                hidden_dim=self.settings.HIDDEN_DIM,
                output_dim=self.settings.EMBEDDING_DIM,
                num_relations=dataset.num_relations,
                num_layers=self.settings.GNN_LAYERS,
            )
        if self.model_name == "node2vec":
            return Node2VecBaseline(
                output_dim=self.settings.EMBEDDING_DIM,
                settings=self.settings,
            )
        msg = f"Unsupported model name: {self.model_name}"
        raise ValueError(msg)

    def _refresh_embedding_store(self, dataset: GraphMLDataset) -> None:
        """Refresh the persisted embedding view after training."""

        model = self._require_model()
        embeddings = model.encode(
            dataset.global_x,
            dataset.edge_index,
            dataset.relation_index,
        )
        embedding_map: dict[str, np.ndarray] = {}
        metadata_map: dict[str, dict[str, object]] = {}
        for node_id, index in dataset.node_index.items():
            node = self.knowledge_graph.get_node(node_id)
            embedding_map[node_id] = embeddings[index].detach().cpu().numpy()
            metadata_map[node_id] = {
                "node_type": node.node_type if node is not None else None,
                "name": node.name if node is not None else node_id,
            }
        self.embedding_store = EmbeddingStore(embedding_map, metadata_map)

    def _require_dataset(self) -> GraphMLDataset:
        """Return the prepared dataset or raise."""

        if self.dataset is None:
            self.prepare_data()
        if self.dataset is None:
            msg = "Dataset preparation failed."
            raise RuntimeError(msg)
        return self.dataset

    def _require_model(self) -> BaseGNNModel:
        """Return the trained model or raise."""

        if self.model is None:
            self.train(epochs=min(self.settings.EPOCHS, 30))
        if self.model is None:
            msg = "Model training failed."
            raise RuntimeError(msg)
        return self.model
