"""Abstract graph neural network model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseGNNModel(nn.Module, ABC):
    """Abstract base interface for graph models."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_relations: int,
    ) -> None:
        """Initialize common link-prediction modules."""

        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_relations = max(num_relations, 1)
        self.relation_embeddings = nn.Embedding(self.num_relations, output_dim)
        self.link_head = nn.Sequential(
            nn.Linear(output_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    @abstractmethod
    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode graph nodes into dense embeddings."""

    def decode(
        self,
        embeddings: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """Decode node embeddings into link logits."""

        if source_index.numel() == 0:
            return torch.empty(0, device=embeddings.device)

        clipped_relations = relation_index.clamp(min=0, max=max(self.num_relations - 1, 0))
        source_embedding = embeddings[source_index]
        target_embedding = embeddings[target_index]
        relation_embedding = self.relation_embeddings(clipped_relations)
        features = torch.cat(
            [
                source_embedding,
                target_embedding,
                torch.abs(source_embedding - target_embedding),
                source_embedding * target_embedding + relation_embedding,
            ],
            dim=-1,
        )
        return self.link_head(features).squeeze(-1)

    def score_pairs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        graph_relation_index: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        pair_relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits for candidate node pairs."""

        embeddings = self.encode(x, edge_index, graph_relation_index)
        return self.decode(embeddings, source_index, target_index, pair_relation_index)

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        graph_relation_index: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        pair_relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return link probabilities for candidate node pairs."""

        return torch.sigmoid(
            self.score_pairs(
                x,
                edge_index,
                graph_relation_index,
                source_index,
                target_index,
                pair_relation_index,
            )
        )

    def fit(self, *args: object, **kwargs: object) -> None:
        """Keep a scikit-learn style API surface for callers."""

        del args, kwargs
        msg = "Use Trainer.fit(...) or a model-specific fit implementation."
        raise RuntimeError(msg)
