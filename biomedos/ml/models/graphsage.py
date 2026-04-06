"""Heterogeneous GraphSAGE model definition."""

from __future__ import annotations

import torch
from torch import nn

from biomedos.ml.models.base import BaseGNNModel


class GraphSAGELayer(nn.Module):
    """A mean-aggregation GraphSAGE layer."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Initialize the layer."""

        super().__init__()
        self.self_linear = nn.Linear(in_dim, out_dim)
        self.neighbor_linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Aggregate neighbor features and update node representations."""

        if edge_index.numel() == 0:
            return torch.relu(self.self_linear(x))

        source_index, target_index = edge_index
        aggregated = torch.zeros_like(x)
        aggregated.index_add_(0, target_index, x[source_index])
        degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        degree.index_add_(
            0,
            target_index,
            torch.ones(target_index.shape[0], device=x.device, dtype=x.dtype),
        )
        degree = degree.clamp(min=1.0).unsqueeze(-1)
        mean_neighbors = aggregated / degree
        return torch.relu(self.self_linear(x) + self.neighbor_linear(mean_neighbors))


class HeteroGraphSAGE(BaseGNNModel):
    """GraphSAGE baseline for heterogeneous biomedical graphs."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_relations: int = 1,
        num_layers: int = 3,
    ) -> None:
        """Initialize the GraphSAGE encoder."""

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_relations=num_relations,
        )
        layer_dims = [input_dim, *([hidden_dim] * (num_layers - 1)), output_dim]
        self.layers = nn.ModuleList(
            GraphSAGELayer(in_dim, out_dim)
            for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:], strict=True)
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode graph nodes into dense embeddings."""

        del relation_index
        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, edge_index)
            if index == len(self.layers) - 1:
                hidden = torch.nn.functional.normalize(hidden, p=2.0, dim=-1)
        return hidden
