"""Relational graph convolution network definition."""

from __future__ import annotations

import torch
from torch import nn

from biomedos.ml.models.base import BaseGNNModel


class RGCNLayer(nn.Module):
    """A simple R-GCN layer with basis decomposition."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        num_relations: int,
        num_bases: int,
    ) -> None:
        """Initialize the R-GCN layer."""

        super().__init__()
        self.out_dim = out_dim
        self.num_relations = max(num_relations, 1)
        self.num_bases = max(1, min(num_bases, self.num_relations))
        self.bases = nn.Parameter(torch.empty(self.num_bases, in_dim, out_dim))
        self.coefficients = nn.Parameter(torch.empty(self.num_relations, self.num_bases))
        self.self_loop = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.bases)
        nn.init.xavier_uniform_(self.coefficients)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """Perform relation-aware message passing."""

        hidden = self.self_loop(x)
        if edge_index.numel() == 0:
            return torch.relu(hidden)

        source_index, target_index = edge_index
        for relation_id in range(self.num_relations):
            relation_mask = relation_index == relation_id
            if not torch.any(relation_mask):
                continue
            rel_source = source_index[relation_mask]
            rel_target = target_index[relation_mask]
            weight = torch.einsum("b,bij->ij", self.coefficients[relation_id], self.bases)
            messages = x[rel_source] @ weight
            aggregated = torch.zeros((x.shape[0], self.out_dim), device=x.device, dtype=x.dtype)
            aggregated.index_add_(0, rel_target, messages)
            degree = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            degree.index_add_(
                0,
                rel_target,
                torch.ones(rel_target.shape[0], device=x.device, dtype=x.dtype),
            )
            hidden = hidden + aggregated / degree.clamp(min=1.0).unsqueeze(-1)
        return torch.relu(hidden)


class RGCNModel(BaseGNNModel):
    """R-GCN model for multi-relational reasoning."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_relations: int = 1,
        num_layers: int = 3,
        num_bases: int = 4,
    ) -> None:
        """Initialize the R-GCN model."""

        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_relations=num_relations,
        )
        layer_dims = [input_dim, *([hidden_dim] * (num_layers - 1)), output_dim]
        self.layers = nn.ModuleList(
            RGCNLayer(
                in_dim,
                out_dim,
                num_relations=num_relations,
                num_bases=num_bases,
            )
            for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:], strict=True)
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """Encode graph nodes into dense embeddings."""

        hidden = x
        for index, layer in enumerate(self.layers):
            hidden = layer(hidden, edge_index, relation_index)
            if index == len(self.layers) - 1:
                hidden = torch.nn.functional.normalize(hidden, p=2.0, dim=-1)
        return hidden
