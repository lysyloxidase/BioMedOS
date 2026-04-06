"""CPU-friendly Node2Vec baseline."""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from biomedos.config import Settings, get_settings
from biomedos.ml.dataset import GraphMLDataset
from biomedos.ml.models.base import BaseGNNModel


class Node2VecBaseline(BaseGNNModel):
    """Lightweight Node2Vec baseline model."""

    def __init__(
        self,
        *,
        output_dim: int = 128,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the baseline."""

        self.settings = settings or get_settings()
        super().__init__(
            input_dim=1,
            hidden_dim=max(output_dim, 32),
            output_dim=output_dim,
            num_relations=1,
        )
        self.register_buffer("_cached_embeddings", torch.empty(0), persistent=False)

    def fit(
        self,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Train the Node2Vec-style baseline with random walks."""

        if not args or not isinstance(args[0], GraphMLDataset):
            msg = "Node2VecBaseline.fit requires a GraphMLDataset instance."
            raise TypeError(msg)
        dataset = args[0]
        walk_length = self._coerce_int(kwargs.get("walk_length"), 10)
        num_walks = self._coerce_int(kwargs.get("num_walks"), 12)
        window_size = self._coerce_int(kwargs.get("window_size"), 3)
        seed_value = kwargs.get("seed")
        seed = seed_value if isinstance(seed_value, int) else None
        rng = random.Random(self.settings.RANDOM_SEED if seed is None else seed)
        neighbors = self._build_neighbors(dataset)
        node_count = dataset.num_nodes
        if node_count == 1:
            self._cached_embeddings = torch.ones((1, self.output_dim), dtype=torch.float32)
            return

        co_occurrence: np.ndarray[tuple[int, int], np.dtype[np.float32]] = np.zeros(
            (node_count, node_count),
            dtype=np.float32,
        )
        for node_index in range(node_count):
            for _ in range(num_walks):
                walk = self._random_walk(
                    node_index,
                    neighbors=neighbors,
                    walk_length=walk_length,
                    rng=rng,
                )
                for center, center_node in enumerate(walk):
                    start = max(0, center - window_size)
                    end = min(len(walk), center + window_size + 1)
                    for context_pos in range(start, end):
                        if context_pos == center:
                            continue
                        context_node = walk[context_pos]
                        co_occurrence[center_node, context_node] += 1.0

        if float(co_occurrence.sum()) == 0.0:
            co_occurrence += np.eye(node_count, dtype=np.float32)

        n_components = min(self.output_dim, max(node_count - 1, 1))
        svd = TruncatedSVD(n_components=n_components, random_state=self.settings.RANDOM_SEED)
        embeddings = svd.fit_transform(co_occurrence)
        normalized = normalize(embeddings)
        if normalized.shape[1] < self.output_dim:
            padding = np.zeros(
                (node_count, self.output_dim - normalized.shape[1]),
                dtype=np.float32,
            )
            normalized = np.concatenate([normalized, padding], axis=1)
        self._cached_embeddings = torch.tensor(
            normalized[:, : self.output_dim],
            dtype=torch.float32,
        )

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return cached baseline embeddings."""

        del x, edge_index, relation_index
        if self._cached_embeddings.numel() == 0:
            msg = "Node2VecBaseline has not been fitted yet."
            raise RuntimeError(msg)
        return self._cached_embeddings

    def score_pairs(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        graph_relation_index: torch.Tensor,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        pair_relation_index: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate links with cosine similarity."""

        del x, edge_index, graph_relation_index, pair_relation_index
        embeddings = self.encode(torch.empty(0), torch.empty(0), torch.empty(0))
        source_embedding = torch.nn.functional.normalize(embeddings[source_index], p=2.0, dim=-1)
        target_embedding = torch.nn.functional.normalize(embeddings[target_index], p=2.0, dim=-1)
        return (source_embedding * target_embedding).sum(dim=-1)

    def _build_neighbors(self, dataset: GraphMLDataset) -> dict[int, list[int]]:
        """Build an undirected neighborhood lookup."""

        neighbors: dict[int, list[int]] = defaultdict(list)
        if dataset.edge_index.numel() == 0:
            return neighbors
        source_index, target_index = dataset.edge_index
        for source, target in zip(source_index.tolist(), target_index.tolist(), strict=True):
            neighbors[int(source)].append(int(target))
        return neighbors

    def _random_walk(
        self,
        start_index: int,
        *,
        neighbors: dict[int, list[int]],
        walk_length: int,
        rng: random.Random,
    ) -> list[int]:
        """Generate a simple random walk."""

        walk = [start_index]
        current = start_index
        for _ in range(max(walk_length - 1, 0)):
            current_neighbors = neighbors.get(current)
            if not current_neighbors:
                break
            current = rng.choice(current_neighbors)
            walk.append(current)
        return walk

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        """Safely coerce an object to an integer."""

        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                return default
        return default
