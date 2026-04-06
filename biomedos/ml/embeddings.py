"""Persistent storage for learned graph embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from biomedos.graph.schema import NodeType


class EmbeddingStore:
    """Store and query learned graph embeddings."""

    def __init__(
        self,
        embeddings: dict[str, np.ndarray] | None = None,
        metadata: dict[str, dict[str, object]] | None = None,
    ) -> None:
        """Initialize the embedding store."""

        self.embeddings = embeddings or {}
        self.metadata = metadata or {}

    def add_embeddings(
        self,
        embeddings: dict[str, np.ndarray],
        metadata: dict[str, dict[str, object]] | None = None,
    ) -> None:
        """Add embeddings to the store."""

        self.embeddings.update(embeddings)
        if metadata is not None:
            self.metadata.update(metadata)

    def save(self, path: str | Path) -> None:
        """Save embeddings and metadata to disk."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embeddings": {node_id: vector.tolist() for node_id, vector in self.embeddings.items()},
            "metadata": self.metadata,
        }
        target.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingStore:
        """Load embeddings from disk."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        embeddings_raw = payload.get("embeddings", {})
        metadata_raw = payload.get("metadata", {})
        embeddings = {
            str(node_id): np.asarray(vector, dtype=np.float32)
            for node_id, vector in embeddings_raw.items()
        }
        metadata = {
            str(node_id): dict(values) if isinstance(values, dict) else {}
            for node_id, values in metadata_raw.items()
        }
        return cls(embeddings=embeddings, metadata=metadata)

    def most_similar(
        self,
        node_id: str,
        *,
        top_k: int = 10,
        node_type: NodeType | None = None,
    ) -> list[tuple[str, float]]:
        """Return nearest neighbors for a node embedding."""

        if node_id not in self.embeddings:
            return []
        query = self.embeddings[node_id].reshape(1, -1)
        candidates: list[tuple[str, np.ndarray]] = []
        for candidate_id, vector in self.embeddings.items():
            if candidate_id == node_id:
                continue
            if node_type is not None and self._metadata_node_type(candidate_id) != node_type:
                continue
            candidates.append((candidate_id, vector))
        if not candidates:
            return []
        matrix = np.stack([vector for _, vector in candidates], axis=0)
        scores = cosine_similarity(query, matrix)[0]
        ranked = sorted(
            zip((candidate_id for candidate_id, _ in candidates), scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [(candidate_id, float(score)) for candidate_id, score in ranked[:top_k]]

    def cross_modal_search(
        self,
        node_id: str,
        *,
        target_type: NodeType,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search for similar embeddings in a different node type."""

        return self.most_similar(node_id, top_k=top_k, node_type=target_type)

    def cluster(self, n_clusters: int = 8) -> dict[int, list[str]]:
        """Cluster embeddings with KMeans."""

        if not self.embeddings:
            return {}
        node_ids = list(self.embeddings)
        matrix = np.stack([self.embeddings[node_id] for node_id in node_ids], axis=0)
        cluster_count = max(1, min(n_clusters, len(node_ids)))
        estimator = KMeans(n_clusters=cluster_count, n_init=10, random_state=42)
        labels = estimator.fit_predict(matrix)
        clusters: dict[int, list[str]] = {}
        for node_id, label in zip(node_ids, labels, strict=True):
            clusters.setdefault(int(label), []).append(node_id)
        return clusters

    def similarity_search(self, node_id: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Backward-compatible alias for nearest-neighbor retrieval."""

        return self.most_similar(node_id, top_k=top_k)

    def _metadata_node_type(self, node_id: str) -> NodeType | None:
        """Return the stored node type for a node identifier."""

        raw_node_type = self.metadata.get(node_id, {}).get("node_type")
        if isinstance(raw_node_type, NodeType):
            return raw_node_type
        if isinstance(raw_node_type, str):
            return NodeType(raw_node_type)
        return None


GraphEmbeddingStore = EmbeddingStore
