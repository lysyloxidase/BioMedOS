"""Dataset conversion utilities for graph ML."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import ceil
from types import SimpleNamespace
from typing import Any

import torch

from biomedos.config import Settings, get_settings
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType

try:
    from torch_geometric.data import HeteroData
except ImportError:  # pragma: no cover - optional at test time
    HeteroData = Any


@dataclass(slots=True)
class EdgeSplit:
    """A train/validation/test link prediction split."""

    source_index: torch.Tensor
    target_index: torch.Tensor
    relation_index: torch.Tensor
    labels: torch.Tensor
    pairs: list[tuple[str, str]]


@dataclass(slots=True)
class GraphMLDataset:
    """Prepared graph ML dataset with PyG and global views."""

    hetero_data: HeteroData
    node_ids: list[str]
    node_index: dict[str, int]
    node_types: dict[str, NodeType]
    global_x: torch.Tensor
    edge_index: torch.Tensor
    relation_index: torch.Tensor
    relation_to_id: dict[EdgeType, int]
    id_to_relation: dict[int, EdgeType]
    heldout_edge_type: EdgeType
    source_node_type: NodeType
    target_node_type: NodeType
    train_split: EdgeSplit
    val_split: EdgeSplit
    test_split: EdgeSplit

    @property
    def num_nodes(self) -> int:
        """Return the number of graph nodes."""

        return len(self.node_ids)

    @property
    def num_relations(self) -> int:
        """Return the number of edge relation types."""

        return len(self.relation_to_id)

    @property
    def feature_dim(self) -> int:
        """Return the global input feature dimension."""

        return int(self.global_x.shape[1])

    def candidate_indices_for_type(self, node_type: NodeType) -> list[int]:
        """Return node indices belonging to a specific node type."""

        return [
            index
            for node_id, index in self.node_index.items()
            if self.node_types[node_id] == node_type
        ]

    def relation_id(self, edge_type: EdgeType) -> int:
        """Return the integer relation identifier for an edge type."""

        return self.relation_to_id[edge_type]


class MiniHeteroData(dict[str, SimpleNamespace]):
    """Minimal fallback when torch-geometric is unavailable."""

    def __getitem__(self, key: str) -> SimpleNamespace:
        """Return a lazily created namespace for one node or edge type."""

        if key not in self:
            self[key] = SimpleNamespace()
        return dict.__getitem__(self, key)


class GraphDatasetBuilder:
    """Convert a NetworkX graph into graph ML datasets."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize the dataset builder."""

        self.settings = settings or get_settings()

    def build_from_kg(
        self,
        knowledge_graph: KnowledgeGraph,
        *,
        edge_type: EdgeType | None = None,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        negative_ratio: float = 1.0,
        seed: int | None = None,
    ) -> GraphMLDataset:
        """Build a graph ML dataset from a knowledge graph."""

        rng = random.Random(self.settings.RANDOM_SEED if seed is None else seed)
        node_ids = list(knowledge_graph.graph.nodes())
        if not node_ids:
            msg = "Knowledge graph is empty and cannot be converted into a dataset."
            raise ValueError(msg)

        node_index = {node_id: index for index, node_id in enumerate(node_ids)}
        node_types = self._build_node_type_lookup(knowledge_graph, node_ids)
        node_type_order = list(NodeType)

        resolved_edge_type = edge_type or self._default_edge_type(knowledge_graph)
        positive_edges = self._collect_positive_edges(
            knowledge_graph,
            node_index,
            edge_type=resolved_edge_type,
        )
        if not positive_edges:
            msg = f"No positive edges found for relation type: {resolved_edge_type.value}"
            raise ValueError(msg)

        source_node_type = positive_edges[0][2]
        target_node_type = positive_edges[0][3]
        source_candidates = [
            node_index[node_id] for node_id in node_ids if node_types[node_id] == source_node_type
        ]
        target_candidates = [
            node_index[node_id] for node_id in node_ids if node_types[node_id] == target_node_type
        ]
        if not source_candidates or not target_candidates:
            msg = "Positive edge types do not have matching source/target candidates."
            raise ValueError(msg)

        relation_to_id = self._build_relation_mapping(knowledge_graph, resolved_edge_type)
        global_x = self._build_global_features(
            knowledge_graph,
            node_ids,
            node_types,
            node_type_order,
        )
        hetero_data = self._build_hetero_data(
            knowledge_graph,
            node_ids,
            node_index,
            node_types,
            global_x,
        )

        shuffled_positive_edges = positive_edges[:]
        rng.shuffle(shuffled_positive_edges)
        train_positive, val_positive, test_positive = self._split_positive_edges(
            shuffled_positive_edges,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        all_positive_pairs = {
            (source_idx, target_idx) for source_idx, target_idx, _, _, _, _ in positive_edges
        }

        train_split = self._create_edge_split(
            train_positive,
            relation_id=relation_to_id[resolved_edge_type],
            negative_ratio=negative_ratio,
            source_candidates=source_candidates,
            target_candidates=target_candidates,
            existing_positive_pairs=all_positive_pairs,
            rng=rng,
        )
        val_split = self._create_edge_split(
            val_positive,
            relation_id=relation_to_id[resolved_edge_type],
            negative_ratio=negative_ratio,
            source_candidates=source_candidates,
            target_candidates=target_candidates,
            existing_positive_pairs=all_positive_pairs,
            rng=rng,
        )
        test_split = self._create_edge_split(
            test_positive,
            relation_id=relation_to_id[resolved_edge_type],
            negative_ratio=negative_ratio,
            source_candidates=source_candidates,
            target_candidates=target_candidates,
            existing_positive_pairs=all_positive_pairs,
            rng=rng,
        )

        held_out_pairs = {
            (source_idx, target_idx)
            for source_idx, target_idx, _, _, _, _ in [*val_positive, *test_positive]
        }
        edge_index, relation_index = self._build_message_graph(
            knowledge_graph,
            node_index,
            relation_to_id=relation_to_id,
            edge_type=resolved_edge_type,
            held_out_pairs=held_out_pairs,
        )
        id_to_relation = {index: relation for relation, index in relation_to_id.items()}

        return GraphMLDataset(
            hetero_data=hetero_data,
            node_ids=node_ids,
            node_index=node_index,
            node_types=node_types,
            global_x=global_x,
            edge_index=edge_index,
            relation_index=relation_index,
            relation_to_id=relation_to_id,
            id_to_relation=id_to_relation,
            heldout_edge_type=resolved_edge_type,
            source_node_type=source_node_type,
            target_node_type=target_node_type,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
        )

    def _build_node_type_lookup(
        self,
        knowledge_graph: KnowledgeGraph,
        node_ids: list[str],
    ) -> dict[str, NodeType]:
        """Build a node-to-type lookup table."""

        node_types: dict[str, NodeType] = {}
        for node_id in node_ids:
            node = knowledge_graph.get_node(node_id)
            if node is None:
                msg = f"Node disappeared during dataset creation: {node_id}"
                raise ValueError(msg)
            node_types[node_id] = node.node_type
        return node_types

    def _default_edge_type(self, knowledge_graph: KnowledgeGraph) -> EdgeType:
        """Choose a default target relation for link prediction."""

        counts: dict[EdgeType, int] = {}
        for _, _, _, payload in knowledge_graph.graph.edges(keys=True, data=True):
            raw_edge_type = payload.get("edge_type")
            if isinstance(raw_edge_type, EdgeType):
                counts[raw_edge_type] = counts.get(raw_edge_type, 0) + 1
        if not counts:
            msg = "Knowledge graph does not contain typed edges."
            raise ValueError(msg)
        return max(counts, key=lambda relation: counts[relation])

    def _collect_positive_edges(
        self,
        knowledge_graph: KnowledgeGraph,
        node_index: dict[str, int],
        *,
        edge_type: EdgeType,
    ) -> list[tuple[int, int, NodeType, NodeType, str, str]]:
        """Collect positive edges for the requested relation."""

        positives: list[tuple[int, int, NodeType, NodeType, str, str]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for source_id, target_id, _, payload in knowledge_graph.graph.edges(keys=True, data=True):
            raw_edge_type = payload.get("edge_type")
            if raw_edge_type != edge_type:
                continue
            source_node = knowledge_graph.get_node(str(source_id))
            target_node = knowledge_graph.get_node(str(target_id))
            if source_node is None or target_node is None:
                continue
            source_idx = node_index[str(source_id)]
            target_idx = node_index[str(target_id)]
            if (source_idx, target_idx) in seen_pairs:
                continue
            seen_pairs.add((source_idx, target_idx))
            positives.append(
                (
                    source_idx,
                    target_idx,
                    source_node.node_type,
                    target_node.node_type,
                    str(source_id),
                    str(target_id),
                )
            )
        return positives

    def _build_global_features(
        self,
        knowledge_graph: KnowledgeGraph,
        node_ids: list[str],
        node_types: dict[str, NodeType],
        node_type_order: list[NodeType],
    ) -> torch.Tensor:
        """Build simple global node features."""

        degree_scores = dict(knowledge_graph.graph.degree())
        pagerank_scores = self._safe_pagerank(knowledge_graph)
        max_degree = max((float(degree) for degree in degree_scores.values()), default=1.0)

        features: list[list[float]] = []
        for node_id in node_ids:
            node_type = node_types[node_id]
            one_hot = [
                1.0 if current_type == node_type else 0.0 for current_type in node_type_order
            ]
            degree_feature = float(degree_scores.get(node_id, 0.0)) / max(max_degree, 1.0)
            pagerank_feature = float(pagerank_scores.get(node_id, 0.0))
            features.append([*one_hot, degree_feature, pagerank_feature])
        return torch.tensor(features, dtype=torch.float32)

    def _build_hetero_data(
        self,
        knowledge_graph: KnowledgeGraph,
        node_ids: list[str],
        node_index: dict[str, int],
        node_types: dict[str, NodeType],
        global_x: torch.Tensor,
    ) -> HeteroData:
        """Build a HeteroData object with local features."""

        try:
            hetero_data = knowledge_graph.to_pyg_heterodata()
        except RuntimeError:
            hetero_data = MiniHeteroData()
        nodes_by_type: dict[NodeType, list[str]] = {}
        for node_id in node_ids:
            nodes_by_type.setdefault(node_types[node_id], []).append(node_id)

        for node_type, typed_ids in nodes_by_type.items():
            indices = [node_index[node_id] for node_id in typed_ids]
            hetero_data[node_type.value].x = global_x[indices]
            hetero_data[node_type.value].global_id = torch.tensor(indices, dtype=torch.long)
        return hetero_data

    def _split_positive_edges(
        self,
        positive_edges: list[tuple[int, int, NodeType, NodeType, str, str]],
        *,
        val_ratio: float,
        test_ratio: float,
    ) -> tuple[
        list[tuple[int, int, NodeType, NodeType, str, str]],
        list[tuple[int, int, NodeType, NodeType, str, str]],
        list[tuple[int, int, NodeType, NodeType, str, str]],
    ]:
        """Split positive edges into train, validation, and test splits."""

        total = len(positive_edges)
        if total == 1:
            return positive_edges, [], []

        val_count = int(total * val_ratio)
        test_count = int(total * test_ratio)
        if total >= 3:
            val_count = max(val_count, 1)
            test_count = max(test_count, 1)
        if val_count + test_count >= total:
            overflow = val_count + test_count - total + 1
            while overflow > 0 and test_count > 0:
                test_count -= 1
                overflow -= 1
            while overflow > 0 and val_count > 0:
                val_count -= 1
                overflow -= 1

        train_count = max(total - val_count - test_count, 1)
        val_start = train_count
        test_start = train_count + val_count
        return (
            positive_edges[:train_count],
            positive_edges[val_start:test_start],
            positive_edges[test_start:],
        )

    def _create_edge_split(
        self,
        positive_edges: list[tuple[int, int, NodeType, NodeType, str, str]],
        *,
        relation_id: int,
        negative_ratio: float,
        source_candidates: list[int],
        target_candidates: list[int],
        existing_positive_pairs: set[tuple[int, int]],
        rng: random.Random,
    ) -> EdgeSplit:
        """Create a binary classification split with negative examples."""

        if not positive_edges:
            empty_tensor = torch.empty(0, dtype=torch.long)
            empty_labels = torch.empty(0, dtype=torch.float32)
            return EdgeSplit(
                source_index=empty_tensor,
                target_index=empty_tensor,
                relation_index=empty_tensor,
                labels=empty_labels,
                pairs=[],
            )

        positive_samples = [
            (source_idx, target_idx, relation_id, 1.0, source_id, target_id)
            for source_idx, target_idx, _, _, source_id, target_id in positive_edges
        ]
        negative_count = max(1, ceil(len(positive_edges) * negative_ratio))
        negative_samples = self._sample_negative_edges(
            relation_id=relation_id,
            count=negative_count,
            source_candidates=source_candidates,
            target_candidates=target_candidates,
            existing_positive_pairs=existing_positive_pairs,
            rng=rng,
        )
        combined = [*positive_samples, *negative_samples]
        rng.shuffle(combined)

        return EdgeSplit(
            source_index=torch.tensor([item[0] for item in combined], dtype=torch.long),
            target_index=torch.tensor([item[1] for item in combined], dtype=torch.long),
            relation_index=torch.tensor([item[2] for item in combined], dtype=torch.long),
            labels=torch.tensor([item[3] for item in combined], dtype=torch.float32),
            pairs=[(item[4], item[5]) for item in combined],
        )

    def _sample_negative_edges(
        self,
        *,
        relation_id: int,
        count: int,
        source_candidates: list[int],
        target_candidates: list[int],
        existing_positive_pairs: set[tuple[int, int]],
        rng: random.Random,
    ) -> list[tuple[int, int, int, float, str, str]]:
        """Sample negative edges that do not exist in the graph."""

        sampled: list[tuple[int, int, int, float, str, str]] = []
        seen = set(existing_positive_pairs)
        max_attempts = max(count * 20, 100)
        attempts = 0

        while len(sampled) < count and attempts < max_attempts:
            attempts += 1
            source_idx = rng.choice(source_candidates)
            target_idx = rng.choice(target_candidates)
            if (source_idx, target_idx) in seen:
                continue
            seen.add((source_idx, target_idx))
            sampled.append(
                (
                    source_idx,
                    target_idx,
                    relation_id,
                    0.0,
                    str(source_idx),
                    str(target_idx),
                )
            )
        return sampled

    def _build_relation_mapping(
        self,
        knowledge_graph: KnowledgeGraph,
        heldout_edge_type: EdgeType,
    ) -> dict[EdgeType, int]:
        """Build a stable relation-to-id mapping with the held-out relation first."""

        relation_to_id = {heldout_edge_type: 0}
        for _, _, _, payload in knowledge_graph.graph.edges(keys=True, data=True):
            raw_edge_type = payload.get("edge_type")
            if isinstance(raw_edge_type, EdgeType) and raw_edge_type not in relation_to_id:
                relation_to_id[raw_edge_type] = len(relation_to_id)
        return relation_to_id

    def _build_message_graph(
        self,
        knowledge_graph: KnowledgeGraph,
        node_index: dict[str, int],
        *,
        relation_to_id: dict[EdgeType, int],
        edge_type: EdgeType,
        held_out_pairs: set[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build the message-passing graph with held-out edges removed."""

        edge_pairs: list[tuple[int, int]] = []
        edge_relations: list[int] = []

        for source_id, target_id, _, payload in knowledge_graph.graph.edges(keys=True, data=True):
            raw_edge_type = payload.get("edge_type")
            if not isinstance(raw_edge_type, EdgeType):
                continue
            source_idx = node_index[str(source_id)]
            target_idx = node_index[str(target_id)]
            if raw_edge_type == edge_type and (source_idx, target_idx) in held_out_pairs:
                continue
            relation_id = relation_to_id[raw_edge_type]
            edge_pairs.append((source_idx, target_idx))
            edge_relations.append(relation_id)
            edge_pairs.append((target_idx, source_idx))
            edge_relations.append(relation_id)

        if not edge_pairs:
            edge_pairs.append((0, 0))
            edge_relations.append(0)

        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        relation_index = torch.tensor(edge_relations, dtype=torch.long)
        return edge_index, relation_index

    def _safe_pagerank(self, knowledge_graph: KnowledgeGraph) -> dict[str, float]:
        """Safely compute PageRank scores."""

        try:
            import networkx as nx

            return dict(nx.pagerank(knowledge_graph.graph))
        except Exception:
            return {str(node_id): 0.0 for node_id in knowledge_graph.graph.nodes()}
