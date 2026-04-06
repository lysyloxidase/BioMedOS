"""Knowledge graph storage and persistence."""

from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx

from biomedos.graph.schema import (
    BioEdge,
    BioNode,
    EdgeType,
    NodeType,
    create_edge_model,
    create_node_model,
)

try:
    import torch
    from torch_geometric.data import HeteroData
except ImportError:  # pragma: no cover - optional at test time
    torch = None
    HeteroData = None


class KnowledgeGraph:
    """NetworkX-backed biomedical knowledge graph."""

    def __init__(self, graph: nx.MultiDiGraph | None = None) -> None:
        """Initialize an empty or pre-populated graph."""

        self.graph = graph or nx.MultiDiGraph()

    def add_node(self, node: BioNode) -> None:
        """Add a node to the graph."""

        self.graph.add_node(node.id, **node.model_dump(mode="python", exclude_none=True))

    def merge_node(self, node: BioNode) -> None:
        """Merge a node with an existing record."""

        if node.id not in self.graph:
            self.add_node(node)
            return

        merged = dict(self.graph.nodes[node.id])
        new_payload = node.model_dump(mode="python", exclude_none=True)
        for key, value in new_payload.items():
            if key == "properties":
                existing = merged.get("properties", {})
                if isinstance(existing, dict) and isinstance(value, dict):
                    merged[key] = {**existing, **value}
                else:
                    merged[key] = value
            elif key == "sources":
                existing_sources = merged.get("sources", [])
                if isinstance(existing_sources, list):
                    merged[key] = sorted({*existing_sources, *value})
                else:
                    merged[key] = value
            elif value not in ("", None, [], {}):
                merged[key] = value
        self.graph.add_node(node.id, **merged)

    def get_node(self, node_id: str) -> BioNode | None:
        """Fetch a validated node by identifier."""

        if node_id not in self.graph:
            return None
        payload = dict(self.graph.nodes[node_id])
        return create_node_model(payload)

    def add_edge(self, edge: BioEdge) -> None:
        """Add an edge to the graph."""

        self.graph.add_edge(
            edge.source_id,
            edge.target_id,
            key=edge.edge_type.value,
            **edge.model_dump(mode="python", exclude_none=True),
        )

    def merge_edge(self, edge: BioEdge) -> None:
        """Merge an edge with an existing same-typed relationship."""

        key = edge.edge_type.value
        if self.graph.has_edge(edge.source_id, edge.target_id, key=key):
            current = dict(self.graph[edge.source_id][edge.target_id][key])
            current_sources = current.get("sources", [])
            if isinstance(current_sources, list):
                current["sources"] = sorted({*current_sources, *edge.sources})
            else:
                current["sources"] = edge.sources
            current_properties = current.get("properties", {})
            if isinstance(current_properties, dict):
                current["properties"] = {**current_properties, **edge.properties}
            else:
                current["properties"] = edge.properties
            current["score"] = max(float(current.get("score", 0.0)), edge.score)
            self.graph.add_edge(edge.source_id, edge.target_id, key=key, **current)
            return
        self.add_edge(edge)

    def get_edges(
        self,
        *,
        source_id: str | None = None,
        target_id: str | None = None,
        edge_type: EdgeType | None = None,
    ) -> list[BioEdge]:
        """Return matching edges from the graph."""

        edges: list[BioEdge] = []
        for current_source, current_target, key, payload in self.graph.edges(keys=True, data=True):
            if source_id is not None and current_source != source_id:
                continue
            if target_id is not None and current_target != target_id:
                continue
            if edge_type is not None and key != edge_type.value:
                continue
            edges.append(create_edge_model(dict(payload)))
        return edges

    def search_nodes(
        self,
        query: str,
        *,
        node_type: NodeType | None = None,
        limit: int = 10,
    ) -> list[BioNode]:
        """Search nodes by name, identifier, or stringified properties."""

        query_normalized = query.lower().strip()
        results: list[BioNode] = []
        for node_id, payload in self.graph.nodes(data=True):
            current_type = payload.get("node_type")
            if node_type is not None and current_type != node_type:
                continue
            haystack = " ".join(
                [
                    str(node_id),
                    str(payload.get("name", "")),
                    str(payload.get("properties", "")),
                ]
            ).lower()
            if query_normalized in haystack:
                results.append(create_node_model(dict(payload)))
        results.sort(key=lambda item: (item.name.lower() != query_normalized, item.name.lower()))
        return results[:limit]

    def to_pyg_heterodata(self) -> HeteroData:
        """Convert the graph into a PyTorch Geometric HeteroData object."""

        if HeteroData is None or torch is None:
            msg = "torch and torch-geometric must be installed to export HeteroData."
            raise RuntimeError(msg)

        hetero_data = HeteroData()
        nodes_by_type: dict[NodeType, list[str]] = defaultdict(list)
        for node_id, payload in self.graph.nodes(data=True):
            node_type = payload.get("node_type")
            if isinstance(node_type, NodeType):
                nodes_by_type[node_type].append(str(node_id))

        node_index: dict[str, tuple[NodeType, int]] = {}
        for node_type, node_ids in nodes_by_type.items():
            hetero_data[node_type.value].node_id = torch.arange(len(node_ids), dtype=torch.long)
            hetero_data[node_type.value].num_nodes = len(node_ids)
            for index, node_id in enumerate(node_ids):
                node_index[node_id] = (node_type, index)

        edge_buckets: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
        for source, target, _, payload in self.graph.edges(keys=True, data=True):
            source_info = node_index.get(str(source))
            target_info = node_index.get(str(target))
            edge_type = payload.get("edge_type")
            if source_info is None or target_info is None or not isinstance(edge_type, EdgeType):
                continue
            relation = (source_info[0].value, edge_type.value, target_info[0].value)
            edge_buckets[relation].append((source_info[1], target_info[1]))

        for relation, pairs in edge_buckets.items():
            edge_index = torch.tensor(pairs, dtype=torch.long).t().contiguous()
            hetero_data[relation].edge_index = edge_index

        return hetero_data

    def stats(self) -> dict[str, object]:
        """Compute high-level graph summary statistics."""

        node_counts = Counter(
            payload.get("node_type").value
            for _, payload in self.graph.nodes(data=True)
            if isinstance(payload.get("node_type"), NodeType)
        )
        edge_counts = Counter(
            payload.get("edge_type").value
            for _, _, _, payload in self.graph.edges(keys=True, data=True)
            if isinstance(payload.get("edge_type"), EdgeType)
        )
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": dict(node_counts),
            "edge_types": dict(edge_counts),
        }

    def save(self, path: str | Path) -> None:
        """Persist the graph to disk using pickle."""

        target_path = Path(path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as file_pointer:
            pickle.dump(self.graph, file_pointer)

    @classmethod
    def load(cls, path: str | Path) -> KnowledgeGraph:
        """Load a persisted graph from disk."""

        with Path(path).open("rb") as file_pointer:
            graph = pickle.load(file_pointer)
        if not isinstance(graph, nx.MultiDiGraph):
            msg = "Persisted graph payload is not a NetworkX MultiDiGraph."
            raise TypeError(msg)
        return cls(graph=graph)
