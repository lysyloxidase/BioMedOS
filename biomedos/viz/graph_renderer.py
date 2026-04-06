"""Interactive graph rendering utilities."""

from __future__ import annotations

import math

from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType

try:
    from pyvis.network import Network
except ImportError:  # pragma: no cover - optional at test time
    Network = None


NODE_COLORS: dict[NodeType, str] = {
    NodeType.GENE: "#0f766e",
    NodeType.PROTEIN: "#1d4ed8",
    NodeType.DISEASE: "#b91c1c",
    NodeType.DRUG: "#ca8a04",
    NodeType.COMPOUND: "#7c3aed",
    NodeType.PATHWAY: "#15803d",
    NodeType.PHENOTYPE: "#c2410c",
    NodeType.CELL_TYPE: "#334155",
    NodeType.TISSUE: "#475569",
    NodeType.SIDE_EFFECT: "#db2777",
    NodeType.CLINICAL_TRIAL: "#4338ca",
    NodeType.PUBLICATION: "#0f172a",
}


class GraphRenderer:
    """Render a knowledge graph as interactive HTML."""

    def to_vis_payload(self, knowledge_graph: KnowledgeGraph) -> dict[str, object]:
        """Serialize the graph into vis-network-friendly node and edge payloads."""

        degree_map = dict(knowledge_graph.graph.degree())
        nodes: list[dict[str, object]] = []
        edges: list[dict[str, object]] = []

        for node_id, payload in knowledge_graph.graph.nodes(data=True):
            node = knowledge_graph.get_node(str(node_id))
            node_type = (
                node.node_type if node is not None else payload.get("node_type", NodeType.GENE)
            )
            degree = degree_map.get(node_id, 0)
            nodes.append(
                {
                    "id": str(node_id),
                    "label": str(payload.get("name", node_id)),
                    "node_type": getattr(node_type, "value", str(node_type)),
                    "color": NODE_COLORS.get(node_type, "#64748b")
                    if isinstance(node_type, NodeType)
                    else "#64748b",
                    "degree": degree,
                    "size": round(14.0 + 8.0 * math.log1p(degree), 2),
                    "details": node.model_dump(mode="python")
                    if node is not None
                    else dict(payload),
                }
            )

        for index, (source, target, _, payload) in enumerate(
            knowledge_graph.graph.edges(keys=True, data=True),
            start=1,
        ):
            edge_type = payload.get("edge_type", "")
            label = edge_type.value if isinstance(edge_type, EdgeType) else str(edge_type)
            edges.append(
                {
                    "id": f"edge-{index}",
                    "from": str(source),
                    "to": str(target),
                    "label": label,
                    "edge_type": label,
                    "score": float(payload.get("score", 1.0)),
                    "details": dict(payload),
                }
            )

        return {"nodes": nodes, "edges": edges}

    def render_html(self, knowledge_graph: KnowledgeGraph) -> str:
        """Render the graph to an embeddable HTML string."""

        if Network is None:
            return self._fallback_html(knowledge_graph)

        network = Network(height="720px", width="100%", bgcolor="#f8fafc", font_color="#0f172a")
        network.barnes_hut()
        payload = self.to_vis_payload(knowledge_graph)
        nodes = payload["nodes"] if isinstance(payload["nodes"], list) else []
        edges = payload["edges"] if isinstance(payload["edges"], list) else []
        for node in nodes:
            network.add_node(
                str(node["id"]),
                label=str(node["label"]),
                title=f"{node['label']} ({node['node_type']})",
                color=str(node["color"]),
                size=float(node["size"]),
            )
        for edge in edges:
            network.add_edge(
                str(edge["from"]),
                str(edge["to"]),
                label=str(edge["label"]),
                title=str(edge["label"]),
            )
        return str(network.generate_html(notebook=False))

    def _fallback_html(self, knowledge_graph: KnowledgeGraph) -> str:
        """Render a plain HTML fallback when PyVis is unavailable."""

        rows = []
        for node_id, payload in knowledge_graph.graph.nodes(data=True):
            label = payload.get("name", node_id)
            node_type = payload.get("node_type")
            rows.append(f"<li><strong>{label}</strong> ({node_type})</li>")
        return (
            "<html><body><h1>BioMedOS Graph</h1>"
            f"<p>Nodes: {knowledge_graph.graph.number_of_nodes()} | "
            f"Edges: {knowledge_graph.graph.number_of_edges()}</p>"
            f"<ul>{''.join(rows)}</ul></body></html>"
        )
