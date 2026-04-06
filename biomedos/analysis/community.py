"""Community detection utilities."""

from __future__ import annotations

from collections import Counter

import networkx as nx
from pydantic import BaseModel, Field

from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import NodeType


class CommunitySummary(BaseModel):
    """Characterization of a detected graph community."""

    community_id: int
    members: list[str] = Field(default_factory=list)
    dominant_pathways: list[str] = Field(default_factory=list)
    dominant_diseases: list[str] = Field(default_factory=list)
    dominant_drugs: list[str] = Field(default_factory=list)
    size: int = 0


class CommunityDetector:
    """Detect graph communities in the biomedical network."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        """Initialize the community detector."""

        self.knowledge_graph = knowledge_graph

    def detect(self) -> list[list[str]]:
        """Detect graph communities."""

        undirected = self.knowledge_graph.graph.to_undirected()
        if undirected.number_of_nodes() == 0:
            return []
        try:
            communities = nx.community.louvain_communities(undirected, seed=42)
        except Exception:
            communities = nx.community.greedy_modularity_communities(undirected)
        return [sorted(str(node_id) for node_id in community) for community in communities]

    def characterize_community(
        self,
        members: list[str],
        *,
        community_id: int = 0,
    ) -> CommunitySummary:
        """Summarize the dominant biomedical content of a community."""

        pathways: Counter[str] = Counter()
        diseases: Counter[str] = Counter()
        drugs: Counter[str] = Counter()
        for member_id in members:
            node = self.knowledge_graph.get_node(member_id)
            if node is None:
                continue
            if node.node_type == NodeType.PATHWAY:
                pathways[node.name] += 1
            elif node.node_type == NodeType.DISEASE:
                diseases[node.name] += 1
            elif node.node_type == NodeType.DRUG:
                drugs[node.name] += 1
        return CommunitySummary(
            community_id=community_id,
            members=members,
            dominant_pathways=[name for name, _ in pathways.most_common(5)],
            dominant_diseases=[name for name, _ in diseases.most_common(5)],
            dominant_drugs=[name for name, _ in drugs.most_common(5)],
            size=len(members),
        )
