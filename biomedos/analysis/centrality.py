"""Network centrality-based ranking."""

from __future__ import annotations

from pydantic import BaseModel

from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType
from biomedos.graph.statistics import GraphStatistics


class TargetRanking(BaseModel):
    """A ranked drug-targetability profile."""

    gene_id: str
    gene_name: str
    degree_centrality: float
    betweenness: float
    pagerank: float
    combined_score: float
    druggability_score: float
    targeted_by_drugs: int
    disease_count: int
    pathway_count: int


class DrugTargetRanker:
    """Rank entities by graph topology signals."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        """Initialize the target ranker."""

        self.knowledge_graph = knowledge_graph
        self.statistics = GraphStatistics(knowledge_graph)

    def rank_targets(
        self,
        target_ids: list[str] | None = None,
        *,
        top_k: int | None = None,
    ) -> list[TargetRanking]:
        """Rank targets by centrality and druggability."""

        degree_scores = self.statistics.degree_centrality()
        betweenness_scores = self.statistics.betweenness()
        pagerank_scores = self.statistics.pagerank()
        candidate_ids = target_ids or [
            node_id
            for node_id, payload in self.knowledge_graph.graph.nodes(data=True)
            if payload.get("node_type") == NodeType.GENE
        ]
        rankings: list[TargetRanking] = []
        for gene_id in candidate_ids:
            node = self.knowledge_graph.get_node(gene_id)
            if node is None or node.node_type != NodeType.GENE:
                continue
            drug_edges = self.knowledge_graph.get_edges(
                target_id=gene_id,
                edge_type=EdgeType.DRUG_TARGET,
            )
            disease_edges = self.knowledge_graph.get_edges(
                source_id=gene_id,
                edge_type=EdgeType.GENE_DISEASE,
            )
            pathway_edges = self.knowledge_graph.get_edges(
                source_id=gene_id,
                edge_type=EdgeType.GENE_PATHWAY,
            )
            combined_score = (
                0.4 * degree_scores.get(gene_id, 0.0)
                + 0.3 * betweenness_scores.get(gene_id, 0.0)
                + 0.3 * pagerank_scores.get(gene_id, 0.0)
            )
            druggability = min(
                1.0,
                combined_score
                + 0.25 * min(len(drug_edges), 3) / 3.0
                + 0.15 * min(len(pathway_edges), 5) / 5.0
                + 0.15 * min(len(disease_edges), 5) / 5.0,
            )
            rankings.append(
                TargetRanking(
                    gene_id=gene_id,
                    gene_name=node.name,
                    degree_centrality=float(degree_scores.get(gene_id, 0.0)),
                    betweenness=float(betweenness_scores.get(gene_id, 0.0)),
                    pagerank=float(pagerank_scores.get(gene_id, 0.0)),
                    combined_score=float(combined_score),
                    druggability_score=float(max(0.0, min(druggability, 1.0))),
                    targeted_by_drugs=len(drug_edges),
                    disease_count=len(disease_edges),
                    pathway_count=len(pathway_edges),
                )
            )
        rankings.sort(key=lambda item: item.combined_score, reverse=True)
        return rankings[:top_k] if top_k is not None else rankings

    def druggability_score(self, gene_id_or_name: str) -> float:
        """Return a normalized druggability score for one gene."""

        rankings = self.rank_targets([gene_id_or_name])
        if rankings:
            return rankings[0].druggability_score
        matches = self.knowledge_graph.search_nodes(
            gene_id_or_name,
            node_type=NodeType.GENE,
            limit=1,
        )
        if not matches:
            return 0.0
        resolved = self.rank_targets([matches[0].id])
        return resolved[0].druggability_score if resolved else 0.0


CentralityAnalyzer = DrugTargetRanker
