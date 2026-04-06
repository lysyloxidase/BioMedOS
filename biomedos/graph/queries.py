"""Query engine for biomedical knowledge graph reasoning."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import networkx as nx
from pydantic import BaseModel, Field

from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import BioNode, EdgeType, NodeType


class RepurposingCandidate(BaseModel):
    """A candidate drug repurposing hypothesis."""

    drug_id: str
    drug_name: str
    score: float
    target_genes: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)


class GraphQueryEngine:
    """High-level query engine for a biomedical knowledge graph."""

    def __init__(self, knowledge_graph: KnowledgeGraph) -> None:
        """Initialize the query engine."""

        self.knowledge_graph = knowledge_graph

    def find_drug_targets(self, drug_name_or_id: str) -> list[BioNode]:
        """Return genes targeted by a drug."""

        drug = self._resolve_node(drug_name_or_id, expected_type=NodeType.DRUG)
        if drug is None:
            return []

        target_nodes: list[BioNode] = []
        for edge in self.knowledge_graph.get_edges(
            source_id=drug.id, edge_type=EdgeType.DRUG_TARGET
        ):
            node = self.knowledge_graph.get_node(edge.target_id)
            if node is not None:
                target_nodes.append(node)
        return target_nodes

    def find_repurposing_candidates(
        self,
        disease_name_or_id: str,
        *,
        min_score: float = 0.0,
    ) -> list[RepurposingCandidate]:
        """Rank drug repurposing candidates for a disease."""

        disease = self._resolve_node(disease_name_or_id, expected_type=NodeType.DISEASE)
        if disease is None:
            return []

        disease_genes = {
            edge.source_id: edge
            for edge in self.knowledge_graph.get_edges(
                target_id=disease.id,
                edge_type=EdgeType.GENE_DISEASE,
            )
        }
        known_drugs = {
            edge.source_id
            for edge in self.knowledge_graph.get_edges(
                target_id=disease.id,
                edge_type=EdgeType.DRUG_DISEASE,
            )
        }

        candidate_scores: dict[str, float] = defaultdict(float)
        candidate_targets: dict[str, list[str]] = defaultdict(list)
        rationales: dict[str, list[str]] = defaultdict(list)

        for gene_id, association in disease_genes.items():
            for edge in self.knowledge_graph.get_edges(
                target_id=gene_id,
                edge_type=EdgeType.DRUG_TARGET,
            ):
                if edge.source_id in known_drugs:
                    continue
                drug = self.knowledge_graph.get_node(edge.source_id)
                gene = self.knowledge_graph.get_node(gene_id)
                if drug is None or gene is None:
                    continue
                contribution = association.score * edge.score
                candidate_scores[drug.id] += contribution
                candidate_targets[drug.id].append(gene.name)
                rationales[drug.id].append(
                    f"{drug.name} targets {gene.name}, which is associated with {disease.name}."
                )

        candidates: list[RepurposingCandidate] = []
        for drug_id, score in candidate_scores.items():
            if score < min_score:
                continue
            drug = self.knowledge_graph.get_node(drug_id)
            if drug is None:
                continue
            candidates.append(
                RepurposingCandidate(
                    drug_id=drug.id,
                    drug_name=drug.name,
                    score=score,
                    target_genes=sorted(set(candidate_targets[drug.id])),
                    rationale=rationales[drug.id],
                )
            )
        candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates

    def gene_disease_evidence(
        self,
        gene_name_or_id: str,
        disease_name_or_id: str,
    ) -> dict[str, Any]:
        """Collect evidence for a gene-disease association."""

        gene = self._resolve_node(gene_name_or_id, expected_type=NodeType.GENE)
        disease = self._resolve_node(disease_name_or_id, expected_type=NodeType.DISEASE)
        if gene is None or disease is None:
            return {"gene": None, "disease": None, "direct_score": 0.0, "paths": []}

        direct_edges = [
            edge
            for edge in self.knowledge_graph.get_edges(
                source_id=gene.id,
                target_id=disease.id,
                edge_type=EdgeType.GENE_DISEASE,
            )
        ]
        path = self.shortest_path(gene.id, disease.id)
        return {
            "gene": gene.name,
            "disease": disease.name,
            "direct_score": max((edge.score for edge in direct_edges), default=0.0),
            "paths": [path] if path else [],
        }

    def shortest_path(self, source_id: str, target_id: str, *, max_length: int = 6) -> list[str]:
        """Find the shortest undirected path between two nodes."""

        if (
            source_id not in self.knowledge_graph.graph
            or target_id not in self.knowledge_graph.graph
        ):
            return []
        undirected = self.knowledge_graph.graph.to_undirected()
        try:
            path = nx.shortest_path(undirected, source=source_id, target=target_id)
        except nx.NetworkXNoPath:
            return []
        return path if len(path) - 1 <= max_length else []

    def subgraph(self, node_ids: list[str], *, radius: int = 1) -> KnowledgeGraph:
        """Return an ego subgraph around one or more seed nodes."""

        seeds = [node_id for node_id in node_ids if node_id in self.knowledge_graph.graph]
        if not seeds:
            return KnowledgeGraph()

        included_nodes: set[str] = set()
        undirected = self.knowledge_graph.graph.to_undirected()
        for seed in seeds:
            included_nodes.update(
                nx.single_source_shortest_path_length(undirected, seed, cutoff=radius)
            )

        subgraph = self.knowledge_graph.graph.subgraph(included_nodes).copy()
        return KnowledgeGraph(graph=nx.MultiDiGraph(subgraph))

    def _resolve_node(self, query: str, *, expected_type: NodeType) -> BioNode | None:
        """Resolve a node by identifier or fuzzy name match."""

        direct = self.knowledge_graph.get_node(query)
        if direct is not None and direct.node_type == expected_type:
            return direct

        matches = self.knowledge_graph.search_nodes(query, node_type=expected_type, limit=1)
        if matches:
            return matches[0]
        return None
