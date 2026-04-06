"""Embedding-based similarity analysis."""

from __future__ import annotations

from pydantic import BaseModel

from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import NodeType
from biomedos.ml.embeddings import EmbeddingStore


class SimilarityHit(BaseModel):
    """A similarity-search result."""

    node_id: str
    name: str
    node_type: NodeType
    score: float


class BioSimilarity:
    """Measure similarity across biomedical entities."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        embedding_store: EmbeddingStore | None = None,
    ) -> None:
        """Initialize the similarity analyzer."""

        self.knowledge_graph = knowledge_graph
        self.embedding_store = embedding_store or EmbeddingStore()

    def nearest_neighbors(self, entity_id: str, *, top_k: int = 10) -> list[tuple[str, float]]:
        """Return nearest neighboring entities."""

        if entity_id in self.embedding_store.embeddings:
            return self.embedding_store.most_similar(entity_id, top_k=top_k)
        return self._neighbor_jaccard(entity_id, top_k=top_k)

    def gene_similarity(self, entity_id: str, *, top_k: int = 10) -> list[SimilarityHit]:
        """Return similar genes for a seed entity."""

        return self._format_hits(
            entity_id,
            self.embedding_store.most_similar(entity_id, top_k=top_k, node_type=NodeType.GENE)
            if entity_id in self.embedding_store.embeddings
            else self._neighbor_jaccard(entity_id, top_k=top_k, node_type=NodeType.GENE),
        )

    def drug_similarity(self, entity_id: str, *, top_k: int = 10) -> list[SimilarityHit]:
        """Return similar drugs for a seed entity."""

        return self._format_hits(
            entity_id,
            self.embedding_store.most_similar(entity_id, top_k=top_k, node_type=NodeType.DRUG)
            if entity_id in self.embedding_store.embeddings
            else self._neighbor_jaccard(entity_id, top_k=top_k, node_type=NodeType.DRUG),
        )

    def cross_modal_search(
        self,
        entity_id: str,
        *,
        target_type: NodeType,
        top_k: int = 10,
    ) -> list[SimilarityHit]:
        """Search for similar entities across node types."""

        if entity_id in self.embedding_store.embeddings:
            hits = self.embedding_store.cross_modal_search(
                entity_id,
                target_type=target_type,
                top_k=top_k,
            )
        else:
            hits = self._neighbor_jaccard(entity_id, top_k=top_k, node_type=target_type)
        return self._format_hits(entity_id, hits)

    def _neighbor_jaccard(
        self,
        entity_id: str,
        *,
        top_k: int,
        node_type: NodeType | None = None,
    ) -> list[tuple[str, float]]:
        """Fallback topology-based similarity via neighbor overlap."""

        if entity_id not in self.knowledge_graph.graph:
            return []
        source_neighbors = set(self.knowledge_graph.graph.neighbors(entity_id))
        if not source_neighbors:
            return []
        scored: list[tuple[str, float]] = []
        for candidate_id in self.knowledge_graph.graph.nodes():
            if candidate_id == entity_id:
                continue
            node = self.knowledge_graph.get_node(str(candidate_id))
            if node is None:
                continue
            if node_type is not None and node.node_type != node_type:
                continue
            candidate_neighbors = set(self.knowledge_graph.graph.neighbors(candidate_id))
            union = source_neighbors | candidate_neighbors
            if not union:
                continue
            score = len(source_neighbors & candidate_neighbors) / len(union)
            scored.append((str(candidate_id), float(score)))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _format_hits(
        self,
        entity_id: str,
        hits: list[tuple[str, float]],
    ) -> list[SimilarityHit]:
        """Format raw similarity hits with node metadata."""

        del entity_id
        formatted: list[SimilarityHit] = []
        for node_id, score in hits:
            node = self.knowledge_graph.get_node(node_id)
            if node is None:
                continue
            formatted.append(
                SimilarityHit(
                    node_id=node.id,
                    name=node.name,
                    node_type=node.node_type,
                    score=score,
                )
            )
        return formatted


SimilarityAnalyzer = BioSimilarity
