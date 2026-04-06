"""Hybrid retrieval for biomedical RAG."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field

from biomedos.config import Settings, get_settings
from biomedos.core.vector_store import ChromaVectorStore, SearchResult
from biomedos.graph.builder import KnowledgeGraph
from biomedos.rag.kg_context import KGContextExtractor
from biomedos.rag.reranker import CrossEncoderReranker


class RetrievedDocument(BaseModel):
    """A retrieved evidence document used for grounded generation."""

    pmid: str | None = None
    title: str
    content: str
    score: float
    source: str
    metadata: dict[str, object] = Field(default_factory=dict)
    kg_context: str | None = None


class HybridRetriever:
    """Hybrid retrieval combining sparse search, dense search, and KG context."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        *,
        knowledge_graph: KnowledgeGraph | None = None,
        kg_context_extractor: KGContextExtractor | None = None,
        reranker: CrossEncoderReranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            vector_store: Local vector store containing indexed documents.
            knowledge_graph: Optional knowledge graph for KG-RAG enrichment.
            kg_context_extractor: Optional prebuilt KG context extractor.
            reranker: Optional cross-encoder reranker.
            settings: Optional application settings.
        """

        self.vector_store = vector_store
        self.settings = settings or get_settings()
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        self.kg_context_extractor = kg_context_extractor or KGContextExtractor(self.knowledge_graph)
        self.reranker = reranker or CrossEncoderReranker(fast_mode=self.settings.FAST_LOCAL_MODE)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
        include_kg: bool = True,
    ) -> list[RetrievedDocument]:
        """Retrieve the best evidence for a query.

        Args:
            query: User question or search statement.
            top_k: Number of documents to return.
            include_kg: Whether to inject knowledge graph context.

        Returns:
            Ranked retrieved documents.
        """

        candidate_count = max(top_k * 3, top_k)
        bm25_results = self._bm25_search(query, top_k=candidate_count)
        dense_results = self._dense_search(query, top_k=candidate_count)
        fused = self._reciprocal_rank_fusion(bm25_results, dense_results)

        kg_context_text: str | None = None
        if include_kg:
            kg_context = await self.kg_context_extractor.extract_context(
                query,
                max_triples=self.settings.KG_CONTEXT_MAX_TRIPLES,
                depth=self.settings.KG_CONTEXT_DEPTH,
            )
            kg_context_text = kg_context.context_text or None

        documents = [
            self._search_result_to_document(
                result,
                score=score,
                kg_context=kg_context_text,
            )
            for result, score in fused
        ]

        if not documents and kg_context_text is not None:
            return [
                RetrievedDocument(
                    title="Knowledge graph context",
                    content=kg_context_text,
                    score=1.0,
                    source="kg_context",
                    metadata={},
                    kg_context=kg_context_text,
                )
            ]

        reranked = self.reranker.rerank(
            query,
            documents,
            top_k=min(len(documents), top_k),
        )
        return reranked[:top_k]

    async def retrieve_for_claim(self, claim: str) -> list[RetrievedDocument]:
        """Retrieve evidence specifically to verify a factual claim."""

        return await self.retrieve(
            claim,
            top_k=self.settings.RAG_RERANK_TOP_K,
            include_kg=True,
        )

    def _bm25_search(self, query: str, *, top_k: int) -> list[SearchResult]:
        """Run sparse retrieval against the backing vector store."""

        if hasattr(self.vector_store, "bm25_search"):
            return self.vector_store.bm25_search(query, top_k=top_k)
        return self.vector_store._bm25_search(query, top_k=top_k)

    def _dense_search(self, query: str, *, top_k: int) -> list[SearchResult]:
        """Run dense retrieval against the backing vector store."""

        if hasattr(self.vector_store, "dense_search"):
            return self.vector_store.dense_search(query, top_k=top_k)
        return self.vector_store.search(query, top_k=top_k)

    def _reciprocal_rank_fusion(
        self,
        bm25_results: list[SearchResult],
        dense_results: list[SearchResult],
        *,
        fusion_k: int = 60,
    ) -> list[tuple[SearchResult, float]]:
        """Merge sparse and dense search results using reciprocal rank fusion."""

        fused_scores: defaultdict[str, float] = defaultdict(float)
        payloads: dict[str, SearchResult] = {}

        for rank, result in enumerate(bm25_results, start=1):
            fused_scores[result.id] += 1.0 / (fusion_k + rank)
            payloads.setdefault(result.id, result)

        for rank, result in enumerate(dense_results, start=1):
            fused_scores[result.id] += 1.0 / (fusion_k + rank)
            payloads.setdefault(result.id, result)

        ranked_ids = sorted(
            fused_scores,
            key=lambda document_id: fused_scores[document_id],
            reverse=True,
        )
        return [(payloads[document_id], fused_scores[document_id]) for document_id in ranked_ids]

    def _search_result_to_document(
        self,
        result: SearchResult,
        *,
        score: float,
        kg_context: str | None,
    ) -> RetrievedDocument:
        """Convert a vector-store search result into a retrieved document."""

        metadata = dict(result.metadata)
        title = str(metadata.get("title", metadata.get("name", result.id)))
        pmid_value = metadata.get("pmid")
        source_value = metadata.get("source", "pubmed")
        return RetrievedDocument(
            pmid=str(pmid_value) if pmid_value is not None else None,
            title=title,
            content=result.text,
            score=score,
            source=str(source_value),
            metadata=metadata,
            kg_context=kg_context,
        )
