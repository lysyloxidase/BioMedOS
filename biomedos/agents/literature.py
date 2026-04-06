"""Literature review agent."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from biomedos.agents.base import BaseAgent
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.data.pubmed import PubMedArticle, PubMedClient
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.entity_extractor import BioEntityExtractor
from biomedos.orchestration.state import AgentResult, Task, WorkflowState
from biomedos.rag.citation_tracker import CitationTracker
from biomedos.rag.pubmed_indexer import PubMedIndexer
from biomedos.rag.retriever import HybridRetriever, RetrievedDocument


class LiteratureAgent(BaseAgent):
    """Search, retrieve, and synthesize biomedical literature."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
        pubmed_client: PubMedClient | None = None,
        indexer: PubMedIndexer | None = None,
        retriever: HybridRetriever | None = None,
        citation_tracker: CitationTracker | None = None,
        entity_extractor: BioEntityExtractor | None = None,
    ) -> None:
        """Initialize the literature agent."""

        resolved_settings = settings
        resolved_vector_store = vector_store or ChromaVectorStore()
        resolved_graph = knowledge_graph or KnowledgeGraph()
        super().__init__(
            name="literature",
            llm_client=llm_client,
            knowledge_graph=resolved_graph,
            vector_store=resolved_vector_store,
            settings=resolved_settings,
        )
        self.pubmed_client = pubmed_client or PubMedClient()
        self.indexer = indexer or PubMedIndexer(self.pubmed_client, resolved_vector_store)
        self.retriever = retriever or HybridRetriever(
            resolved_vector_store,
            knowledge_graph=resolved_graph,
            settings=self.settings,
        )
        self.citation_tracker = citation_tracker or CitationTracker()
        self.entity_extractor = entity_extractor or self._build_entity_extractor(llm_client)

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the literature synthesis agent."""

        del state
        query = str(task.payload.get("query", task.description))
        search_queries = await self._extract_search_queries(query)
        articles: list[PubMedArticle] = []
        if not self.should_use_fast_path():
            pmid_candidates: list[str] = []
            for search_query in search_queries:
                pmids = await self.pubmed_client.search(search_query, max_results=10)
                for pmid in pmids:
                    if pmid not in pmid_candidates:
                        pmid_candidates.append(pmid)
                if len(pmid_candidates) >= 30:
                    break

            articles = await self.pubmed_client.fetch_abstracts(pmid_candidates[:30])
            self.indexer.index_articles(articles)

        retrieved_documents = await self.retriever.retrieve(
            query,
            top_k=self.settings.RAG_TOP_K,
            include_kg=True,
        )
        selected_documents = retrieved_documents[: self.settings.RAG_RERANK_TOP_K]
        references = self._resolve_references(articles, selected_documents)

        raw_answer = await self._generate_grounded_answer(query, selected_documents)
        available_pmids = {article.pmid for article in references}
        invalid_citations = self.citation_tracker.verify_citations(raw_answer, available_pmids)
        if invalid_citations or not self.citation_tracker.extract_cited_pmids(raw_answer):
            raw_answer = self._fallback_answer(query, selected_documents)
            invalid_citations = self.citation_tracker.verify_citations(raw_answer, available_pmids)

        cited_text, bibliography = self.citation_tracker.format_citations(raw_answer, references)
        extracted_nodes: list[str] = []
        if self.entity_extractor is not None:
            try:
                extraction = await self.entity_extractor.extract_triples(cited_text)
            except ValueError:
                extraction = None
            if extraction is not None:
                self.update_kg(nodes=extraction.entities, edges=extraction.relations)
                extracted_nodes = [node.id for node in extraction.entities]

        cited_pmids = sorted(self.citation_tracker.extract_cited_pmids(raw_answer))
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=cited_text,
            output={
                "content": cited_text,
                "bibliography": bibliography,
                "references": [article.model_dump(mode="python") for article in references],
                "search_queries": search_queries,
                "retrieved_documents": [
                    document.model_dump(mode="python") for document in selected_documents
                ],
                "invalid_citations": invalid_citations,
            },
            citations=cited_pmids,
            confidence=self._confidence(selected_documents, invalid_citations),
            node_ids=extracted_nodes,
            errors=self._build_errors(invalid_citations),
        )

    def _build_entity_extractor(
        self,
        llm_client: OllamaClient | None,
    ) -> BioEntityExtractor | None:
        """Create an optional entity extractor."""

        if llm_client is None:
            return None
        return BioEntityExtractor(llm_client, settings=self.settings)

    async def _extract_search_queries(self, query: str) -> list[str]:
        """Extract focused PubMed queries from a user request."""

        if self.llm_client is None or self.should_use_fast_path():
            return self._fallback_queries(query)

        prompt = (
            "Generate up to 3 PubMed search queries for the user request below.\n"
            'Return JSON with the schema {"queries": ["..."]}.\n'
            f"Request: {query}"
        )
        try:
            response = await self.llm_client.generate(
                prompt,
                model=self.settings.MODEL_ROUTER,
                system="Return only valid JSON.",
            )
            payload = self._parse_json(response)
        except ValueError:
            return self._fallback_queries(query)

        queries = payload.get("queries", [])
        if not isinstance(queries, list):
            return self._fallback_queries(query)

        cleaned = [str(item).strip() for item in queries if str(item).strip()]
        return cleaned[:3] if cleaned else self._fallback_queries(query)

    async def _generate_grounded_answer(
        self,
        query: str,
        documents: list[RetrievedDocument],
    ) -> str:
        """Generate an answer grounded in retrieved PubMed evidence."""

        if self.llm_client is None or not documents or self.should_use_fast_path():
            return self._fallback_answer(query, documents)

        evidence_blocks = []
        for index, document in enumerate(documents, start=1):
            pmid = document.pmid or "unknown"
            kg_text = f"\nKG context:\n{document.kg_context}" if document.kg_context else ""
            evidence_blocks.append(
                f"Document {index}\n"
                f"PMID: {pmid}\n"
                f"Title: {document.title}\n"
                f"Source: {document.source}\n"
                f"Content:\n{document.content}{kg_text}"
            )
        evidence_text = "\n\n".join(evidence_blocks)

        prompt = (
            "Answer the biomedical question using ONLY the evidence below.\n"
            "Cite every factual claim using PMID markers in the exact form [PMID:12345].\n"
            "If the evidence is insufficient, say so explicitly.\n\n"
            f"Question: {query}\n\n"
            f"Evidence:\n{evidence_text}"
        )
        response = await self.llm_client.generate(
            prompt,
            model=self.settings.MODEL_REASONER,
            system=(
                "Use only the provided abstracts. Do not invent citations or unsupported claims."
            ),
        )
        return response.strip()

    def _resolve_references(
        self,
        fetched_articles: list[PubMedArticle],
        documents: list[RetrievedDocument],
    ) -> list[PubMedArticle]:
        """Resolve PubMed references for retrieved documents."""

        fetched_by_pmid = {article.pmid: article for article in fetched_articles}
        references: list[PubMedArticle] = []
        for document in documents:
            if document.pmid is None:
                continue
            article = (
                fetched_by_pmid.get(document.pmid)
                or self.indexer.get_article(document.pmid)
                or self._article_from_document(document)
            )
            if article is not None and article.pmid not in {ref.pmid for ref in references}:
                references.append(article)
        return references

    def _fallback_answer(self, query: str, documents: list[RetrievedDocument]) -> str:
        """Create a deterministic evidence-grounded answer when the LLM is unavailable."""

        if not documents:
            return f"I do not know based on the available abstracts for: {query}"

        lines = [f"Evidence-grounded summary for: {query}"]
        for document in documents[:3]:
            citation = f"[PMID:{document.pmid}]" if document.pmid is not None else ""
            snippet = self._first_sentence(document.content)
            lines.append(f"- {document.title}: {snippet} {citation}".strip())
        return "\n".join(lines)

    def _fallback_queries(self, query: str) -> list[str]:
        """Build fallback search queries without LLM assistance."""

        quoted_terms = re.findall(r'"([^"]+)"', query)
        if quoted_terms:
            return [query, *quoted_terms][:3]

        tokens = re.findall(r"[A-Za-z0-9_-]{4,}", query)
        if not tokens:
            return [query]

        compact = " ".join(tokens[:5])
        return [query, compact][:2]

    def _confidence(
        self,
        documents: list[RetrievedDocument],
        invalid_citations: list[str],
    ) -> float:
        """Estimate confidence from evidence coverage and citation validity."""

        if not documents:
            return 0.1
        coverage = min(len(documents) / max(self.settings.RAG_RERANK_TOP_K, 1), 1.0)
        penalty = 0.2 if invalid_citations else 0.0
        return max(0.0, min(0.95, 0.55 + 0.35 * coverage - penalty))

    @staticmethod
    def _build_errors(invalid_citations: list[str]) -> list[str]:
        """Build structured error messages."""

        if not invalid_citations:
            return []
        return [f"Invalid citations: {', '.join(invalid_citations)}"]

    @staticmethod
    def _first_sentence(text: str) -> str:
        """Return the first sentence-like snippet from a document."""

        match = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)
        return match[0] if match and match[0] else text.strip()

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse the first JSON object from a response."""

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)

        msg = "Failed to parse JSON payload from literature agent response."
        raise ValueError(msg)

    @staticmethod
    def _article_from_document(document: RetrievedDocument) -> PubMedArticle | None:
        """Reconstruct a PubMedArticle from retrieved-document metadata when possible."""

        if document.pmid is None:
            return None
        metadata = document.metadata
        raw_authors = metadata.get("authors", [])
        if isinstance(raw_authors, str):
            authors = [item.strip() for item in raw_authors.split(";") if item.strip()]
        elif isinstance(raw_authors, list):
            authors = [str(item).strip() for item in raw_authors if str(item).strip()]
        else:
            authors = []
        raw_year = metadata.get("year")
        if isinstance(raw_year, int):
            year = raw_year
        elif isinstance(raw_year, str) and raw_year.isdigit():
            year = int(raw_year)
        else:
            year = None
        journal_value = metadata.get("journal")
        journal = str(journal_value).strip() if journal_value is not None else None
        return PubMedArticle(
            pmid=document.pmid,
            title=document.title,
            abstract=document.content,
            journal=journal or None,
            authors=authors,
            year=year,
        )
