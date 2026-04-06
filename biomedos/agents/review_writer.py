"""Narrative review writing agent."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from pydantic import BaseModel, Field

from biomedos.agents.base import BaseAgent
from biomedos.agents.literature import LiteratureAgent
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


class ReviewSection(BaseModel):
    """A narrative review section definition."""

    heading: str
    question: str
    content: str = ""
    critique_score: float = 0.0
    critique_feedback: str = ""
    citations: list[str] = Field(default_factory=list)


class ReviewOutline(BaseModel):
    """Review title and section outline."""

    title: str
    sections: list[ReviewSection]


class ReviewWriterAgent(BaseAgent):
    """Draft biomedical reviews from structured evidence."""

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
        """Initialize the review writer agent."""

        resolved_vector_store = vector_store or ChromaVectorStore()
        resolved_graph = knowledge_graph or KnowledgeGraph()
        super().__init__(
            name="review_writer",
            llm_client=llm_client,
            knowledge_graph=resolved_graph,
            vector_store=resolved_vector_store,
            settings=settings,
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
        self._literature_helper = LiteratureAgent(
            llm_client=llm_client,
            knowledge_graph=resolved_graph,
            vector_store=resolved_vector_store,
            settings=self.settings,
            pubmed_client=self.pubmed_client,
            indexer=self.indexer,
            retriever=self.retriever,
            citation_tracker=self.citation_tracker,
            entity_extractor=self.entity_extractor,
        )

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the review writer agent."""

        del state
        topic = str(task.payload.get("query", task.description))
        if self._use_fast_review_mode():
            return await self._run_fast_review(task.id, topic)
        outline = await self._generate_outline(topic)

        all_references: dict[str, PubMedArticle] = {}
        completed_sections: list[ReviewSection] = []
        for section in outline.sections:
            queries = await self._generate_section_queries(topic, section)
            for query in queries:
                await self.indexer.index_for_query(query, max_results=15)

            retrieved = await self.retriever.retrieve(
                f"{topic} {section.heading} {section.question}",
                top_k=15,
                include_kg=True,
            )
            references = self._resolve_references(retrieved)
            for reference in references:
                all_references[reference.pmid] = reference

            drafted = await self._draft_section(section, topic, retrieved, references)
            completed_sections.append(drafted)

        abstract_text = await self._generate_abstract(topic, outline.title, completed_sections)
        raw_review = self._assemble_review(outline.title, abstract_text, completed_sections)
        references = list(all_references.values())
        formatted_review, bibliography = self.citation_tracker.format_citations(
            raw_review,
            references,
        )

        invalid_citations = self.citation_tracker.verify_citations(
            raw_review,
            set(all_references),
        )
        extracted_nodes: list[str] = []
        if self.entity_extractor is not None:
            try:
                extraction = await self.entity_extractor.extract_triples(formatted_review)
            except ValueError:
                extraction = None
            if extraction is not None:
                self.update_kg(nodes=extraction.entities, edges=extraction.relations)
                extracted_nodes = [node.id for node in extraction.entities]

        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=formatted_review,
            output={
                "content": formatted_review,
                "bibliography": bibliography,
                "outline": outline.model_dump(mode="python"),
                "references": [reference.model_dump(mode="python") for reference in references],
                "invalid_citations": invalid_citations,
            },
            citations=self.citation_tracker.extract_cited_pmids(raw_review),
            confidence=self._review_confidence(completed_sections, invalid_citations),
            node_ids=extracted_nodes,
            errors=self._build_errors(invalid_citations),
        )

    async def _run_fast_review(self, task_id: str, topic: str) -> AgentResult:
        """Execute a lightweight deterministic review flow for local CPU-first use."""

        outline = self._fallback_outline(topic)
        outline.sections = outline.sections[:3]
        completed_sections: list[ReviewSection] = []
        all_references: dict[str, PubMedArticle] = {}

        for section in outline.sections:
            retrieved = await self.retriever.retrieve(
                f"{topic} {section.heading} {section.question}",
                top_k=6,
                include_kg=True,
            )
            references = self._resolve_references(retrieved)
            for reference in references:
                all_references[reference.pmid] = reference
            content = self._fallback_section_text(section, retrieved[:4])
            critique_score, critique_feedback = self._heuristic_critique(content)
            completed_sections.append(
                section.model_copy(
                    update={
                        "content": content,
                        "critique_score": critique_score,
                        "critique_feedback": critique_feedback,
                        "citations": self.citation_tracker.extract_cited_pmids(content),
                    }
                )
            )

        abstract_text = (
            f"This rapid narrative review summarizes local evidence for {topic}. "
            f"It covers {', '.join(section.heading for section in completed_sections)}."
        )
        raw_review = self._assemble_review(outline.title, abstract_text, completed_sections)
        references = list(all_references.values())
        formatted_review, bibliography = self.citation_tracker.format_citations(
            raw_review,
            references,
        )
        invalid_citations = self.citation_tracker.verify_citations(raw_review, set(all_references))
        return AgentResult(
            agent_name=self.name,
            task_id=task_id,
            summary=formatted_review,
            output={
                "content": formatted_review,
                "bibliography": bibliography,
                "outline": outline.model_dump(mode="python"),
                "references": [reference.model_dump(mode="python") for reference in references],
                "invalid_citations": invalid_citations,
                "mode": "fast_local",
            },
            citations=self.citation_tracker.extract_cited_pmids(raw_review),
            confidence=self._review_confidence(completed_sections, invalid_citations),
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

    async def _generate_outline(self, topic: str) -> ReviewOutline:
        """Generate a review outline."""

        if self.llm_client is None:
            return self._fallback_outline(topic)

        prompt = (
            "Create a biomedical narrative review outline.\n"
            "Return JSON with the schema:\n"
            '{"title": "...", "sections": [{"heading": "...", "question": "..."}]}\n'
            "Generate 4 to 6 sections.\n"
            f"Topic: {topic}"
        )
        try:
            response = await self.llm_client.generate(
                prompt,
                model=self.settings.MODEL_REASONER,
                system="Return only valid JSON.",
            )
            payload = self._parse_json(response)
        except ValueError:
            return self._fallback_outline(topic)

        try:
            return ReviewOutline.model_validate(payload)
        except Exception:
            return self._fallback_outline(topic)

    async def _generate_section_queries(self, topic: str, section: ReviewSection) -> list[str]:
        """Generate PubMed queries for a review section."""

        if self.llm_client is None:
            return self._fallback_section_queries(topic, section)

        prompt = (
            "Generate 2 or 3 PubMed search queries for the review section below.\n"
            'Return JSON with the schema {"queries": ["..."]}.\n'
            f"Topic: {topic}\n"
            f"Section heading: {section.heading}\n"
            f"Section question: {section.question}"
        )
        try:
            response = await self.llm_client.generate(
                prompt,
                model=self.settings.MODEL_ROUTER,
                system="Return only valid JSON.",
            )
            payload = self._parse_json(response)
        except ValueError:
            return self._fallback_section_queries(topic, section)

        queries = payload.get("queries", [])
        if not isinstance(queries, list):
            return self._fallback_section_queries(topic, section)
        cleaned = [str(item).strip() for item in queries if str(item).strip()]
        return cleaned[:3] if cleaned else self._fallback_section_queries(topic, section)

    async def _draft_section(
        self,
        section: ReviewSection,
        topic: str,
        documents: list[RetrievedDocument],
        references: list[PubMedArticle],
    ) -> ReviewSection:
        """Draft and critique a single review section."""

        best_section = section.model_copy(deep=True)
        for _ in range(3):
            raw_section = await self._generate_section_text(topic, best_section, documents)
            invalid = self.citation_tracker.verify_citations(
                raw_section,
                {reference.pmid for reference in references},
            )
            if invalid or not self.citation_tracker.extract_cited_pmids(raw_section):
                raw_section = self._fallback_section_text(best_section, documents)

            critique_score, critique_feedback = await self._critique_section(
                raw_section,
                documents,
            )
            best_section.content = raw_section
            best_section.critique_score = critique_score
            best_section.critique_feedback = critique_feedback
            best_section.citations = self.citation_tracker.extract_cited_pmids(raw_section)
            if critique_score >= 7.0:
                break
        return best_section

    async def _generate_section_text(
        self,
        topic: str,
        section: ReviewSection,
        documents: list[RetrievedDocument],
    ) -> str:
        """Generate raw section text with PMID citations."""

        if self.llm_client is None or not documents:
            return self._fallback_section_text(section, documents)

        evidence = "\n\n".join(
            (
                f"PMID: {document.pmid or 'unknown'}\n"
                f"Title: {document.title}\n"
                f"Content:\n{document.content}\n"
                f"KG context:\n{document.kg_context or ''}"
            )
            for document in documents[:10]
        )
        prompt = (
            "Write one section of a biomedical narrative review.\n"
            "Use ONLY the provided evidence.\n"
            "Cite factual claims with PMID markers in the form [PMID:12345].\n"
            f"Topic: {topic}\n"
            f"Section heading: {section.heading}\n"
            f"Section question: {section.question}\n\n"
            f"Evidence:\n{evidence}"
        )
        return await self.llm_client.generate(
            prompt,
            model=self.settings.MODEL_REASONER,
            system="Use only the supplied evidence and cite all claims.",
        )

    async def _critique_section(
        self,
        content: str,
        documents: list[RetrievedDocument],
    ) -> tuple[float, str]:
        """Critique a drafted review section."""

        del documents
        if self.llm_client is None:
            return self._heuristic_critique(content)

        prompt = (
            "Critique the following review section for citation coverage, logical flow, "
            "and hallucination risk.\n"
            'Return JSON with {"score": number, "feedback": "..."} '
            "where score is between 0 and 10.\n"
            f"Section:\n{content}"
        )
        try:
            response = await self.llm_client.generate(
                prompt,
                model=self.settings.MODEL_VERIFIER,
                system="Return only valid JSON.",
            )
            payload = self._parse_json(response)
            score = float(payload.get("score", 0.0))
            feedback = str(payload.get("feedback", ""))
            return score, feedback
        except ValueError:
            return self._heuristic_critique(content)

    async def _generate_abstract(
        self,
        topic: str,
        title: str,
        sections: list[ReviewSection],
    ) -> str:
        """Generate a review abstract."""

        section_summaries = "\n".join(
            f"- {section.heading}: {section.content[:200]}" for section in sections
        )
        if self.llm_client is None:
            return (
                f"This narrative review summarizes current evidence on {topic}. "
                f"It covers {', '.join(section.heading for section in sections[:3])}."
            )

        section_summary_block = f"Section summaries:\n{section_summaries}"
        prompt = (
            "Write a concise biomedical review abstract based only on the section "
            "summaries below.\n"
            f"Title: {title}\n"
            f"Topic: {topic}\n"
            f"{section_summary_block}"
        )
        return await self.llm_client.generate(
            prompt,
            model=self.settings.MODEL_REASONER,
            system="Be concise and evidence-focused.",
        )

    def _resolve_references(self, documents: list[RetrievedDocument]) -> list[PubMedArticle]:
        """Resolve indexed references from retrieved documents."""

        references: list[PubMedArticle] = []
        seen_pmids: set[str] = set()
        for document in documents:
            if document.pmid is None or document.pmid in seen_pmids:
                continue
            article = self.indexer.get_article(document.pmid) or self._article_from_document(
                document
            )
            if article is None:
                continue
            references.append(article)
            seen_pmids.add(article.pmid)
        return references

    def _assemble_review(
        self,
        title: str,
        abstract_text: str,
        sections: list[ReviewSection],
    ) -> str:
        """Assemble the full review manuscript."""

        parts = [f"# {title}", "## Abstract", abstract_text.strip()]
        for section in sections:
            parts.append(f"## {section.heading}")
            parts.append(section.content.strip())
        return "\n\n".join(part for part in parts if part)

    def _fallback_outline(self, topic: str) -> ReviewOutline:
        """Return a deterministic outline when the LLM is unavailable."""

        return ReviewOutline(
            title=f"Narrative Review of {topic}",
            sections=[
                ReviewSection(
                    heading="Background",
                    question=f"What is the current biomedical background for {topic}?",
                ),
                ReviewSection(
                    heading="Molecular Mechanisms",
                    question=f"What molecular mechanisms are implicated in {topic}?",
                ),
                ReviewSection(
                    heading="Clinical Evidence",
                    question=f"What clinical evidence exists for {topic}?",
                ),
                ReviewSection(
                    heading="Research Gaps",
                    question=f"What unanswered questions remain for {topic}?",
                ),
            ],
        )

    def _fallback_section_queries(self, topic: str, section: ReviewSection) -> list[str]:
        """Return deterministic fallback PubMed queries for a section."""

        return [
            f"{topic} {section.heading}",
            f"{topic} {section.question}",
        ]

    def _fallback_section_text(
        self,
        section: ReviewSection,
        documents: list[RetrievedDocument],
    ) -> str:
        """Create a deterministic evidence-grounded section draft."""

        if not documents:
            return (
                f"{section.heading}.\n"
                f"Insufficient evidence was retrieved to answer: {section.question}"
            )

        lines = [f"{section.heading}.", f"This section addresses: {section.question}"]
        for document in documents[:3]:
            citation = f"[PMID:{document.pmid}]" if document.pmid is not None else ""
            sentence = self._literature_helper._first_sentence(document.content)
            lines.append(f"- {sentence} {citation}".strip())
        return "\n".join(lines)

    def _heuristic_critique(self, content: str) -> tuple[float, str]:
        """Heuristic critique fallback."""

        score = 5.0
        feedback = []
        if len(content.split()) >= 80:
            score += 2.0
        else:
            feedback.append("Section is brief and may need more detail.")

        citation_count = len(self.citation_tracker.extract_cited_pmids(content))
        if citation_count >= 2:
            score += 2.5
        else:
            feedback.append("Citation coverage is limited.")

        if "insufficient evidence" not in content.lower():
            score += 0.5
        return min(score, 10.0), " ".join(feedback) or "Section is adequately grounded."

    def _review_confidence(
        self,
        sections: list[ReviewSection],
        invalid_citations: list[str],
    ) -> float:
        """Estimate confidence for the assembled review."""

        if not sections:
            return 0.1
        average_score = sum(section.critique_score for section in sections) / len(sections)
        confidence = 0.3 + min(average_score / 10.0, 1.0) * 0.6
        if invalid_citations:
            confidence -= 0.2
        return max(0.0, min(confidence, 0.95))

    @staticmethod
    def _build_errors(invalid_citations: list[str]) -> list[str]:
        """Build structured error messages."""

        if not invalid_citations:
            return []
        return [f"Invalid citations: {', '.join(invalid_citations)}"]

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse the first JSON object from an LLM response."""

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

        msg = "Failed to parse JSON payload from review writer response."
        raise ValueError(msg)

    def _use_fast_review_mode(self) -> bool:
        """Return whether review generation should prefer a lightweight local path."""

        return self.llm_client is None or self.should_use_fast_path()

    @staticmethod
    def _article_from_document(document: RetrievedDocument) -> PubMedArticle | None:
        """Reconstruct reference metadata from a retrieved document when needed."""

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
