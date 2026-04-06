"""PubMed indexing utilities for the vector store."""

from __future__ import annotations

from collections import Counter

from biomedos.core.vector_store import ChromaVectorStore, VectorDocument
from biomedos.data.pubmed import PubMedArticle, PubMedClient


class PubMedIndexer:
    """Index PubMed abstracts into ChromaDB and the BM25 corpus."""

    def __init__(
        self,
        pubmed_client: PubMedClient,
        vector_store: ChromaVectorStore,
    ) -> None:
        """Initialize the indexer.

        Args:
            pubmed_client: PubMed search and fetch client.
            vector_store: Local vector store used for retrieval.
        """

        self.pubmed_client = pubmed_client
        self.vector_store = vector_store
        self._articles_by_pmid: dict[str, PubMedArticle] = {}
        self._source_counts: Counter[str] = Counter()
        self._year_counts: Counter[str] = Counter()

    async def index_for_genes(self, genes: list[str], max_per_gene: int = 100) -> int:
        """Index PubMed abstracts for a list of genes.

        Args:
            genes: Gene symbols to query in PubMed.
            max_per_gene: Maximum number of abstracts per gene.

        Returns:
            Number of newly indexed abstracts.
        """

        indexed_total = 0
        for gene in genes:
            indexed_total += await self.index_for_query(
                f"{gene} AND (disease OR biomarker OR therapy)",
                max_results=max_per_gene,
            )
        return indexed_total

    async def index_for_query(self, query: str, max_results: int = 500) -> int:
        """Index PubMed abstracts for a custom query.

        Args:
            query: Arbitrary PubMed query string.
            max_results: Maximum number of records to index.

        Returns:
            Number of newly indexed documents.
        """

        pmids = await self.pubmed_client.search(query, max_results=max_results)
        articles = await self.pubmed_client.fetch_abstracts(pmids)
        return self.index_articles(articles, source="pubmed")

    def index_articles(self, articles: list[PubMedArticle], source: str = "pubmed") -> int:
        """Index a batch of already-fetched PubMed articles.

        Args:
            articles: PubMed articles to index.
            source: Source label stored in metadata and stats.

        Returns:
            Number of newly indexed documents.
        """

        documents: list[VectorDocument] = []
        new_count = 0
        for article in articles:
            if article.pmid in self._articles_by_pmid:
                continue

            self._articles_by_pmid[article.pmid] = article
            self._source_counts[source] += 1
            self._year_counts[str(article.year) if article.year is not None else "unknown"] += 1
            new_count += 1
            documents.append(
                VectorDocument(
                    id=self._document_id(article.pmid),
                    text=self._article_text(article),
                    metadata={
                        "pmid": article.pmid,
                        "title": article.title,
                        "journal": article.journal or "",
                        "year": article.year or 0,
                        "authors": "; ".join(article.authors),
                        "source": source,
                    },
                )
            )

        if documents:
            self.vector_store.add_documents(documents)
        return new_count

    def get_article(self, pmid: str) -> PubMedArticle | None:
        """Return a previously indexed PubMed article."""

        return self._articles_by_pmid.get(pmid)

    def get_articles(self, pmids: list[str]) -> list[PubMedArticle]:
        """Return indexed PubMed articles for a list of PMIDs."""

        return [article for pmid in pmids if (article := self.get_article(pmid)) is not None]

    def get_stats(self) -> dict[str, object]:
        """Return indexing statistics."""

        return {
            "total_docs": len(self._articles_by_pmid),
            "by_source": dict(self._source_counts),
            "by_year": dict(self._year_counts),
        }

    @staticmethod
    def _document_id(pmid: str) -> str:
        """Return a stable vector-store document identifier."""

        return f"pmid:{pmid}"

    @staticmethod
    def _article_text(article: PubMedArticle) -> str:
        """Build the retrievable text payload for an article."""

        parts = [article.title.strip(), article.abstract.strip()]
        return "\n\n".join(part for part in parts if part)
