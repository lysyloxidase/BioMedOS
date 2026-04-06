"""PubMed E-utilities client."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel, Field

from biomedos.data.base_client import AsyncAPIClient

PUBTATOR_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"


class PubMedAnnotation(BaseModel):
    """A named entity annotation from PubTator."""

    text: str
    type: str
    identifier: str | None = None
    offset: int | None = None
    length: int | None = None


class PubMedArticle(BaseModel):
    """Structured representation of a PubMed record."""

    pmid: str
    title: str
    abstract: str
    journal: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    mesh_terms: list[str] = Field(default_factory=list)
    doi: str | None = None
    keywords: list[str] = Field(default_factory=list)


class PubMedClient(AsyncAPIClient):
    """Client for PubMed search and article retrieval."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Initialize the PubMed client."""

        super().__init__(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
            requests_per_second=3.0,
            max_concurrency=3,
            client=client,
        )

    async def search(self, query: str, *, max_results: int = 20) -> list[str]:
        """Search PubMed and return matching PMIDs."""

        payload = await self._request_json(
            "GET",
            "/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmode": "json",
                "retmax": max_results,
                "sort": "relevance",
            },
        )
        search_result = payload.get("esearchresult", {})
        if not isinstance(search_result, dict):
            return []
        ids = search_result.get("idlist", [])
        if not isinstance(ids, list):
            return []
        return [str(identifier) for identifier in ids]

    async def fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        """Fetch PubMed article metadata and abstracts in XML."""

        if not pmids:
            return []

        xml_payload = await self._request_text(
            "GET",
            "/efetch.fcgi",
            params={
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            },
        )
        root = ET.fromstring(xml_payload)
        return [self._parse_article(node) for node in root.findall(".//PubmedArticle")]

    async def fetch_annotations(self, pmids: list[str]) -> dict[str, list[PubMedAnnotation]]:
        """Fetch PubTator annotations for one or more PMIDs."""

        if not pmids:
            return {}

        payload = await self._request_json(
            "GET",
            PUBTATOR_URL,
            params={"pmids": ",".join(pmids)},
        )
        documents = payload.get("documents", [])
        if not isinstance(documents, list):
            return {}

        annotations: dict[str, list[PubMedAnnotation]] = {}
        for document in documents:
            if not isinstance(document, dict):
                continue
            pmid = str(document.get("id", ""))
            collected: list[PubMedAnnotation] = []
            passages = document.get("passages", [])
            if not isinstance(passages, list):
                continue
            for passage in passages:
                if not isinstance(passage, dict):
                    continue
                items = passage.get("annotations", [])
                if not isinstance(items, list):
                    continue
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    infons = item.get("infons", {})
                    locations = item.get("locations", [])
                    location = locations[0] if isinstance(locations, list) and locations else {}
                    if not isinstance(infons, dict) or not isinstance(location, dict):
                        continue
                    collected.append(
                        PubMedAnnotation(
                            text=str(item.get("text", "")),
                            type=str(infons.get("type", "")),
                            identifier=(
                                str(infons.get("identifier"))
                                if infons.get("identifier") is not None
                                else None
                            ),
                            offset=self._safe_int(location.get("offset")),
                            length=self._safe_int(location.get("length")),
                        )
                    )
            annotations[pmid] = collected
        return annotations

    def _parse_article(self, node: ET.Element) -> PubMedArticle:
        """Parse a PubMed XML article node into a model."""

        pmid = node.findtext(".//PMID", default="")
        title = "".join(node.findtext(".//ArticleTitle", default="") or "")

        abstract_fragments = []
        for abstract_node in node.findall(".//Abstract/AbstractText"):
            text = "".join(abstract_node.itertext()).strip()
            if text:
                abstract_fragments.append(text)
        abstract = "\n".join(abstract_fragments)

        authors: list[str] = []
        for author_node in node.findall(".//AuthorList/Author"):
            last_name = author_node.findtext("LastName")
            initials = author_node.findtext("Initials")
            collective_name = author_node.findtext("CollectiveName")
            if collective_name:
                authors.append(collective_name)
            elif last_name and initials:
                authors.append(f"{last_name} {initials}")
            elif last_name:
                authors.append(last_name)

        mesh_terms = [
            mesh.text.strip()
            for mesh in node.findall(".//MeshHeading/DescriptorName")
            if mesh.text is not None
        ]
        keywords = [
            keyword.text.strip()
            for keyword in node.findall(".//KeywordList/Keyword")
            if keyword.text is not None
        ]

        doi: str | None = None
        for article_id in node.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi" and article_id.text:
                doi = article_id.text.strip()
                break

        year_text = node.findtext(".//PubDate/Year") or node.findtext(".//ArticleDate/Year")
        year = int(year_text) if year_text and year_text.isdigit() else None

        return PubMedArticle(
            pmid=pmid,
            title=title.strip(),
            abstract=abstract,
            journal=node.findtext(".//Journal/Title"),
            authors=authors,
            year=year,
            mesh_terms=mesh_terms,
            doi=doi,
            keywords=keywords,
        )

    @staticmethod
    def _safe_int(value: object) -> int | None:
        """Safely coerce a JSON scalar to an integer."""

        if value is None:
            return None
        return int(str(value))
