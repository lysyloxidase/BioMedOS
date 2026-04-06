"""Tests for the PubMed client."""

from __future__ import annotations

import httpx
import pytest

from biomedos.data.pubmed import PubMedClient

PUBMED_XML = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>12345</PMID>
      <Article>
        <ArticleTitle>EGFR in lung cancer</ArticleTitle>
        <Abstract>
          <AbstractText>EGFR drives oncogenic signaling.</AbstractText>
        </Abstract>
        <Journal>
          <Title>Journal of Oncology</Title>
        </Journal>
        <AuthorList>
          <Author><LastName>Doe</LastName><Initials>J</Initials></Author>
        </AuthorList>
        <KeywordList>
          <Keyword>EGFR</Keyword>
        </KeywordList>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Carcinoma</DescriptorName></MeshHeading>
      </MeshHeadingList>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1000/egfr</ArticleId>
      </ArticleIdList>
    </MedlineCitation>
    <PubmedData>
      <History />
    </PubmedData>
  </PubmedArticle>
</PubmedArticleSet>
"""


@pytest.mark.asyncio
async def test_pubmed_search_fetch_and_annotations() -> None:
    """PubMed search, XML parsing, and annotations work with mocked responses."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/esearch.fcgi"):
            return httpx.Response(200, json={"esearchresult": {"idlist": ["12345"]}})
        if request.url.path.endswith("/efetch.fcgi"):
            return httpx.Response(200, text=PUBMED_XML)
        return httpx.Response(
            200,
            json={
                "documents": [
                    {
                        "id": "12345",
                        "passages": [
                            {
                                "annotations": [
                                    {
                                        "text": "EGFR",
                                        "infons": {"type": "Gene", "identifier": "HGNC:3236"},
                                        "locations": [{"offset": 0, "length": 4}],
                                    }
                                ]
                            }
                        ],
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils",
        transport=transport,
    ) as client:
        pubmed = PubMedClient(client=client)
        pmids = await pubmed.search("EGFR lung cancer")
        articles = await pubmed.fetch_abstracts(pmids)
        annotations = await pubmed.fetch_annotations(pmids)

    assert pmids == ["12345"]
    assert articles[0].title == "EGFR in lung cancer"
    assert articles[0].doi == "10.1000/egfr"
    assert annotations["12345"][0].identifier == "HGNC:3236"
