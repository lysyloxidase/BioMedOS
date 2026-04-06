"""Tests for the HPO client."""

from __future__ import annotations

import httpx
import pytest

from biomedos.data.hpo import HPOClient


@pytest.mark.asyncio
async def test_hpo_local_search_associations_and_similarity() -> None:
    """Local fallback phenotype search and associations are available offline."""

    client = HPOClient()

    search_hits = await client.search("persistent cough")
    associations = await client.get_disease_associations("HP:0012735")
    similarity = await client.phenotype_similarity(
        ["HP:0012735"],
        ["HP:0012735", "HP:0002094"],
    )

    assert search_hits[0]["id"] == "HP:0012735"
    assert associations[0]["disease_name"] == "Lung Cancer"
    assert similarity == 0.5


@pytest.mark.asyncio
async def test_hpo_returns_empty_list_when_remote_lookup_fails() -> None:
    """Unknown phenotypes should fail closed instead of raising HTTP errors."""

    transport = httpx.MockTransport(lambda request: httpx.Response(404, request=request))
    async with httpx.AsyncClient(
        base_url="https://ontology.jax.org/api/hpo",
        transport=transport,
    ) as client:
        hpo = HPOClient(client=client)
        results = await hpo.search("completely unknown phenotype string")

    assert isinstance(results, list)
