"""Tests for the RxNorm client."""

from __future__ import annotations

import httpx
import pytest

from biomedos.data.rxnorm import RxNormClient


@pytest.mark.asyncio
async def test_rxnorm_normalization_interactions_and_ndc_codes() -> None:
    """RxNorm endpoints are parsed into normalized interaction data."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/rxcui.json"):
            return httpx.Response(
                200,
                json={"idGroup": {"name": "Gefitinib", "rxnormId": ["1234"]}},
            )
        if request.url.path.endswith("/interaction/list.json"):
            return httpx.Response(
                200,
                json={
                    "fullInteractionTypeGroup": [
                        {
                            "sourceName": "ONCHigh",
                            "fullInteractionType": [
                                {
                                    "interactionPair": [
                                        {
                                            "description": "Major interaction. Avoid combination.",
                                            "interactionConcept": [
                                                {
                                                    "minConceptItem": {
                                                        "rxcui": "1234",
                                                        "name": "Gefitinib",
                                                    }
                                                },
                                                {
                                                    "minConceptItem": {
                                                        "rxcui": "5678",
                                                        "name": "Olaparib",
                                                    }
                                                },
                                            ],
                                        }
                                    ]
                                }
                            ],
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={
                "historicalNdcConcept": {
                    "historicalNdcTime": [{"ndcTime": [{"ndc": "00011-1111-11"}]}]
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="https://rxnav.nlm.nih.gov/REST",
        transport=transport,
    ) as client:
        rxnorm = RxNormClient(client=client)
        normalized = await rxnorm.normalize_drug("gefitinib")
        interactions = await rxnorm.check_interactions(["1234", "5678"])
        ndc_codes = await rxnorm.get_ndc_codes("1234")

    assert normalized["rxcui"] == "1234"
    assert normalized["normalized_name"] == "Gefitinib"
    assert interactions[0]["drug_a"] == "Gefitinib"
    assert interactions[0]["severity"] == "contraindicated"
    assert ndc_codes == ["00011-1111-11"]


@pytest.mark.asyncio
async def test_rxnorm_falls_back_to_local_records_on_http_errors() -> None:
    """Known demo drugs should still resolve when the live API fails."""

    transport = httpx.MockTransport(lambda request: httpx.Response(404, request=request))
    async with httpx.AsyncClient(
        base_url="https://rxnav.nlm.nih.gov/REST",
        transport=transport,
    ) as client:
        rxnorm = RxNormClient(client=client)
        normalized = await rxnorm.normalize_drug("gefitinib")
        interactions = await rxnorm.check_interactions(["328134", "6809"])
        ndc_codes = await rxnorm.get_ndc_codes("328134")

    assert normalized["rxcui"] == "328134"
    assert interactions[0]["drug_b"] == "Metformin"
    assert ndc_codes == ["0052-0603-30"]
