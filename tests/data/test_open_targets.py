"""Tests for the Open Targets client."""

from __future__ import annotations

import json

import httpx
import pytest

from biomedos.data.open_targets import OpenTargetsClient


@pytest.mark.asyncio
async def test_open_targets_gene_and_associations() -> None:
    """GraphQL responses are parsed into typed models."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        query = str(payload.get("query", ""))
        if "approvedSymbol" in query:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "target": {
                            "id": "ENSG00000146648",
                            "approvedSymbol": "EGFR",
                            "approvedName": "epidermal growth factor receptor",
                            "biotype": "protein_coding",
                            "functionDescriptions": ["Receptor tyrosine kinase."],
                        }
                    }
                },
            )
        if "associatedDiseases" in query:
            return httpx.Response(
                200,
                json={
                    "data": {
                        "target": {
                            "associatedDiseases": {
                                "rows": [
                                    {
                                        "disease": {"id": "EFO:0001071", "name": "lung cancer"},
                                        "score": 0.97,
                                        "datasourceScores": [{"score": 0.7}, {"score": 0.6}],
                                    }
                                ]
                            }
                        }
                    }
                },
            )
        return httpx.Response(
            200,
            json={
                "data": {
                    "target": {
                        "tractability": [
                            {"modality": "SM", "label": "Small molecule", "value": True}
                        ]
                    }
                }
            },
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="https://api.platform.opentargets.org/api/v4/graphql",
        transport=transport,
    ) as client:
        open_targets = OpenTargetsClient(client=client)
        gene = await open_targets.get_gene_info("ENSG00000146648")
        associations = await open_targets.get_disease_associations("ENSG00000146648")
        tractability = await open_targets.get_tractability("ENSG00000146648")

    assert gene is not None
    assert gene.symbol == "EGFR"
    assert associations[0].disease_name == "lung cancer"
    assert associations[0].evidence_count == 2
    assert tractability[0].value is True
