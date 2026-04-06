"""Tests for the OpenFDA client."""

from __future__ import annotations

import httpx
import pytest

from biomedos.data.openfda import OpenFDAClient


@pytest.mark.asyncio
async def test_openfda_event_label_and_recall_queries() -> None:
    """OpenFDA result payloads are normalized into plain dictionaries."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/event.json"):
            return httpx.Response(200, json={"results": [{"safetyreportid": "1"}]})
        if request.url.path.endswith("/label.json"):
            return httpx.Response(
                200,
                json={"results": [{"openfda": {"generic_name": ["gefitinib"]}}]},
            )
        return httpx.Response(200, json={"results": [{"recall_number": "R-1"}]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(
        base_url="https://api.fda.gov/drug",
        transport=transport,
    ) as client:
        openfda = OpenFDAClient(client=client)
        events = await openfda.adverse_events("gefitinib")
        labels = await openfda.drug_labels("gefitinib")
        recalls = await openfda.recalls("gefitinib")

    assert events == [{"safetyreportid": "1"}]
    assert labels[0]["openfda"] == {"generic_name": ["gefitinib"]}
    assert recalls[0]["recall_number"] == "R-1"


@pytest.mark.asyncio
async def test_openfda_falls_back_to_local_demo_records_on_http_errors() -> None:
    """Bundled OpenFDA-like records should be used when the live API fails."""

    transport = httpx.MockTransport(lambda request: httpx.Response(404, request=request))
    async with httpx.AsyncClient(
        base_url="https://api.fda.gov/drug",
        transport=transport,
    ) as client:
        openfda = OpenFDAClient(client=client)
        events = await openfda.adverse_events("gefitinib")
        labels = await openfda.drug_labels("gefitinib")
        recalls = await openfda.recalls("gefitinib")

    assert events
    assert labels
    assert recalls == []
