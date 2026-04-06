"""Tests for the DDI checker."""

from __future__ import annotations

import pytest

from biomedos.clinical.ddi_checker import DDIChecker, DDISeverity


class FakeRxNormClient:
    """Minimal RxNorm stub for interaction tests."""

    async def normalize_drug(self, name: str) -> dict[str, object]:
        """Return a deterministic normalization payload."""

        mapping = {"gefitinib": "1234", "olaparib": "5678"}
        return {"input_name": name, "normalized_name": name.title(), "rxcui": mapping[name.lower()]}

    async def check_interactions(self, rxcuis: list[str]) -> list[dict[str, object]]:
        """Return one major interaction record."""

        assert rxcuis == ["1234", "5678"]
        return [
            {
                "drug_a": "Gefitinib",
                "drug_b": "Olaparib",
                "description": "Major interaction requiring specialist oversight.",
                "severity": "major",
            }
        ]


class FakeOpenFDAClient:
    """Minimal OpenFDA stub for shared-event severity escalation."""

    async def adverse_events(self, drug_name: str) -> list[dict[str, object]]:
        """Return overlapping adverse events for both drugs."""

        del drug_name
        return [
            {
                "patient": {
                    "reaction": [
                        {"reactionmeddrapt": "Interstitial lung disease"},
                    ]
                }
            }
        ]


@pytest.mark.asyncio
async def test_ddi_checker_combines_rxnorm_and_openfda() -> None:
    """Severity and recommendations combine interaction and adverse-event evidence."""

    checker = DDIChecker(
        rxnorm_client=FakeRxNormClient(),  # type: ignore[arg-type]
        openfda_client=FakeOpenFDAClient(),  # type: ignore[arg-type]
    )

    results = await checker.check(["Gefitinib", "Olaparib"])

    assert len(results) == 1
    assert results[0].severity == DDISeverity.CONTRAINDICATED
    assert "Avoid" in results[0].recommendation
    assert "OpenFDA" in results[0].evidence_sources
