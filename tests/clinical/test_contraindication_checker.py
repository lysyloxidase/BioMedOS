"""Tests for contraindication checking."""

from __future__ import annotations

import pytest

from biomedos.clinical.contraindication_checker import ContraindicationChecker


class FakeOpenFDALabels:
    """Minimal OpenFDA label stub."""

    async def drug_labels(self, drug_name: str) -> list[dict[str, object]]:
        """Return one label record mentioning fibrosis."""

        del drug_name
        return [
            {
                "contraindications": ["Avoid in patients with pulmonary fibrosis."],
                "warnings": ["Monitor for respiratory toxicity."],
            }
        ]


@pytest.mark.asyncio
async def test_contraindication_checker_finds_matching_conditions() -> None:
    """Contraindications are matched from OpenFDA label text."""

    checker = ContraindicationChecker(FakeOpenFDALabels())  # type: ignore[arg-type]
    warnings = await checker.check("Gefitinib", ["fibrosis"])

    assert len(warnings) == 1
    assert warnings[0].condition == "fibrosis"
    assert "pulmonary fibrosis" in warnings[0].warning.lower()
