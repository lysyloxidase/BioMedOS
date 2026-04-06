"""Tests for the pharmacologist agent."""

from __future__ import annotations

from typing import cast

import pytest

from biomedos.agents.pharmacologist import PharmacologistAgent
from biomedos.core.llm_client import OllamaClient
from biomedos.data.chembl import ChEMBLMechanism
from biomedos.orchestration.state import Task, TaskType


class FakeRxNormClient:
    """Minimal RxNorm stub."""

    async def normalize_drug(self, name: str) -> dict[str, object]:
        """Normalize a drug name."""

        mapping = {"gefitinib": "1234", "olaparib": "5678"}
        return {"normalized_name": name.title(), "rxcui": mapping.get(name.lower(), "")}

    async def get_ndc_codes(self, rxcui: str) -> list[str]:
        """Return fake NDC codes."""

        return [f"{rxcui}-NDC"] if rxcui else []


class FakeOpenFDAClient:
    """Minimal OpenFDA stub."""

    async def adverse_events(self, drug_name: str) -> list[dict[str, object]]:
        """Return one adverse-event record."""

        del drug_name
        return [{"patient": {"reaction": [{"reactionmeddrapt": "Rash"}]}}]

    async def drug_labels(self, drug_name: str) -> list[dict[str, object]]:
        """Return one label record."""

        return [
            {
                "pharmacokinetics": [f"{drug_name} has predictable oral absorption."],
                "clinical_pharmacology": [f"{drug_name} is metabolized through CYP3A4."],
                "contraindications": ["Avoid in fibrosis."],
            }
        ]

    async def recalls(self, drug_name: str) -> list[dict[str, object]]:
        """Return one recall record."""

        return [{"drug": drug_name, "status": "none"}]


class FakeDDIChecker:
    """Minimal DDI checker stub."""

    async def check(self, drugs: list[str]) -> list[object]:
        """Return an empty interaction list."""

        del drugs
        return []


class FakeContraindicationChecker:
    """Minimal contraindication checker stub."""

    async def check(self, drug_name: str, conditions: list[str]) -> list[object]:
        """Return no contraindication hits."""

        del drug_name, conditions
        return []


class FakeChemblClient:
    """Minimal ChEMBL stub."""

    async def get_mechanisms(self, chembl_id: str) -> list[ChEMBLMechanism]:
        """Return one mechanism for a known drug."""

        del chembl_id
        return [
            ChEMBLMechanism(
                target_chembl_id="CHEMBL203",
                mechanism="EGFR inhibition",
                action_type="Inhibitor",
            )
        ]


@pytest.mark.asyncio
async def test_pharmacologist_agent_summarizes_safety_and_pk(
    mock_ollama: object,
    sample_kg: object,
) -> None:
    """Pharmacologist agent returns PK and safety details for supplied drugs."""

    agent = PharmacologistAgent(
        llm_client=cast(OllamaClient, mock_ollama),
        knowledge_graph=sample_kg,
        rxnorm_client=FakeRxNormClient(),  # type: ignore[arg-type]
        openfda_client=FakeOpenFDAClient(),  # type: ignore[arg-type]
        chembl_client=FakeChemblClient(),  # type: ignore[arg-type]
        ddi_checker=FakeDDIChecker(),  # type: ignore[arg-type]
        contraindication_checker=FakeContraindicationChecker(),  # type: ignore[arg-type]
    )
    task = Task(
        id="pharm-1",
        type=TaskType.PHARMACOLOGIST,
        description="Review Gefitinib and Olaparib.",
        payload={"drugs": ["Gefitinib", "Olaparib"], "conditions": ["fibrosis"]},
    )

    result = await agent.run(task)

    assert "Pharmacology review for: Gefitinib, Olaparib" in result.summary
    assert "PK=" in result.summary
    assert "metabolism=" in result.summary
    assert len(result.output["drug_reports"]) == 2
