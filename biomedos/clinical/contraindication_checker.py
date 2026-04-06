"""Contraindication checking utilities."""

from __future__ import annotations

from pydantic import BaseModel

from biomedos.data.openfda import OpenFDAClient


class ContraindicationResult(BaseModel):
    """A contraindication warning extracted from drug labels."""

    drug_name: str
    condition: str
    warning: str
    source: str = "OpenFDA"


class ContraindicationChecker:
    """Check drug labels for contraindications."""

    def __init__(self, openfda_client: OpenFDAClient | None = None) -> None:
        """Initialize the contraindication checker."""

        self.openfda_client = openfda_client or OpenFDAClient()

    async def check(self, drug_name: str, conditions: list[str]) -> list[ContraindicationResult]:
        """Check a drug against known contraindications."""

        labels = await self.openfda_client.drug_labels(drug_name)
        warnings: list[ContraindicationResult] = []
        for label in labels:
            text = " ".join(
                str(label.get(field, ""))
                for field in ("contraindications", "warnings", "warnings_and_cautions")
            ).lower()
            for condition in conditions:
                if condition.lower() in text:
                    warnings.append(
                        ContraindicationResult(
                            drug_name=drug_name,
                            condition=condition,
                            warning=str(label.get("contraindications", label.get("warnings", ""))),
                        )
                    )
        return warnings
