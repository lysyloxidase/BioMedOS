"""Drug-drug interaction detection."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from biomedos.data.openfda import OpenFDAClient
from biomedos.data.rxnorm import RxNormClient


class DDISeverity(StrEnum):
    """Drug-drug interaction severity labels."""

    MINOR = "minor"
    MODERATE = "moderate"
    MAJOR = "major"
    CONTRAINDICATED = "contraindicated"


class DDIResult(BaseModel):
    """A normalized drug-drug interaction record."""

    drug_a: str
    drug_b: str
    severity: DDISeverity
    mechanism: str
    recommendation: str
    evidence_sources: list[str] = Field(default_factory=list)


class DDIChecker:
    """Check drug-drug interactions for medication lists."""

    def __init__(
        self,
        rxnorm_client: RxNormClient | None = None,
        openfda_client: OpenFDAClient | None = None,
    ) -> None:
        """Initialize the DDI checker."""

        self.rxnorm_client = rxnorm_client or RxNormClient()
        self.openfda_client = openfda_client or OpenFDAClient()

    async def check(self, drugs: list[str]) -> list[DDIResult]:
        """Check a medication list for interactions."""

        normalized = [await self.rxnorm_client.normalize_drug(drug) for drug in drugs]
        names_by_rxcui = {
            str(item.get("rxcui", "")): str(
                item.get("normalized_name", item.get("input_name", ""))
            ).title()
            for item in normalized
            if str(item.get("rxcui", ""))
        }
        rxcuis = [str(item.get("rxcui", "")) for item in normalized if str(item.get("rxcui", ""))]
        interaction_records = await self.rxnorm_client.check_interactions(rxcuis)
        adverse_events = {
            drug.lower(): await self.openfda_client.adverse_events(drug) for drug in drugs
        }

        results: list[DDIResult] = []
        seen_pairs: set[tuple[str, str]] = set()
        for record in interaction_records:
            rxcui_a = str(record.get("rxcui_a", ""))
            rxcui_b = str(record.get("rxcui_b", ""))
            drug_a = names_by_rxcui.get(rxcui_a, str(record.get("drug_a", "") or rxcui_a))
            drug_b = names_by_rxcui.get(rxcui_b, str(record.get("drug_b", "") or rxcui_b))
            if not drug_a or not drug_b:
                continue
            lowered_a = drug_a.lower()
            lowered_b = drug_b.lower()
            pair_key = (lowered_a, lowered_b) if lowered_a <= lowered_b else (lowered_b, lowered_a)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            description = str(record.get("description", "")).strip()
            severity = self._coerce_severity(str(record.get("severity", "minor")))
            original_severity = severity
            if self._share_severe_events(drug_a, drug_b, adverse_events):
                severity = self._escalate(severity)
            results.append(
                DDIResult(
                    drug_a=drug_a,
                    drug_b=drug_b,
                    severity=severity,
                    mechanism=description or "Potential pharmacologic interaction.",
                    recommendation=self._recommendation(severity),
                    evidence_sources=[
                        "RxNorm",
                        *(["OpenFDA"] if severity != original_severity else []),
                    ],
                )
            )
        return results

    def _coerce_severity(self, value: str) -> DDISeverity:
        """Normalize a severity string."""

        lowered = value.lower()
        if lowered in {DDISeverity.CONTRAINDICATED.value, "avoid"}:
            return DDISeverity.CONTRAINDICATED
        if lowered in {DDISeverity.MAJOR.value, "severe"}:
            return DDISeverity.MAJOR
        if lowered in {DDISeverity.MODERATE.value, "monitor"}:
            return DDISeverity.MODERATE
        return DDISeverity.MINOR

    def _escalate(self, severity: DDISeverity) -> DDISeverity:
        """Increase severity by one level."""

        order = [
            DDISeverity.MINOR,
            DDISeverity.MODERATE,
            DDISeverity.MAJOR,
            DDISeverity.CONTRAINDICATED,
        ]
        position = min(order.index(severity) + 1, len(order) - 1)
        return order[position]

    def _recommendation(self, severity: DDISeverity) -> str:
        """Map severity onto a clinical recommendation."""

        if severity == DDISeverity.CONTRAINDICATED:
            return "Avoid this drug combination."
        if severity == DDISeverity.MAJOR:
            return "Use only with specialist oversight or consider alternatives."
        if severity == DDISeverity.MODERATE:
            return "Monitor closely and consider dose adjustment."
        return "Routine monitoring is usually sufficient."

    def _share_severe_events(
        self,
        drug_a: str,
        drug_b: str,
        adverse_events: dict[str, list[dict[str, object]]],
    ) -> bool:
        """Return whether both drugs share serious adverse-event terms."""

        terms_a = self._extract_reaction_terms(adverse_events.get(drug_a.lower(), []))
        terms_b = self._extract_reaction_terms(adverse_events.get(drug_b.lower(), []))
        shared = terms_a & terms_b
        return bool(shared)

    def _extract_reaction_terms(self, records: list[dict[str, object]]) -> set[str]:
        """Extract reaction terms from adverse-event payloads."""

        terms: set[str] = set()
        for record in records:
            patient = record.get("patient", {})
            if not isinstance(patient, dict):
                continue
            reactions = patient.get("reaction", [])
            if not isinstance(reactions, list):
                continue
            for reaction in reactions:
                if not isinstance(reaction, dict):
                    continue
                term = reaction.get("reactionmeddrapt")
                if term is not None:
                    terms.add(str(term).lower())
        return terms
