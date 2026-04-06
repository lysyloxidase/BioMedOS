"""GRADE-style evidence level definitions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class EvidenceLevel(StrEnum):
    """Clinical evidence quality labels."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"


class EvidenceAssessment(BaseModel):
    """A GRADE-style evidence assessment."""

    level: EvidenceLevel
    score: float
    rationale: str


class GradeEvidenceClassifier:
    """Classify biomedical evidence with simple GRADE heuristics."""

    def classify(self, evidence_text: str) -> EvidenceAssessment:
        """Classify evidence strength from free text."""

        lowered = evidence_text.lower()
        if any(keyword in lowered for keyword in ("systematic review", "meta-analysis")):
            return EvidenceAssessment(
                level=EvidenceLevel.HIGH,
                score=0.9,
                rationale="Systematic review or meta-analysis language detected.",
            )
        if any(keyword in lowered for keyword in ("randomized", "phase 3", "clinical trial")):
            return EvidenceAssessment(
                level=EvidenceLevel.MODERATE,
                score=0.75,
                rationale="Prospective clinical-trial language detected.",
            )
        if any(keyword in lowered for keyword in ("cohort", "case-control", "observational")):
            return EvidenceAssessment(
                level=EvidenceLevel.LOW,
                score=0.55,
                rationale="Observational-study language detected.",
            )
        if any(
            keyword in lowered for keyword in ("case report", "preclinical", "mouse", "in vitro")
        ):
            return EvidenceAssessment(
                level=EvidenceLevel.VERY_LOW,
                score=0.3,
                rationale="Preclinical or case-report language detected.",
            )
        return EvidenceAssessment(
            level=EvidenceLevel.LOW,
            score=0.45,
            rationale="No strong trial or review indicators were detected.",
        )
