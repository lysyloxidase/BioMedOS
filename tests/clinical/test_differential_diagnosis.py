"""Tests for differential diagnosis ranking."""

from __future__ import annotations

from biomedos.clinical.differential_diagnosis import DifferentialDiagnosis
from biomedos.clinical.phenotype_matcher import PhenotypeMatcher


async def test_differential_diagnosis_uses_hpo_and_kg(sample_kg: object) -> None:
    """Differential diagnosis combines phenotype evidence with KG support."""

    matcher = PhenotypeMatcher(knowledge_graph=sample_kg)
    engine = DifferentialDiagnosis(matcher, knowledge_graph=sample_kg)

    diagnoses = await engine.rank(["pulmonary fibrosis"], top_k=3)

    assert diagnoses[0].disease_name == "Fibrosis"
    assert diagnoses[0].literature_evidence_count >= 1
