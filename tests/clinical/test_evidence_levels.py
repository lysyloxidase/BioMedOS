"""Tests for GRADE-style evidence classification."""

from __future__ import annotations

from biomedos.clinical.evidence_levels import EvidenceLevel, GradeEvidenceClassifier


def test_grade_classifier_assigns_high_evidence_to_meta_analysis() -> None:
    """Systematic review language maps to high evidence."""

    classifier = GradeEvidenceClassifier()

    assessment = classifier.classify(
        "This systematic review and meta-analysis summarized randomized trials."
    )

    assert assessment.level == EvidenceLevel.HIGH
    assert assessment.score > 0.8
