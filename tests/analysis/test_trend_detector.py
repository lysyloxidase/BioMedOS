"""Tests for literature trend detection."""

from __future__ import annotations

from biomedos.analysis.trend_detector import TrendDetector


def test_trend_detector_summarizes_yearly_counts(sample_articles: object) -> None:
    """Trend detector reports yearly counts and velocity."""

    detector = TrendDetector(articles=sample_articles)
    trend = detector.detect("EGFR")

    assert trend["yearly_counts"]
    assert "publication_velocity" in trend
