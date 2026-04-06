"""Tests for community detection."""

from __future__ import annotations

from biomedos.analysis.community import CommunityDetector


def test_community_detection_returns_characterized_groups(sample_kg: object) -> None:
    """Community detector returns at least one summary."""

    detector = CommunityDetector(sample_kg)
    communities = detector.detect()
    summary = detector.characterize_community(communities[0], community_id=1)

    assert communities
    assert summary.size == len(communities[0])
