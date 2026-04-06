"""Tests for the phenotype matcher."""

from __future__ import annotations

from typing import cast

import pytest

from biomedos.clinical.phenotype_matcher import PhenotypeMatcher
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient


@pytest.mark.asyncio
async def test_phenotype_matcher_maps_terms_and_ranks_diseases(
    mock_ollama: object,
    sample_kg: object,
) -> None:
    """Symptoms can be normalized to HPO terms and matched to diseases."""

    mock_ollama.responses = ['{"terms": ["chronic cough", "dyspnea"]}']  # type: ignore[attr-defined]
    matcher = PhenotypeMatcher(
        llm_client=cast(OllamaClient, mock_ollama),
        knowledge_graph=sample_kg,
    )

    matched_terms = await matcher.map_to_hpo("persistent cough and breathlessness")
    diseases = await matcher.match("persistent cough and breathlessness", top_k=3)

    assert {term.hpo_id for term in matched_terms} >= {"HP:0012735", "HP:0002094"}
    assert diseases[0].disease_name == "Lung Cancer"
    assert {disease.disease_name for disease in diseases[:2]} >= {"Lung Cancer", "Fibrosis"}


@pytest.mark.asyncio
async def test_phenotype_matcher_fast_local_mode_skips_llm_and_uses_local_terms(
    mock_ollama: object,
    sample_kg: object,
) -> None:
    """Fast local mode should avoid LLM normalization and remain fully local."""

    matcher = PhenotypeMatcher(
        llm_client=cast(OllamaClient, mock_ollama),
        knowledge_graph=sample_kg,
        settings=Settings(FAST_LOCAL_MODE=True),
    )

    matched_terms = await matcher.map_to_hpo("persistent cough, dyspnea, weight loss")

    assert {term.hpo_id for term in matched_terms} >= {"HP:0012735", "HP:0002094", "HP:0001824"}
