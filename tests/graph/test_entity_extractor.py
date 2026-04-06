"""Tests for the LLM-based entity extractor."""

from __future__ import annotations

import json
from typing import cast

import pytest

from biomedos.core.llm_client import OllamaClient
from biomedos.graph.entity_extractor import BioEntityExtractor


@pytest.mark.asyncio
async def test_entity_extractor_parses_json(mock_ollama: object) -> None:
    """Extractor parses entity and relation JSON from the model response."""

    payload = {
        "entities": [
            {
                "id": "gene:egfr",
                "name": "EGFR",
                "node_type": "Gene",
                "properties": {},
                "sources": ["test"],
            },
            {
                "id": "drug:gefitinib",
                "name": "Gefitinib",
                "node_type": "Drug",
                "properties": {},
                "sources": ["test"],
            },
        ],
        "relations": [
            {
                "source_id": "drug:gefitinib",
                "target_id": "gene:egfr",
                "edge_type": "drug_targets_gene",
                "score": 0.99,
                "properties": {},
                "sources": ["test"],
            }
        ],
    }
    mock_ollama.responses = [json.dumps(payload)]  # type: ignore[attr-defined]
    extractor = BioEntityExtractor(cast(OllamaClient, mock_ollama))
    result = await extractor.extract_triples("Gefitinib inhibits EGFR.")

    assert len(result.entities) == 2
    assert result.entities[0].name == "EGFR"
    assert result.relations[0].edge_type.value == "drug_targets_gene"
