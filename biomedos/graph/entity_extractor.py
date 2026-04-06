"""LLM-based biomedical entity and relation extraction."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from pydantic import BaseModel, Field

from biomedos.config import Settings, get_settings
from biomedos.core.llm_client import OllamaClient
from biomedos.graph.schema import BioEdge, BioNode, EdgeType, NodeType, create_node_model


class ExtractionResult(BaseModel):
    """Structured output from the entity extractor."""

    entities: list[BioNode] = Field(default_factory=list)
    relations: list[BioEdge] = Field(default_factory=list)


class BioEntityExtractor:
    """Use an LLM to extract biomedical graph triples from free text."""

    def __init__(self, llm_client: OllamaClient, settings: Settings | None = None) -> None:
        """Initialize the entity extractor."""

        self.llm_client = llm_client
        self.settings = settings or get_settings()

    async def extract_entities(self, text: str) -> list[BioNode]:
        """Extract biomedical entities from text."""

        result = await self.extract_triples(text)
        return result.entities

    async def extract_relations(self, text: str) -> list[BioEdge]:
        """Extract biomedical relations from text."""

        result = await self.extract_triples(text)
        return result.relations

    async def extract_triples(self, text: str) -> ExtractionResult:
        """Extract entities and relations from text using the extractor model."""

        prompt = self._build_prompt(text)
        response = await self.llm_client.generate(
            prompt,
            model=self.settings.MODEL_EXTRACTOR,
            system="Return only valid JSON.",
        )
        payload = self._parse_json_payload(response)

        entity_payloads = payload.get("entities", [])
        relation_payloads = payload.get("relations", [])

        entities: list[BioNode] = []
        if isinstance(entity_payloads, list):
            for item in entity_payloads:
                if not isinstance(item, dict):
                    continue
                entities.append(self._build_entity(item))

        relations: list[BioEdge] = []
        if isinstance(relation_payloads, list):
            for item in relation_payloads:
                if not isinstance(item, dict):
                    continue
                relations.append(self._build_relation(item))

        return ExtractionResult(entities=entities, relations=relations)

    def _build_prompt(self, text: str) -> str:
        """Build a constrained extraction prompt."""

        node_types = ", ".join(item.value for item in NodeType)
        edge_types = ", ".join(item.value for item in EdgeType)
        return (
            "Extract biomedical entities and relations from the text below.\n"
            "Return JSON with keys `entities` and `relations`.\n"
            f"Allowed node types: {node_types}.\n"
            f"Allowed edge types: {edge_types}.\n"
            "Each entity must contain id, name, node_type, properties, and sources.\n"
            "Each relation must contain source_id, target_id, edge_type, score,\n"
            "properties, and sources.\n"
            f"Text:\n{text}"
        )

    def _build_entity(self, payload: dict[str, Any]) -> BioNode:
        """Validate and normalize a raw entity payload."""

        raw_type = payload.get("node_type", NodeType.GENE.value)
        node_type = NodeType(str(raw_type))
        node_id = str(
            payload.get("id") or self._slugify(node_type.value, str(payload.get("name", "")))
        )
        return create_node_model(
            {
                "id": node_id,
                "name": str(payload.get("name", node_id)),
                "node_type": node_type,
                "properties": payload.get("properties", {}),
                "sources": payload.get("sources", ["llm_extraction"]),
            }
        )

    def _build_relation(self, payload: dict[str, Any]) -> BioEdge:
        """Validate and normalize a raw relation payload."""

        edge_type = EdgeType(str(payload.get("edge_type", EdgeType.GENE_GENE.value)))
        return BioEdge(
            source_id=str(payload.get("source_id", "")),
            target_id=str(payload.get("target_id", "")),
            edge_type=edge_type,
            score=float(payload.get("score", 1.0)),
            properties=payload.get("properties", {}),
            sources=payload.get("sources", ["llm_extraction"]),
        )

    def _parse_json_payload(self, text: str) -> dict[str, Any]:
        """Extract the first JSON object from a model response."""

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)
        except json.JSONDecodeError:
            pass

        fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fence_match:
            payload = json.loads(fence_match.group(1))
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)

        object_match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if object_match:
            payload = json.loads(object_match.group(1))
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)

        msg = "Failed to parse JSON payload from extractor response."
        raise ValueError(msg)

    @staticmethod
    def _slugify(prefix: str, name: str) -> str:
        """Create a deterministic node identifier from free text."""

        cleaned = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
        return f"{prefix.lower()}:{cleaned}" if cleaned else f"{prefix.lower()}:unknown"
