"""Knowledge-graph context extraction for RAG prompts."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from pydantic import BaseModel, Field

from biomedos.core.llm_client import OllamaClient
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType


class KGContext(BaseModel):
    """Structured knowledge graph context for prompt grounding."""

    entities_found: list[str] = Field(default_factory=list)
    triples: list[str] = Field(default_factory=list)
    context_text: str = ""
    subgraph_size: int = 0


class KGContextExtractor:
    """Extract relevant knowledge graph context for RAG prompts."""

    def __init__(
        self,
        knowledge_graph: KnowledgeGraph,
        llm_client: OllamaClient | None = None,
    ) -> None:
        """Initialize the KG context extractor.

        Args:
            knowledge_graph: Source biomedical knowledge graph.
            llm_client: Optional LLM used for entity extraction fallback.
        """

        self.knowledge_graph = knowledge_graph
        self.llm_client = llm_client

    async def extract_context(
        self,
        query: str,
        max_triples: int = 50,
        depth: int = 2,
    ) -> KGContext:
        """Extract graph context for a query.

        Args:
            query: User query or claim.
            max_triples: Maximum number of triples to include.
            depth: Neighborhood depth around matched entities.

        Returns:
            Structured KG context.
        """

        entity_strings = self._match_entity_strings(query)
        if not entity_strings and self.llm_client is not None:
            entity_strings = await self._extract_entities_with_llm(query)
        return self._build_context(entity_strings, max_triples=max_triples, depth=depth)

    def build(self, query: str, *, depth: int = 2, max_triples: int = 50) -> list[str]:
        """Build a list of prompt-ready graph triples.

        Args:
            query: Query string used for node matching.
            depth: Traversal depth around matched nodes.
            max_triples: Maximum number of triples to return.

        Returns:
            Natural-language triples.
        """

        return self._build_context(
            self._match_entity_strings(query),
            max_triples=max_triples,
            depth=depth,
        ).triples

    def _build_context(
        self,
        entity_strings: list[str],
        *,
        max_triples: int,
        depth: int,
    ) -> KGContext:
        """Build a KG context object from candidate entity strings."""

        matched_ids: list[str] = []
        matched_names: list[str] = []
        for entity in entity_strings:
            for node in self._select_matching_nodes(entity):
                if node.id not in matched_ids:
                    matched_ids.append(node.id)
                    matched_names.append(node.name)

        if not matched_ids:
            return KGContext()

        included_nodes = self._expand_nodes(matched_ids, depth=depth)
        triples = self._collect_triples(included_nodes, max_triples=max_triples)
        context_text = self._format_context_text(matched_names, triples)
        return KGContext(
            entities_found=matched_names,
            triples=triples,
            context_text=context_text,
            subgraph_size=len(included_nodes),
        )

    def _select_matching_nodes(self, entity: str) -> list[Any]:
        """Select the best-matching graph nodes for an entity string."""

        matches = self.knowledge_graph.search_nodes(entity, limit=10)
        if not matches:
            return []

        normalized = entity.strip().lower()
        exact_matches = [
            node
            for node in matches
            if node.name.strip().lower() == normalized or node.id.strip().lower() == normalized
        ]
        if exact_matches:
            return exact_matches[:3]
        if " " in normalized:
            return []
        return matches[:3]

    def _match_entity_strings(self, query: str) -> list[str]:
        """Collect candidate entity strings via direct string matching."""

        candidates: list[str] = [query.strip()]
        quoted_terms = re.findall(r'"([^"]+)"', query)
        candidates.extend(quoted_terms)

        tokens = re.findall(r"[A-Za-z0-9_-]+", query)
        for size in (3, 2):
            for index in range(len(tokens) - size + 1):
                phrase = " ".join(tokens[index : index + size]).strip()
                if len(phrase) >= 6:
                    candidates.append(phrase)

        for token in tokens:
            if len(token) < 3:
                continue
            if any(character.isupper() for character in token) or any(
                character.isdigit() for character in token
            ):
                candidates.append(token)

        unique_candidates: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.strip()
            if not normalized:
                continue
            folded = normalized.lower()
            if folded not in seen:
                seen.add(folded)
                unique_candidates.append(normalized)
        return unique_candidates

    async def _extract_entities_with_llm(self, query: str) -> list[str]:
        """Use the LLM as a fallback biomedical entity extractor."""

        if self.llm_client is None:
            return []

        prompt = (
            "Extract the biomedical entities from the query below.\n"
            'Return JSON with the schema {"entities": ["..."]}.\n'
            f"Query: {query}"
        )
        try:
            response = await self.llm_client.generate(
                prompt,
                system="Return only valid JSON.",
            )
            payload = self._parse_json(response)
        except ValueError:
            return []

        entities = payload.get("entities", [])
        if not isinstance(entities, list):
            return []
        return [str(entity) for entity in entities if str(entity).strip()]

    def _expand_nodes(self, seed_ids: list[str], *, depth: int) -> set[str]:
        """Expand seed nodes into an N-hop undirected neighborhood."""

        graph = self.knowledge_graph.graph.to_undirected()
        included_nodes: set[str] = set()
        for seed_id in seed_ids:
            if seed_id not in graph:
                continue
            lengths = nx_single_source_shortest_path_length(graph, seed_id, depth)
            included_nodes.update(lengths)
        return included_nodes

    def _collect_triples(self, included_nodes: set[str], *, max_triples: int) -> list[str]:
        """Collect and rank triples from the selected subgraph."""

        ranked_edges: list[tuple[float, str]] = []
        for source_id, target_id, _, payload in self.knowledge_graph.graph.edges(
            keys=True,
            data=True,
        ):
            if source_id not in included_nodes or target_id not in included_nodes:
                continue

            source_node = self.knowledge_graph.get_node(str(source_id))
            target_node = self.knowledge_graph.get_node(str(target_id))
            edge_type = payload.get("edge_type")
            if source_node is None or target_node is None or not isinstance(edge_type, EdgeType):
                continue

            triple = f"{source_node.name} {edge_type.value} {target_node.name}"
            ranked_edges.append((float(payload.get("score", 1.0)), triple))

        ranked_edges.sort(key=lambda item: item[0], reverse=True)
        return [triple for _, triple in ranked_edges[:max_triples]]

    def _format_context_text(self, entity_names: list[str], triples: list[str]) -> str:
        """Convert raw triples into prompt-ready context text."""

        if not triples:
            return ""

        lines = [
            "Relevant knowledge graph context:",
            f"Matched entities: {', '.join(entity_names)}.",
        ]
        lines.extend(f"- {triple}" for triple in triples)
        return "\n".join(lines)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse the first JSON object from a model response."""

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)
        except json.JSONDecodeError:
            pass

        match = re.search(r"(\{.*\})", text, flags=re.DOTALL)
        if match:
            payload = json.loads(match.group(1))
            if isinstance(payload, dict):
                return cast(dict[str, Any], payload)

        msg = "Failed to parse JSON object from KG context response."
        raise ValueError(msg)


class KGContextBuilder(KGContextExtractor):
    """Backward-compatible alias for earlier prompt context builder code."""


def nx_single_source_shortest_path_length(
    graph: Any,
    source_id: str,
    depth: int,
) -> dict[str, int]:
    """Return shortest path lengths without importing NetworkX globally."""

    import networkx as nx

    return cast(
        dict[str, int],
        nx.single_source_shortest_path_length(graph, source_id, cutoff=depth),
    )
