"""Verification and hallucination detection agent."""

from __future__ import annotations

import json
import re
from typing import Any, cast

from biomedos.agents.base import BaseAgent
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import BioNode
from biomedos.orchestration.state import AgentResult, Task, WorkflowState


class SentinelAgent(BaseAgent):
    """Verify claims and detect unsupported reasoning."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the sentinel agent."""

        super().__init__(
            name="sentinel",
            llm_client=llm_client,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            settings=settings,
        )

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the verification agent."""

        payload_results = task.payload.get("results")
        candidate_results = payload_results if isinstance(payload_results, list) else []
        if not candidate_results and state is not None:
            candidate_results = [
                result.model_dump(mode="python")
                for result in state.results.values()
                if result.agent_name not in {self.name, "router"}
            ]

        issues: list[str] = []
        checked_claims = 0
        supported_claims = 0
        for result in candidate_results:
            if not isinstance(result, dict):
                continue
            content = str(result.get("summary", result.get("content", "")))
            checked_claims += len(self._extract_claims(content))
            supported, found_issues = self._verify_text(content)
            supported_claims += supported
            issues.extend(found_issues)
            invalid_citations = result.get("output", {})
            if isinstance(invalid_citations, dict):
                invalid = invalid_citations.get("invalid_citations", [])
                if isinstance(invalid, list):
                    issues.extend(f"Invalid citation: {item}" for item in invalid)

        llm_issues = await self._llm_consistency_check(candidate_results)
        issues.extend(llm_issues)
        unique_issues = list(dict.fromkeys(issues))
        confidence = self._confidence(checked_claims, supported_claims, unique_issues)
        summary = (
            f"Sentinel confidence: {confidence:.2f}. "
            f"Checked {checked_claims} claims and found {len(unique_issues)} issues."
        )
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=summary,
            output={
                "checked_claims": checked_claims,
                "supported_claims": supported_claims,
                "issues": unique_issues,
            },
            confidence=confidence,
            errors=unique_issues,
        )

    def _verify_text(self, text: str) -> tuple[int, list[str]]:
        """Verify claim text against the local knowledge graph."""

        claims = self._extract_claims(text)
        supported = 0
        issues: list[str] = []
        for claim in claims:
            entities = self._resolve_claim_entities(claim)
            if len(entities) < 2:
                if self._claim_has_literature_support(claim):
                    supported += 1
                    continue
                if not self._has_citation_marker(claim):
                    issues.append(f"Insufficiently grounded claim: {claim}")
                continue
            source = entities[0]
            target = entities[1]
            if self.knowledge_graph.get_edges(source_id=source.id, target_id=target.id):
                supported += 1
                continue
            if self.knowledge_graph.get_edges(source_id=target.id, target_id=source.id):
                supported += 1
                continue
            if self._claim_has_literature_support(claim):
                supported += 1
                continue
            issues.append(f"No direct KG evidence for claim: {claim}")
        return supported, issues

    async def _llm_consistency_check(self, results: list[object]) -> list[str]:
        """Use the verifier model for a coarse consistency pass."""

        if self.llm_client is None or not results or self.should_use_fast_path():
            return []
        prompt = (
            "Review the following biomedical agent outputs for contradictions or unsupported "
            "claims.\n"
            'Return JSON with the schema {"issues": ["..."]}.\n'
            f"Outputs: {json.dumps(results)[:4000]}"
        )
        try:
            response = await self.llm_client.generate(
                prompt,
                model=self.settings.MODEL_VERIFIER,
                system="Return only valid JSON.",
            )
            payload = self._parse_json(response)
        except ValueError:
            return []
        issues = payload.get("issues", [])
        if not isinstance(issues, list):
            return []
        return [str(issue) for issue in issues if str(issue).strip()]

    def _extract_claims(self, text: str) -> list[str]:
        """Extract simple sentence-level claims."""

        return [claim.strip() for claim in re.split(r"(?<=[.!?])\s+", text) if claim.strip()]

    def _resolve_claim_entities(self, claim: str) -> list[BioNode]:
        """Resolve graph entities mentioned in a claim."""

        entities: list[BioNode] = []
        for node_id, payload in self.knowledge_graph.graph.nodes(data=True):
            name = str(payload.get("name", ""))
            if name and name.lower() in claim.lower():
                node = self.knowledge_graph.get_node(str(node_id))
                if node is not None:
                    entities.append(node)
        return entities[:2]

    def _claim_has_literature_support(self, claim: str) -> bool:
        """Check whether local literature retrieval supports a claim."""

        if self.vector_store is None or self.vector_store.count() == 0:
            return False
        results = self.vector_store.hybrid_search(claim, top_k=2)
        if not results:
            return False
        best = results[0]
        claim_terms = set(re.findall(r"[A-Za-z0-9_:-]+", claim.lower()))
        doc_terms = set(re.findall(r"[A-Za-z0-9_:-]+", best.text.lower()))
        return len(claim_terms & doc_terms) >= 2

    def _has_citation_marker(self, claim: str) -> bool:
        """Return whether a claim contains an inline citation marker."""

        return bool(re.search(r"\[[^\]]+\]", claim))

    def _confidence(self, checked_claims: int, supported_claims: int, issues: list[str]) -> float:
        """Estimate verification confidence."""

        if checked_claims == 0:
            return 0.3 if issues else 0.6
        support_ratio = supported_claims / checked_claims
        penalty = min(len(issues) * 0.1, 0.5)
        return max(0.0, min(0.95, 0.35 + 0.55 * support_ratio - penalty))

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any]:
        """Parse the first JSON object from a verifier response."""

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
        msg = "Sentinel response did not contain valid JSON."
        raise ValueError(msg)
