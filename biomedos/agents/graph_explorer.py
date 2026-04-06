"""Graph exploration agent for paths, neighborhoods, and summaries."""

from __future__ import annotations

from biomedos.agents.base import BaseAgent
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.queries import GraphQueryEngine
from biomedos.orchestration.state import AgentResult, Task, WorkflowState


class GraphExplorerAgent(BaseAgent):
    """Agent that interrogates the knowledge graph and explains results."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize the graph explorer agent."""

        super().__init__(
            name="graph_explorer",
            llm_client=llm_client,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            settings=settings,
        )
        self.query_engine = GraphQueryEngine(self.knowledge_graph)

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Run graph-centric reasoning for a task."""

        del state
        payload = task.payload
        source_id = (
            str(payload.get("source_id", "")) if payload.get("source_id") is not None else ""
        )
        target_id = (
            str(payload.get("target_id", "")) if payload.get("target_id") is not None else ""
        )
        query = str(payload.get("query", task.description))
        if not source_id or not target_id:
            source_id, target_id = self._resolve_path_nodes(query, source_id, target_id)

        if source_id and target_id:
            path = self.query_engine.shortest_path(source_id, target_id)
            summary = await self._summarize_path(query, path)
            return AgentResult(
                agent_name=self.name,
                task_id=task.id,
                summary=summary,
                output={"path": path, "path_length": max(len(path) - 1, 0)},
                confidence=0.8 if path else 0.2,
                node_ids=path,
            )

        matches = self.knowledge_graph.search_nodes(query, limit=5)
        seed_ids = [node.id for node in matches]
        subgraph = self.query_engine.subgraph(seed_ids, radius=1)
        summary = await self._summarize_subgraph(query, seed_ids, subgraph.stats())
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=summary,
            output={
                "matched_nodes": [node.model_dump(mode="python") for node in matches],
                "subgraph_stats": subgraph.stats(),
            },
            confidence=0.7 if matches else 0.1,
            node_ids=seed_ids,
        )

    async def _summarize_path(self, query: str, path: list[str]) -> str:
        """Create a narrative summary for a discovered path."""

        if not path:
            return f"No graph path was found for: {query}"

        labels = [self._node_label(node_id) for node_id in path]
        narrative = " -> ".join(labels)
        if self.llm_client is None or self.should_use_fast_path():
            return (
                f"Shortest graph path for `{query}`: {narrative}. "
                f"Path length: {max(len(path) - 1, 0)}."
            )

        prompt = (
            "Summarize this biomedical graph path in 2-3 sentences and explain why it may matter.\n"
            f"User request: {query}\n"
            f"Path: {narrative}"
        )
        return await self.llm_client.generate(prompt, model=self.settings.MODEL_REASONER)

    async def _summarize_subgraph(
        self,
        query: str,
        seed_ids: list[str],
        stats: dict[str, object],
    ) -> str:
        """Create a narrative summary for a subgraph exploration result."""

        if not seed_ids:
            return f"No matching graph nodes were found for: {query}"

        if self.llm_client is None or self.should_use_fast_path():
            return (
                f"Matched {len(seed_ids)} seed nodes for `{query}` and built a local subgraph with "
                f"{stats.get('nodes', 0)} nodes."
            )

        prompt = (
            "Summarize this biomedical subgraph search result in concise research language.\n"
            f"User request: {query}\n"
            f"Seed nodes: {', '.join(seed_ids)}\n"
            f"Subgraph stats: {stats}"
        )
        return await self.llm_client.generate(prompt, model=self.settings.MODEL_REASONER)

    def _resolve_path_nodes(
        self,
        query: str,
        source_id: str,
        target_id: str,
    ) -> tuple[str, str]:
        """Resolve source and target nodes from a natural-language path query."""

        resolved_source = source_id
        resolved_target = target_id
        matches = self.extract_nodes_from_text(query, limit=4)
        if not resolved_source and matches:
            resolved_source = matches[0].id
        if not resolved_target and len(matches) >= 2:
            resolved_target = matches[1].id
        return resolved_source, resolved_target

    def _node_label(self, node_id: str) -> str:
        """Return a readable node label for summaries."""

        node = self.knowledge_graph.get_node(node_id)
        if node is None:
            return node_id
        return node.name
