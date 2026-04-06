"""Link prediction specialist agent."""

from __future__ import annotations

from biomedos.agents.base import BaseAgent
from biomedos.config import Settings
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType
from biomedos.ml.link_prediction import LinkPredictionPipeline, PredictedLink
from biomedos.orchestration.state import AgentResult, Task, WorkflowState


class LinkPredictorAgent(BaseAgent):
    """Infer plausible novel graph relationships."""

    def __init__(
        self,
        *,
        llm_client: OllamaClient | None = None,
        knowledge_graph: KnowledgeGraph | None = None,
        vector_store: ChromaVectorStore | None = None,
        settings: Settings | None = None,
        pipeline: LinkPredictionPipeline | None = None,
    ) -> None:
        """Initialize the link prediction agent."""

        super().__init__(
            name="link_predictor",
            llm_client=llm_client,
            knowledge_graph=knowledge_graph,
            vector_store=vector_store,
            settings=settings,
        )
        self.pipeline = pipeline

    async def run(self, task: Task, state: WorkflowState | None = None) -> AgentResult:
        """Execute the link predictor agent."""

        del state
        source_query = str(
            task.payload.get("source_id", task.payload.get("source", task.description))
        )
        target_type_raw = str(task.payload.get("target_type", NodeType.DISEASE.value))
        resolved_target_type = NodeType(target_type_raw)
        source_node = self.resolve_node(source_query)
        if source_node is None:
            return AgentResult(
                agent_name=self.name,
                task_id=task.id,
                summary=f"Could not resolve source node: {source_query}",
                errors=[f"Unknown source node: {source_query}"],
            )

        relation = self._resolve_edge_type(task, source_node.node_type, resolved_target_type)
        pipeline = self.pipeline or LinkPredictionPipeline(
            self.knowledge_graph,
            model_name=str(task.payload.get("model", "graphsage")),
            edge_type=relation,
            settings=self.settings,
        )
        epochs = self.coerce_int(task.payload.get("epochs"), min(self.settings.EPOCHS, 30))
        training_summary = pipeline.train(epochs=epochs)
        predictions = pipeline.predict_novel_links(
            source_node.id,
            resolved_target_type,
            top_k=self.coerce_int(task.payload.get("top_k"), 10),
            edge_type=relation,
        )
        summary = self._format_summary(source_node.name, predictions, relation)
        return AgentResult(
            agent_name=self.name,
            task_id=task.id,
            summary=summary,
            output={
                "predictions": [prediction.model_dump(mode="python") for prediction in predictions],
                "metrics": training_summary.metrics.model_dump(mode="python"),
                "model_name": training_summary.model_name,
                "edge_type": relation.value,
            },
            confidence=min(
                0.95,
                training_summary.metrics.auprc * 0.5 + training_summary.metrics.mrr * 0.4 + 0.2,
            ),
        )

    def _resolve_edge_type(
        self,
        task: Task,
        source_type: NodeType,
        target_type: NodeType,
    ) -> EdgeType:
        """Resolve the relation type for prediction."""

        raw_edge_type = task.payload.get("edge_type")
        if isinstance(raw_edge_type, str):
            return EdgeType(raw_edge_type)
        inferred = self.infer_edge_type(source_type, target_type)
        return inferred or EdgeType.GENE_DISEASE

    def _format_summary(
        self,
        source_name: str,
        predictions: list[PredictedLink],
        edge_type: EdgeType,
    ) -> str:
        """Build a concise prediction narrative."""

        if not predictions:
            return f"No novel {edge_type.value} candidates were identified for {source_name}."
        lines = [
            f"Top novel {edge_type.value} candidates for {source_name}:",
            *[
                f"- {prediction.target_name} ({prediction.score:.3f})"
                for prediction in predictions[:10]
            ],
        ]
        return "\n".join(lines)
