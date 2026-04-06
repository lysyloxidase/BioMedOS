"""FastAPI entrypoint for BioMedOS."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from biomedos.agents.clinician import ClinicianAgent
from biomedos.agents.graph_explorer import GraphExplorerAgent
from biomedos.agents.link_predictor import LinkPredictorAgent
from biomedos.agents.pathway_analyst import PathwayAnalystAgent
from biomedos.agents.pharmacologist import PharmacologistAgent
from biomedos.agents.review_writer import ReviewWriterAgent
from biomedos.agents.router import RouterAgent
from biomedos.analysis.centrality import DrugTargetRanker
from biomedos.analysis.community import CommunityDetector
from biomedos.api.websocket import ChatWebSocketHandler
from biomedos.clinical.evidence_levels import GradeEvidenceClassifier
from biomedos.config import Settings, get_settings, resolve_project_path
from biomedos.core.llm_client import OllamaClient
from biomedos.core.vector_store import ChromaVectorStore, VectorDocument
from biomedos.demo_data import build_demo_graph, demo_articles
from biomedos.graph.builder import KnowledgeGraph
from biomedos.graph.schema import EdgeType, NodeType
from biomedos.orchestration.state import AgentResult, Task, TaskType, WorkflowState
from biomedos.orchestration.workflow import BiomedicalWorkflow
from biomedos.viz.graph_renderer import GraphRenderer


class AskRequest(BaseModel):
    """Request model for the `/ask` endpoint."""

    query: str


class LinkPredictionRequest(BaseModel):
    """Request model for link prediction."""

    source: str
    target_type: str = NodeType.DISEASE.value
    model: str = "graphsage"
    top_k: int = 10
    edge_type: str | None = None
    epochs: int | None = None


class DDIRequest(BaseModel):
    """Request model for pharmacology checks."""

    drugs: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)


class DiagnosisRequest(BaseModel):
    """Request model for phenotype and diagnosis workflows."""

    symptoms: str


class AnalysisRequest(BaseModel):
    """Request model for gene-set analysis."""

    genes: list[str] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    """Request model for narrative review generation."""

    topic: str


def _load_or_create_graph(settings: Settings) -> KnowledgeGraph:
    """Load the persisted graph if present."""

    graph_path = _resolved_graph_path(settings)
    if graph_path.exists():
        return KnowledgeGraph.load(graph_path)
    if graph_path.name == "demo_knowledge_graph.gpickle":
        return build_demo_graph(settings.DEFAULT_GENES[:20])
    return KnowledgeGraph()


def _resolved_graph_path(settings: Settings) -> Path:
    """Resolve the most useful graph path for local startup."""

    graph_path = settings.graph_path()
    if graph_path.exists():
        return graph_path
    demo_graph_path = resolve_project_path("data/demo_knowledge_graph.gpickle")
    if graph_path.name == "knowledge_graph.gpickle":
        return demo_graph_path
    return graph_path


def _ui_index_path() -> Path:
    """Return the path to the SPA entrypoint."""

    return resolve_project_path("web/index.html")


def _result_payload(result: AgentResult) -> dict[str, object]:
    """Serialize an agent result for API consumers."""

    return {
        "agent_name": result.agent_name,
        "task_id": result.task_id,
        "summary": result.summary,
        "output": result.output,
        "citations": result.citations,
        "confidence": result.confidence,
        "errors": result.errors,
    }


def _seed_demo_vector_store(vector_store: ChromaVectorStore) -> None:
    """Preload the bundled demo literature corpus into an empty vector store."""

    if vector_store.count() > 0:
        return

    documents = [
        VectorDocument(
            id=f"pmid:{article.pmid}",
            text="\n\n".join(part for part in (article.title, article.abstract) if part),
            metadata={
                "pmid": article.pmid,
                "title": article.title,
                "journal": article.journal or "",
                "year": article.year or 0,
                "authors": "; ".join(article.authors),
                "source": "demo",
            },
        )
        for article in demo_articles()
    ]
    vector_store.add_documents(documents)


def _workflow_payload(state: WorkflowState) -> dict[str, object]:
    """Serialize a workflow state into a compact API response."""

    primary_task = state.tasks[0].type.value if state.tasks else TaskType.ROUTER.value
    return {
        "task_type": primary_task,
        "answer": state.final_response or "",
        "citations": state.citations,
        "agents": state.visited_agents,
        "tasks": [task.model_dump(mode="python") for task in state.tasks],
        "results": {task_id: _result_payload(result) for task_id, result in state.results.items()},
    }


def create_app(
    settings: Settings | None = None,
    *,
    knowledge_graph: KnowledgeGraph | None = None,
    llm_client: OllamaClient | None = None,
) -> FastAPI:
    """Create the FastAPI application."""

    resolved_settings = settings or get_settings()
    resolved_graph_path = (
        resolved_settings.graph_path()
        if knowledge_graph is not None
        else _resolved_graph_path(resolved_settings)
    )
    graph = knowledge_graph or _load_or_create_graph(resolved_settings)
    llm = llm_client or OllamaClient(
        base_url=resolved_settings.OLLAMA_HOST,
        settings=resolved_settings,
    )
    vector_store = ChromaVectorStore(
        persist_dir=(
            ":memory:"
            if resolved_settings.CHROMA_PERSIST_DIR == ":memory:"
            else str(resolved_settings.chroma_path())
        ),
        collection_name=resolved_settings.CHROMA_COLLECTION,
        settings=resolved_settings,
    )
    if resolved_graph_path.name == "demo_knowledge_graph.gpickle":
        _seed_demo_vector_store(vector_store)

    router = RouterAgent(
        llm_client=llm,
        knowledge_graph=graph,
        vector_store=vector_store,
        settings=resolved_settings,
    )
    graph_explorer = GraphExplorerAgent(
        llm_client=llm,
        knowledge_graph=graph,
        vector_store=vector_store,
        settings=resolved_settings,
    )
    link_predictor = LinkPredictorAgent(
        llm_client=llm,
        knowledge_graph=graph,
        vector_store=vector_store,
        settings=resolved_settings,
    )
    pharmacologist = PharmacologistAgent(
        llm_client=llm,
        knowledge_graph=graph,
        vector_store=vector_store,
        settings=resolved_settings,
    )
    clinician = ClinicianAgent(
        llm_client=llm,
        knowledge_graph=graph,
        vector_store=vector_store,
        settings=resolved_settings,
    )
    pathway_analyst = PathwayAnalystAgent(
        llm_client=llm,
        knowledge_graph=graph,
        vector_store=vector_store,
        settings=resolved_settings,
    )
    review_writer = ReviewWriterAgent(
        llm_client=llm,
        knowledge_graph=graph,
        vector_store=vector_store,
        settings=resolved_settings,
    )
    workflow = BiomedicalWorkflow(
        llm_client=llm,
        knowledge_graph=graph,
        vector_store=vector_store,
        settings=resolved_settings,
        router=router,
        agents={
            TaskType.GRAPH_EXPLORER: graph_explorer,
            TaskType.LINK_PREDICTOR: link_predictor,
            TaskType.PHARMACOLOGIST: pharmacologist,
            TaskType.CLINICIAN: clinician,
            TaskType.PATHWAY_ANALYST: pathway_analyst,
            TaskType.REVIEW_WRITER: review_writer,
        },
    )
    websocket_handler = ChatWebSocketHandler(workflow=workflow, router=router)
    renderer = GraphRenderer()
    target_ranker = DrugTargetRanker(graph)
    community_detector = CommunityDetector(graph)
    evidence_classifier = GradeEvidenceClassifier()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        """Manage application startup and shutdown resources."""

        try:
            yield
        finally:
            await llm.close()

    app = FastAPI(title="BioMedOS", version="0.1.0", lifespan=lifespan)
    app.state.settings = resolved_settings
    app.state.knowledge_graph = graph
    app.state.llm_client = llm
    app.state.vector_store = vector_store
    app.state.router = router
    app.state.graph_explorer = graph_explorer
    app.state.link_predictor = link_predictor
    app.state.pharmacologist = pharmacologist
    app.state.clinician = clinician
    app.state.pathway_analyst = pathway_analyst
    app.state.review_writer = review_writer
    app.state.workflow = workflow
    app.state.websocket_handler = websocket_handler
    app.state.renderer = renderer
    app.state.target_ranker = target_ranker
    app.state.community_detector = community_detector
    app.state.evidence_classifier = evidence_classifier

    @app.get("/")
    async def index() -> HTMLResponse:
        """Serve the BioMedOS SPA."""

        return HTMLResponse(_ui_index_path().read_text(encoding="utf-8"))

    @app.get("/health")
    async def health() -> dict[str, object]:
        """Report API, graph, and Ollama health."""

        return {
            "status": "ok",
            "ollama": await app.state.llm_client.health_check(),
            "graph": app.state.knowledge_graph.stats(),
            "vector_store_docs": app.state.vector_store.count(),
            "graph_path": str(resolved_graph_path),
            "chroma_persist_dir": (
                resolved_settings.CHROMA_PERSIST_DIR
                if resolved_settings.CHROMA_PERSIST_DIR == ":memory:"
                else str(resolved_settings.chroma_path())
            ),
            "fast_local_mode": resolved_settings.FAST_LOCAL_MODE,
        }

    @app.get("/graph/stats")
    async def graph_stats() -> dict[str, object]:
        """Return graph summary statistics."""

        return app.state.knowledge_graph.stats()

    @app.get("/graph/network")
    async def graph_network() -> dict[str, object]:
        """Return the full graph as vis-network-compatible JSON."""

        payload = app.state.renderer.to_vis_payload(app.state.knowledge_graph)
        payload["node_types"] = [node_type.value for node_type in NodeType]
        payload["edge_types"] = [edge_type.value for edge_type in EdgeType]
        return payload

    @app.get("/graph/search")
    async def graph_search(
        q: str = Query(..., min_length=1),
        node_type: str | None = None,
    ) -> list[dict[str, object]]:
        """Search graph nodes by text."""

        parsed_type = NodeType(node_type) if node_type is not None else None
        matches = app.state.knowledge_graph.search_nodes(q, node_type=parsed_type, limit=20)
        return [match.model_dump(mode="python") for match in matches]

    @app.get("/graph/node/{node_id}")
    async def graph_node(node_id: str) -> dict[str, object]:
        """Return a node, its edges, and its immediate neighbors."""

        node = app.state.knowledge_graph.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found.")
        edges = app.state.knowledge_graph.get_edges(
            source_id=node_id
        ) + app.state.knowledge_graph.get_edges(target_id=node_id)
        neighbor_ids = {
            edge.target_id if edge.source_id == node_id else edge.source_id for edge in edges
        }
        neighbors = [
            neighbor.model_dump(mode="python")
            for neighbor_id in sorted(neighbor_ids)
            if (neighbor := app.state.knowledge_graph.get_node(neighbor_id)) is not None
        ]
        return {
            "node": node.model_dump(mode="python"),
            "neighbors": neighbors,
            "edges": [edge.model_dump(mode="python") for edge in edges],
        }

    @app.get("/graph/neighbors/{node_id}")
    async def graph_neighbors(node_id: str) -> dict[str, object]:
        """Return one-hop neighbor nodes and connecting edges."""

        node = app.state.knowledge_graph.get_node(node_id)
        if node is None:
            raise HTTPException(status_code=404, detail="Node not found.")
        payload = await graph_node(node_id)
        return payload

    @app.get("/graph/paths")
    async def graph_paths(source: str, target: str) -> dict[str, object]:
        """Return the shortest graph path between two nodes."""

        path = app.state.graph_explorer.query_engine.shortest_path(source, target)
        return {"path": path, "length": max(len(path) - 1, 0)}

    @app.post("/ask")
    async def ask(request: AskRequest) -> dict[str, object]:
        """Route a question through the full BioMedOS workflow."""

        state = await app.state.workflow.run(request.query)
        return _workflow_payload(state)

    @app.post("/predict")
    async def predict(request: LinkPredictionRequest) -> dict[str, object]:
        """Run graph link prediction for the Predictions tab."""

        payload: dict[str, object] = {
            "source": request.source,
            "target_type": request.target_type,
            "model": request.model,
            "top_k": request.top_k,
        }
        if request.edge_type is not None:
            payload["edge_type"] = request.edge_type
        if request.epochs is not None:
            payload["epochs"] = request.epochs
        result = await app.state.link_predictor.run(
            Task(
                id="predict-task",
                type=TaskType.LINK_PREDICTOR,
                description=request.source,
                payload=payload,
            )
        )
        return _result_payload(result)

    @app.post("/clinical/ddi")
    async def clinical_ddi(request: DDIRequest) -> dict[str, object]:
        """Run local pharmacology and DDI analysis."""

        result = await app.state.pharmacologist.run(
            Task(
                id="ddi-task",
                type=TaskType.PHARMACOLOGIST,
                description=", ".join(request.drugs),
                payload={"drugs": request.drugs, "conditions": request.conditions},
            )
        )
        return _result_payload(result)

    @app.post("/clinical/phenotypes")
    async def clinical_phenotypes(request: DiagnosisRequest) -> dict[str, object]:
        """Map symptoms to phenotypes and disease matches."""

        matched_terms = await app.state.clinician.phenotype_matcher.map_to_hpo(request.symptoms)
        diseases = await app.state.clinician.phenotype_matcher.match(request.symptoms, top_k=10)
        assessments = [
            app.state.evidence_classifier.classify(
                f"Observational phenotype evidence for {disease.disease_name}"
            ).model_dump(mode="python")
            for disease in diseases[:3]
        ]
        return {
            "matched_terms": [item.model_dump(mode="python") for item in matched_terms],
            "disease_matches": [item.model_dump(mode="python") for item in diseases],
            "evidence_levels": assessments,
        }

    @app.post("/clinical/differential")
    async def clinical_differential(request: DiagnosisRequest) -> dict[str, object]:
        """Run differential diagnosis over phenotype input."""

        result = await app.state.clinician.run(
            Task(
                id="clinician-task",
                type=TaskType.CLINICIAN,
                description=request.symptoms,
                payload={"symptoms": request.symptoms},
            )
        )
        return _result_payload(result)

    @app.post("/analysis/enrichment")
    async def analysis_enrichment(request: AnalysisRequest) -> dict[str, object]:
        """Run pathway enrichment and crosstalk analysis."""

        result = await app.state.pathway_analyst.run(
            Task(
                id="analysis-task",
                type=TaskType.PATHWAY_ANALYST,
                description=", ".join(request.genes),
                payload={"genes": request.genes},
            )
        )
        return _result_payload(result)

    @app.get("/analysis/communities")
    async def analysis_communities() -> dict[str, object]:
        """Return characterized graph communities."""

        communities = app.state.community_detector.detect()
        communities.sort(key=len, reverse=True)
        summaries = [
            app.state.community_detector.characterize_community(
                members,
                community_id=index,
            ).model_dump(mode="python")
            for index, members in enumerate(communities[:10], start=1)
        ]
        return {"communities": summaries}

    @app.get("/analysis/targets")
    async def analysis_targets(top_k: int = Query(20, ge=1, le=100)) -> dict[str, object]:
        """Return ranked gene targets by graph topology."""

        rankings = app.state.target_ranker.rank_targets(top_k=top_k)
        return {"targets": [item.model_dump(mode="python") for item in rankings]}

    @app.post("/review")
    async def review(request: ReviewRequest) -> dict[str, object]:
        """Generate a narrative review."""

        result = await app.state.review_writer.run(
            Task(
                id="review-task",
                type=TaskType.REVIEW_WRITER,
                description=request.topic,
                payload={"query": request.topic},
            )
        )
        return _result_payload(result)

    @app.get("/viz")
    async def viz() -> HTMLResponse:
        """Render the current knowledge graph as HTML."""

        html = app.state.renderer.render_html(app.state.knowledge_graph)
        return HTMLResponse(content=html)

    @app.websocket("/ws/chat")
    async def websocket_chat(websocket: WebSocket) -> None:
        """Serve the streaming chat websocket."""

        await app.state.websocket_handler.handle(websocket)

    return app


app = create_app()
