"""Tests for the FastAPI application entrypoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from biomedos.api.app import create_app
from biomedos.config import Settings
from biomedos.graph.builder import KnowledgeGraph


class StubOllamaClient:
    """Minimal async Ollama stub for API tests."""

    async def health_check(self) -> bool:
        """Return a healthy status without external network access."""

        return True

    async def close(self) -> None:
        """Close the stub client."""


def test_health_endpoint_reports_absolute_paths(sample_kg: KnowledgeGraph) -> None:
    """Health payload should expose resolved local paths for debugging."""

    app = create_app(
        settings=Settings(
            GRAPH_PERSIST_PATH="data/knowledge_graph.gpickle",
            CHROMA_PERSIST_DIR=":memory:",
            FAST_LOCAL_MODE=True,
        ),
        knowledge_graph=sample_kg,
        llm_client=StubOllamaClient(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["fast_local_mode"] is True
    assert payload["graph_path"].endswith("data\\knowledge_graph.gpickle")
    assert payload["vector_store_docs"] == 0


def test_app_falls_back_to_demo_graph_when_default_graph_is_missing() -> None:
    """API should load the bundled demo graph if the default graph path does not exist."""

    app = create_app(
        settings=Settings(
            GRAPH_PERSIST_PATH="data/knowledge_graph.gpickle",
            CHROMA_PERSIST_DIR=":memory:",
            FAST_LOCAL_MODE=True,
        ),
        llm_client=StubOllamaClient(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["graph"]["nodes"] > 0
    assert payload["graph"]["edges"] > 0
    assert payload["vector_store_docs"] >= 6
    assert payload["graph_path"].endswith("data\\demo_knowledge_graph.gpickle")


def test_websocket_chat_emits_type_compatible_events(sample_kg: KnowledgeGraph) -> None:
    """WebSocket chat should emit `type` fields understood by the SPA."""

    app = create_app(
        settings=Settings(
            GRAPH_PERSIST_PATH="data/knowledge_graph.gpickle",
            CHROMA_PERSIST_DIR=":memory:",
            FAST_LOCAL_MODE=True,
        ),
        knowledge_graph=sample_kg,
        llm_client=StubOllamaClient(),  # type: ignore[arg-type]
    )

    with TestClient(app) as client:
        with client.websocket_connect("/ws/chat") as websocket:
            websocket.send_json({"query": "Show the graph path between EGFR and lung cancer."})
            routing = websocket.receive_json()
            metadata = websocket.receive_json()
            chunk = websocket.receive_json()
            done = chunk
            while done["type"] != "done":
                done = websocket.receive_json()

    assert routing["type"] == "routing"
    assert metadata["type"] == "metadata"
    assert chunk["type"] == "chunk"
    assert done["type"] == "done"
