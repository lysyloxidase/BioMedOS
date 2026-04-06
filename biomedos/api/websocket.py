"""WebSocket chat helpers."""

from __future__ import annotations

import json

from fastapi import WebSocket, WebSocketDisconnect

from biomedos.agents.router import RouterAgent
from biomedos.orchestration.workflow import BiomedicalWorkflow


class ChatWebSocketHandler:
    """Handle WebSocket chat interactions."""

    def __init__(self, workflow: BiomedicalWorkflow, router: RouterAgent) -> None:
        """Initialize the WebSocket handler."""

        self.workflow = workflow
        self.router = router

    async def handle(self, websocket: WebSocket) -> None:
        """Serve a bi-directional chat session."""

        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_text()
                query = self._extract_query(message)
                task_type = await self.router.classify_task(query)
                await websocket.send_json(
                    {
                        "event": "routing",
                        "type": "routing",
                        "task_type": task_type.value,
                        "agents": [task_type.value],
                        "message": f"Routing query to {task_type.value}.",
                    }
                )

                try:
                    state = await self.workflow.run(query)
                except Exception as exc:  # pragma: no cover - runtime network/model failures
                    await websocket.send_json(
                        {
                            "event": "error",
                            "type": "error",
                            "task_type": task_type.value,
                            "message": str(exc),
                        }
                    )
                    continue

                answer = state.final_response or ""
                await websocket.send_json(
                    {
                        "event": "metadata",
                        "type": "metadata",
                        "task_type": task_type.value,
                        "agents": state.visited_agents,
                        "citations": state.citations,
                        "message": "Workflow metadata ready.",
                    }
                )
                for chunk in self._chunk_text(answer):
                    await websocket.send_json(
                        {
                            "event": "chunk",
                            "type": "chunk",
                            "content": chunk,
                        }
                    )
                await websocket.send_json(
                    {
                        "event": "done",
                        "type": "done",
                        "task_type": task_type.value,
                        "agents": state.visited_agents,
                        "citations": state.citations,
                        "answer": answer,
                        "message": "Workflow completed.",
                        "results": {
                            task_id: {
                                "agent_name": result.agent_name,
                                "summary": result.summary,
                                "confidence": result.confidence,
                            }
                            for task_id, result in state.results.items()
                        },
                    }
                )
        except WebSocketDisconnect:
            return

    def _extract_query(self, message: str) -> str:
        """Extract a query string from raw or JSON websocket input."""

        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            return message
        if isinstance(payload, dict) and payload.get("query") is not None:
            return str(payload["query"])
        return message

    def _chunk_text(self, text: str, *, chunk_size: int = 180) -> list[str]:
        """Split a message into UI-friendly streaming chunks."""

        if not text:
            return [""]
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]
