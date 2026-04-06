"""Run the BioMedOS API with explicitly constructed settings."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn  # type: ignore[import-not-found]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biomedos.api.app import create_app
from biomedos.config import Settings, resolve_project_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Run the BioMedOS local API server.")
    parser.add_argument("--host", default=None, help="Optional host override.")
    parser.add_argument("--port", type=int, default=None, help="Optional port override.")
    parser.add_argument("--graph", default=None, help="Optional graph path override.")
    parser.add_argument(
        "--full-quality",
        action="store_true",
        help="Disable fast local mode and use the configured full LLM stack.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the API server."""

    args = parse_args()
    settings = Settings()
    updates: dict[str, object] = {}
    if args.graph is not None:
        updates["GRAPH_PERSIST_PATH"] = str(resolve_project_path(args.graph))
    elif not settings.graph_path().exists():
        demo_graph_path = resolve_project_path("data/demo_knowledge_graph.gpickle")
        if demo_graph_path.exists():
            updates["GRAPH_PERSIST_PATH"] = str(demo_graph_path)
    if not args.full_quality and not settings.FAST_LOCAL_MODE:
        updates["FAST_LOCAL_MODE"] = True
    if updates:
        settings = settings.model_copy(update=updates)
    app = create_app(settings=settings)
    uvicorn.run(
        app,
        host=args.host or settings.API_HOST,
        port=args.port or settings.API_PORT,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
