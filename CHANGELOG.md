# Changelog

## 0.4.0 - 2026-04-06

- Added the final production-facing SPA in `web/index.html` with graph exploration, chat, prediction, clinical, analysis, and review tabs.
- Expanded the FastAPI surface to support the SPA with graph network payloads, prediction, clinical, analysis, and review endpoints.
- Added an interactive `scripts/demo.py` runner that checks Ollama, builds a mini graph, indexes local literature, trains GraphSAGE for 50 epochs, and showcases seven specialist agents.
- Rewrote `README.md` with architecture, schema, agent, RAG, hardware, and comparison sections.
- Added benchmark placeholder tables in `RESULTS.md`.
- Completed end-to-end workflow, clinical tools, graph ML, RAG, and specialist agent coverage from prior phases.

## 0.1.0 - 2026-04-05

- Initial scaffold for BioMedOS.
- Added the local Ollama client, vector store, graph core, API surface, and foundational tests.
- Added typed stubs for future biomedical agents, analysis modules, and clinical tooling.
