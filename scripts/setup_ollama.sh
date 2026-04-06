#!/usr/bin/env bash
set -euo pipefail

ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull qwen2.5:14b
ollama pull phi4:14b
ollama pull nomic-embed-text
