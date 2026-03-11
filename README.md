# ENML

ENML is a local memory layer for local LLMs. It wraps `llama-server`, Qdrant, deterministic authority memory, local record storage, retrieval policies, and model-specific prompt rendering into one CLI and web chat system.

## What The System Does Now

- extracts user facts, preferences, and profile updates from conversation
- validates memories before storage (blocks noise and injection)
- stores deterministic identity facts in `memory/authority/profile.json`
- stores local memory records even when Qdrant is unavailable
- uses hybrid dense+sparse retrieval with reranking when Qdrant is up
- builds grounded prompts differently for each model family
- exposes both a CLI (`chat.py`) and a Flask web UI (`web_server.py`)
- logs runtime replay and citation metrics for later evaluation
- prevents hallucination on self-referential questions
- includes garbage collection for old memories

## Runtime Stack

Typical local services:

- Qdrant on `http://localhost:6333`
- `llama-server` on `http://localhost:8080`
- ENML web UI on `http://localhost:5000`

Main entrypoints:

- [chat.py](/home/flex/Projects/enml/chat.py)
- [web_server.py](/home/flex/Projects/enml/web_server.py)
- [run_qdrant.sh](/home/flex/Projects/enml/run_qdrant.sh)
- [run_server.sh](/home/flex/Projects/enml/run_server.sh)
- [run_web.sh](/home/flex/Projects/enml/run_web.sh)

## Setup

```bash
git clone https://github.com/flexcreates/ENML.git
cd ENML
chmod +x setup.sh
./setup.sh
```

After setup, review `.env` and update at least:

- `MODELS_DIR`
- `LLAMA_SERVER`
- `LLAMA_FIT_PARAMS`
- `LLAMA_SERVER_URL`
- `QDRANT_URL`
- `ALLOWED_PATHS`
- `AI_NAME`

## Start The System

Start Qdrant:

```bash
./run_qdrant.sh
```

Primary check:

```bash
curl http://localhost:6333/readyz
```

Start the model server:

```bash
./run_server.sh
```

`run_server.sh` inspects the selected GGUF, reads current free VRAM, asks `llama-fit-params` for a fit, and launches `llama-server` with the planned context and GPU layer count.

Start the CLI:

```bash
source .venv/bin/activate
python3 chat.py
```

Start the web UI:

```bash
./run_web.sh
```

## CLI Features

- `python3 chat.py`
- `python3 chat.py --session <session_id>`
- `python3 chat.py --diagnose`
- `python3 chat.py --eval-runtime`
- `python3 chat.py --eval-citations`

CLI commands:

- `exit` or `quit`
- `/remember <text>`

Large pastes are classified as document input and go through the document ingester instead of normal realtime extraction.

## Current Memory Layers

1. Authority memory
   Stored in `memory/authority/profile.json` and always injected into recall prompts.

2. Local record repository
   Stored as JSON memory records and used as a fallback even when vector memory is down.

3. Vector memory
   Qdrant collections for knowledge, episodic, document, project, research, conversation, and profile retrieval.

4. Session history
   Stored under `memory/conversations/YYYY/MM/`.

## Current Retrieval Behavior

High-level flow:

```text
User input
  -> Orchestrator
  -> Memory extraction / preference capture
  -> Memory validation (blocks noise/injection)
  -> Query routing
  -> Retrieval policy selection
  -> Hallucination guard (for self-reference)
  -> Vector retrieval + local fallback
  -> Context building
  -> Authority memory injection
  -> Model-specific prompt rendering
  -> llama-server generation
  -> Citation tracking / runtime replay
```

Important runtime files:

- [core/orchestrator.py](/home/flex/Projects/enml/core/orchestrator.py)
- [core/memory_manager.py](/home/flex/Projects/enml/core/memory_manager.py)
- [core/context_builder.py](/home/flex/Projects/enml/core/context_builder.py)
- [core/prompt_templates.py](/home/flex/Projects/enml/core/prompt_templates.py)
- [core/router/query_router.py](/home/flex/Projects/enml/core/router/query_router.py)
- [core/vector/retriever.py](/home/flex/Projects/enml/core/vector/retriever.py)
- [core/memory/validators.py](/home/flex/Projects/enml/core/memory/validators.py) (memory validation)
- [core/hallucination_guard.py](/home/flex/Projects/enml/core/hallucination_guard.py) (hallucination prevention)
- [core/memory/garbage_collector.py](/home/flex/Projects/enml/core/memory/garbage_collector.py) (cleanup)

## Model Families

Current prompt families:

- `llama3`
- `mistral`
- `qwen`
- `deepseek-coder`
- `deepseek`
- `gemma`
- `phi3`
- `openchat`
- `wizardcoder`
- `smollm3`
- `generic`

ENML detects the active server model from `/v1/models` and uses that for prompt formatting and helper LLM calls.

## Diagnostics And Evaluation

```bash
python3 chat.py --diagnose
python3 chat.py --eval-runtime
python3 chat.py --eval-citations
python3 tools/eval_lifecycle.py --json
python3 tools/retrieval_benchmark.py --iterations 100
```

Health checks:

```bash
curl http://localhost:6333/readyz
curl http://localhost:8080/v1/models
curl http://localhost:5000/api/health
```

## Logs

- `logs/memory_system.log`
- `logs/pipeline.log`
- `logs/audit.jsonl`
- `logs/runtime_replay.jsonl`
- `logs/citations.jsonl`

## Documentation

- [User Guide](docs/USER_GUIDE.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Development Guide](docs/DEVELOPMENT.md)
- [Resource Architecture](docs/RESOURCE_ARCHITECTURE.md)
- [Web Connectivity](docs/WEB_CONNECTIVITY.md)
- [Core Overview](core/README.md)
