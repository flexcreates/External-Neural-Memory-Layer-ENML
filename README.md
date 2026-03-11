# ENML

ENML is a local memory layer for local LLMs. It combines fact extraction, authority memory, Qdrant retrieval, evidence-grounded prompting, model-aware prompt templating, and runtime logging into one system for CLI and web chat.

## What Changed

The current system is not a simple "chat plus vector DB" stack.

- Prompt construction is model-aware and routes by active server model.
- The main chat path uses a single prompt-template system across CLI, web, routing helpers, distillation, and episodic summarization.
- Retrieval is policy-driven and injects explicit evidence plus answer rules into the prompt.
- Runtime traces, citations, and audit logs are written for debugging and evaluation.

## Requirements

- Python 3.10+
- `python3-venv`
- `pip`
- Docker for local Qdrant
- `llama.cpp` with `llama-server`
- At least one instruct GGUF model

Typical local setup:

- Qdrant on `http://localhost:6333`
- `llama-server` on `http://localhost:8080`
- Web UI on `http://localhost:5000`

## Setup

```bash
git clone https://github.com/flexcreates/ENML.git
cd ENML
chmod +x setup.sh
./setup.sh
```

`setup.sh` will:

- create `.venv`
- install `requirements.txt`
- create `.env` from `.env.example` if needed
- create runtime directories
- initialize `memory/authority/profile.json`
- optionally start Qdrant if Docker is available

After setup, review `.env` and update at least:

- `MODELS_DIR`
- `LLAMA_SERVER`
- `LLAMA_SERVER_URL`
- `ALLOWED_PATHS`
- `AI_NAME`

## Start The Stack

Start Qdrant:

```bash
./run_qdrant.sh
```

Start the model server:

```bash
./run_server.sh
```

`run_server.sh` scans your GGUF directory, shows the detected template family for each model, and starts `llama-server` with an alias that ENML can discover.

Start the CLI:

```bash
source .venv/bin/activate
python3 chat.py
```

Or start the web UI:

```bash
./run_web.sh
```

## Runtime Flow

```text
User Input
  -> Orchestrator
  -> Memory Extraction / Preference Capture
  -> Retrieval Policy + Query Routing
  -> Evidence Packet
  -> Context Builder
  -> Model-Specific Prompt Template
  -> llama.cpp Generation
  -> Citation Tracking + Runtime Logging
```

Important files:

- `core/orchestrator.py`
- `core/context_builder.py`
- `core/prompt_templates.py`
- `core/router/query_router.py`
- `core/context/distiller.py`
- `core/memory_manager.py`
- `web_server.py`

## Model Template Routing

Current template families in ENML:

- `llama3`
- `mistral`
- `qwen`
- `deepseek-coder` via ChatML-compatible routing
- `deepseek` legacy instruction style
- `gemma`
- `phi3`
- `openchat`
- `wizardcoder`
- `smollm3`
- `generic` fallback

The most important detail is that ENML now detects the active server model and routes prompts accordingly. Internal LLM helpers use the same template system as the main chat path.

## Configuration

Main environment variables:

```bash
ENML_ROOT=./
MEMORY_ROOT=./memory

ALLOWED_PATHS=/home/user/Projects,/home/user/Documents

MODELS_DIR=/home/user/ai-models
LLAMA_SERVER=/home/user/Tools/llama.cpp/build/bin/llama-server
LLAMA_SERVER_URL=http://localhost:8080
CONTEXT_SIZE=4096
CACHE_TYPE_K=q8_0
CACHE_TYPE_V=q8_0

QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=

EMBED_MODEL=BAAI/bge-base-en-v1.5
EMBED_DIM=768

AI_NAME=ENML Assistant
AI_HINT=running on local hardware

DEFAULT_CHAT_MODEL=Meta-Llama-3-8B-Instruct
FAST_CHAT_MODEL=Meta-Llama-3-8B-Instruct
CODING_CHAT_MODEL=Meta-Llama-3-8B-Instruct
REASONING_CHAT_MODEL=Meta-Llama-3-8B-Instruct

WEB_SERVER_PORT=5000
ENML_DEBUG=0
```

Note: the router defaults above are still valid as config defaults, but when `llama-server` exposes a concrete active model, ENML detects and uses that model directly.

## Evaluation And Diagnostics

```bash
python3 chat.py --diagnose
python3 chat.py --eval-runtime
python3 chat.py --eval-citations
python3 tools/eval_lifecycle.py --json
python3 tools/retrieval_benchmark.py --iterations 100
```

If the full stack is running:

```bash
curl http://localhost:6333/health
curl http://localhost:8080/health
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
- [Web Connectivity](docs/WEB_CONNECTIVITY.md)
- [Resource Architecture](docs/RESOURCE_ARCHITECTURE.md)
