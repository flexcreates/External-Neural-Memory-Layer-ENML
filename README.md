<p align="center">
  <h1 align="center">ENML — External Neural Memory Layer</h1>
  <p align="center">
    <em>A local cognitive memory layer for local LLM assistants.</em>
  </p>
  <p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#what-is-included">What Is Included</a> •
    <a href="#current-architecture">Architecture</a> •
    <a href="#runtime-evaluation">Runtime Evaluation</a> •
    <a href="docs/USER_GUIDE.md">User Guide</a> •
    <a href="docs/ARCHITECTURE.md">Architecture Doc</a> •
    <a href="docs/DEVELOPMENT.md">Development Guide</a>
  </p>
</p>

---

## Quick Start

### Requirements

| Requirement | Notes |
|---|---|
| Python 3.10+ | Tested locally on Python 3.12 |
| Docker | For local Qdrant |
| llama.cpp | Needs `llama-server` |
| GGUF model | Example: `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` |
| Local disk | For memory store, logs, and Qdrant data |

### Setup

```bash
git clone https://github.com/flexcreates/ENML.git
cd ENML
chmod +x setup.sh
./setup.sh
```

### Start Services

```bash
./run_qdrant.sh
./run_server.sh
./run_web.sh
```

Or use the CLI:

```bash
source .venv/bin/activate
python3 chat.py
```

---

## What Is Included

ENML now includes:

- layered fact extraction: LLM, rules, regex fallback
- authority identity and user preference memory
- rich `MemoryRecord` storage alongside legacy vector payloads
- policy-driven retrieval
- model profiles for small vs medium local models
- evidence-packet grounding for prompt injection
- semantic-claim fallback for schema-agnostic memory growth
- episodic memory summaries
- background consolidation and lifecycle pruning/archive hooks
- runtime replay logging
- citation tracking
- offline evaluation scripts for runtime, citations, and lifecycle

This is no longer just “vector memory plus prompt stuffing”. It is a local memory pipeline with ingestion, consolidation, retrieval policy, grounding, and observability.

---

## Current Architecture

High-level runtime flow:

```text
User Input
  -> Orchestrator
  -> Memory Extraction / Preference Capture
  -> MemoryRecord + Qdrant Storage
  -> Retrieval Policy Resolution
  -> Evidence Packet Construction
  -> Prompt Grounding
  -> llama.cpp Generation
  -> Citation Tracking
  -> Runtime Replay Logging
  -> Background Lifecycle / Episodic Maintenance
```

Important runtime modules:

| Module | Purpose |
|---|---|
| `core/orchestrator.py` | End-to-end chat pipeline |
| `core/memory_manager.py` | Ingestion, retrieval, evidence packet assembly |
| `core/memory/types.py` | `MemoryRecord`, `EvidencePacket`, memory enums |
| `core/retrieval/policy.py` | Retrieval policy engine |
| `core/context_builder.py` | Prompt grounding and answer policy injection |
| `core/citation_tracker.py` | Response-to-evidence tracking |
| `core/runtime_replay.py` | Runtime replay log writer |
| `core/memory/lifecycle_service.py` | Archive/prune lifecycle passes |
| `core/memory/document_ingester.py` | Document classification, summarization, fact extraction |

Collections currently used:

- `knowledge_collection`
- `project_collection`
- `research_collection`
- `document_collection`
- `episodic_collection`

---

## Runtime Evaluation

ENML now includes built-in runtime evaluation commands.

### CLI Metrics

```bash
python3 chat.py --eval-runtime
python3 chat.py --eval-citations
python3 tools/eval_lifecycle.py --json
python3 tools/retrieval_benchmark.py --iterations 1000
```

### Web Debug Endpoints

- `/api/debug/retrieve`
- `/api/debug/runtime-metrics`
- `/api/debug/citation-metrics`
- `/api/debug/memories`

### Logged Artifacts

| File | Purpose |
|---|---|
| `logs/runtime_replay.jsonl` | End-to-end request traces |
| `logs/citations.jsonl` | Evidence usage logs |
| `logs/audit.jsonl` | Structured system audit log |
| `logs/pipeline.log` | Retrieval and injection pipeline events |

---

## Configuration

Core `.env` settings:

```bash
AI_NAME=Jarvis
MODEL_PATH=/path/to/model.gguf
LLAMA_SERVER=/path/to/llama-server
LLAMA_SERVER_URL=http://localhost:8080
QDRANT_URL=http://localhost:6333

EMBED_MODEL=BAAI/bge-base-en-v1.5
EMBED_DIM=768

CONTEXT_SIZE=4096
PROMPT_BUDGET_SYSTEM=400
PROMPT_BUDGET_MEMORY=1200
PROMPT_BUDGET_DOCUMENTS=2000
PROMPT_BUDGET_USER=200

MIN_RETRIEVAL_CONFIDENCE=0.30
WEB_SERVER_PORT=5000
```

Model routing variables:

```bash
DEFAULT_CHAT_MODEL=Meta-Llama-3-8B-Instruct
FAST_CHAT_MODEL=Meta-Llama-3-8B-Instruct
CODING_CHAT_MODEL=Meta-Llama-3-8B-Instruct
REASONING_CHAT_MODEL=Meta-Llama-3-8B-Instruct
```

---

## Web UI

The web UI uses the same backend pipeline as the CLI:

- memory extraction
- evidence-packet grounding
- citation/runtime logging
- document ingestion
- SSE streaming responses

Launch:

```bash
./run_web.sh
```

Default URL:

```text
http://localhost:5000
```

---

## Verification

Recommended local checks before release:

```bash
python3 -m unittest discover -s tests -v
python3 tools/retrieval_benchmark.py --iterations 100
python3 chat.py --eval-runtime
python3 chat.py --eval-citations
```

If the full stack is running:

```bash
curl http://localhost:6333/health
curl http://localhost:8080/health
curl http://localhost:5000/api/health
```

---

## Notes

- ENML is designed to work fully locally.
- Internet tooling is documented separately in [docs/WEB_CONNECTIVITY.md](docs/WEB_CONNECTIVITY.md), but it is not required for the current memory stack.
- Runtime quality now depends more on retrieval policy and grounding discipline than on raw storage volume.
