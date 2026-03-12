# ENML Development Guide

This guide documents the current developer workflow.

## Local Setup

```bash
./setup.sh
source .venv/bin/activate
python3 -m compileall core
```

Start services as needed:

```bash
./run_qdrant.sh
./run_server.sh
./run_web.sh
```

## High-Risk Areas

Changes in these files usually require documentation, tests, and runtime verification together:

- `core/prompt_templates.py`
- `core/context_builder.py`
- `core/memory_manager.py`
- `core/memory/extractor.py`
- `core/memory/validators.py` (NEW)
- `core/hallucination_guard.py` (NEW)
- `core/memory/garbage_collector.py` (NEW)
- `core/vector/retriever.py`
- `core/router/query_router.py`
- `core/router/pipeline_router.py` (NEW)
- `core/model_registry.py` (NEW)
- `core/coding/memory.py` (NEW)
- `run_server.sh`
- `run_qdrant.sh`

## Prompt Work

When editing prompt behavior:

- verify the main chat path
- verify helper LLM paths that use prompt templates indirectly
- verify at least one Mistral-family render
- verify at least one Llama 3 render
- verify at least one Llama 3 render

Important current rule:

- Mistral prompt construction should not manually prepend `<s>` for the current GGUF server path

## Server Startup Work

`run_server.sh` is no longer a simple shell wrapper.

Current behavior:

- parses `.env` without shell-sourcing it
- enumerates GGUF models
- reads model `context_length` from the GGUF header
- uses `llama-fit-params` to plan context and GPU layer count
- launches `llama-server` with explicit `-c` and `--gpu-layers`

If you change startup logic, update:

- `README.md`
- `docs/USER_GUIDE.md`
- `docs/RESOURCE_ARCHITECTURE.md`
- `.env.example`

## Memory Work

Current memory behavior depends on all of these layers:

- authority memory
- local record repository
- Qdrant vector memory
- retrieval policies
- evidence packet formatting
- **memory validators** (NEW - prevents noise/injection)
- **hallucination guard** (NEW - prevents self-hallucination)
- **garbage collector** (NEW - cleanup)

If you change extraction or memory storage rules, verify:

- direct profile statements
- corrections
- degraded mode with Qdrant down
- personal recall questions
- ordinary conversation that should not trigger strict grounding
- validation is blocking conversational noise
- hallucination guard is triggering on self-reference queries

## Coding Pipeline Work

The new coding module handles task tracking and context generation for code models. It operates across:

- `core/router/pipeline_router.py` (routes input based on active model tier)
- `core/coding/models.py` (data structures for tasks)
- `core/coding/task_store.py` (JSON persistence of tasks)
- `core/coding/vector_store.py` (optional Qdrant vector index for code tasks and context)
- `core/coding/context_builder.py` (builds injectable task/project contexts)
- `core/coding/memory.py` (facade composing the above)

When modifying the coding pipeline, verify:
- General tier models do not silently re-route to code paths if the user asks for code
- Coder tier models correctly embed task context via the `get_prompt_injection` facade
- `task_store` saves gracefully without crashing if the disk is full or unavailable
- `vector_store` returns gracefully without crashing if Qdrant is absent

## Useful Checks

Syntax and quick checks:

```bash
python3 -m compileall core
bash -n run_qdrant.sh
bash -n run_server.sh
bash -n run_web.sh
```

Tests:
(Tests have been removed for production stabilization)

Runtime evaluations:

```bash
python3 chat.py --diagnose
python3 chat.py --eval-runtime
python3 chat.py --eval-citations
python3 tools/eval_lifecycle.py --json
python3 tools/retrieval_benchmark.py --iterations 100
```

Service checks:

```bash
curl http://localhost:6333/readyz
curl http://localhost:8080/v1/models
curl http://localhost:5000/api/health
```

## Documentation Rule

If you change:

- environment variables
- startup scripts
- prompt family behavior
- routing behavior
- retrieval policy behavior
- storage layout

then update the matching markdown docs in the same change.
