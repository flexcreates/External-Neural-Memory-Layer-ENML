# ENML Development Guide

This guide covers current development workflow and the rules that matter after the recent refactors.

## Local Dev Setup

```bash
./setup.sh
source .venv/bin/activate
python3 -m compileall core
```

Start supporting services as needed:

```bash
./run_qdrant.sh
./run_server.sh
```

## Important Runtime Rules

When you change prompt construction, you must check all of these paths:

- main chat generation in `core/orchestrator.py`
- query routing helper in `core/router/query_router.py`
- context distiller in `core/context/distiller.py`
- episodic summarization in `core/orchestrator.py`

They should all follow the same active-model detection and prompt-template system.

## Prompt Template Work

Primary file:

- `core/prompt_templates.py`

Supporting files:

- `core/llm_runtime.py`
- `core/context_builder.py`
- `core/orchestrator.py`

If you add a new model family:

1. add routing logic in `get_model_template_info`
2. add a renderer
3. verify `run_server.sh` classification output
4. test at least one real prompt render
5. make sure helper paths do not bypass the same template logic

## Config Surface

Runtime config is defined in:

- `core/config.py`
- `.env.example`

If you add a new environment variable, update both.

## Setup And Ops Scripts

Bootstrap / runtime shell scripts:

- `setup.sh`
- `run_qdrant.sh`
- `run_server.sh`
- `run_web.sh`
- `reset_memory.sh`

If you change `.env` parsing or runtime assumptions, keep these scripts aligned. Avoid shell-sourcing `.env` directly when values may contain quotes, spaces, or comments.

## Tests And Verification

Fast checks:

```bash
python3 -m compileall core
bash -n setup.sh
bash -n run_qdrant.sh
bash -n run_server.sh
bash -n run_web.sh
```

Project tests:

```bash
pytest -q
python3 chat.py --diagnose
python3 chat.py --eval-runtime
python3 chat.py --eval-citations
python3 tools/eval_lifecycle.py --json
```

Service checks:

```bash
curl http://localhost:6333/health
curl http://localhost:8080/health
curl http://localhost:8080/v1/models
curl http://localhost:5000/api/health
```

## Data Compatibility Notes

- authority memory lives in `memory/authority/profile.json`
- old stored facts may predate newer extraction rules
- preference memories created before the `likes` / `loves` split may need migration if exact verb recall matters

## Common Development Mistakes

- updating only the main chat path but not helper LLM calls
- documenting config keys that no longer exist
- assuming one default model while `run_server.sh` can launch any GGUF
- letting warning logs print into live chat output
- over-trusting model family names when a GGUF’s embedded template says otherwise
