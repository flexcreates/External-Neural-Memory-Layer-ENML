# ENML Core

This directory contains the main runtime pipeline.

## Main Files

| File | Purpose |
|---|---|
| `orchestrator.py` | Main request pipeline and generation orchestration |
| `memory_manager.py` | Memory updates, retrieval assembly, authority/local/vector coordination |
| `context_builder.py` | Grounded prompt construction and history trimming |
| `prompt_templates.py` | Model-family-specific prompt rendering |
| `llm_runtime.py` | Active model detection from `llama-server` |
| `citation_tracker.py` | Citation usage tracking |
| `runtime_replay.py` | Runtime telemetry logging |
| `config.py` | Environment-driven configuration |
| `logger.py` | Shared logging configuration |

## Subdirectories

| Directory | Purpose |
|---|---|
| `memory/` | extraction, authority memory, local record storage, lifecycle services |
| `vector/` | embeddings, Qdrant lifecycle, hybrid retrieval |
| `router/` | query routing and model routing heuristics |
| `retrieval/` | retrieval policy definitions |
| `context/` | distillation and prompt budget helpers |
| `storage/` | session persistence |

## Current Runtime Flow

```text
User input
  -> Orchestrator
  -> Memory extraction / profile capture
  -> Query router
  -> Retrieval policy
  -> Authority + local + vector retrieval
  -> Context builder
  -> Prompt template renderer
  -> llama-server
  -> Citation tracking / runtime replay
```

## Important Current Behaviors

- the active server model is detected at runtime instead of being assumed statically
- prompt formatting depends on the detected model family
- direct personal recall questions are treated more strictly than general chat
- Qdrant failure does not stop startup; the system degrades to local-only memory

## Related Docs

- [Memory](memory/README.md)
- [Vector](vector/README.md)
- [Router](router/README.md)
- [Storage](storage/README.md)
