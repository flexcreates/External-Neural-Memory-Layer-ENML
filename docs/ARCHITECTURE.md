# ENML Architecture

This document describes the current runtime architecture after the prompt-template and startup refactors.

## End-To-End Flow

```text
User Input
  -> Orchestrator
  -> Profile Update / Memory Extraction
  -> Query Routing
  -> Retrieval Policy
  -> Retriever + Reranking
  -> Evidence Packet
  -> Context Builder
  -> Authority Memory Injection
  -> Model-Specific Prompt Builder
  -> llama-server
  -> Citation Tracking
  -> Runtime Replay Logging
```

## Main Components

| Component | File | Purpose |
|---|---|---|
| Orchestrator | `core/orchestrator.py` | Main runtime pipeline and streaming generation |
| Memory Manager | `core/memory_manager.py` | Fact storage, retrieval, evidence packet building |
| Context Builder | `core/context_builder.py` | System prompt grounding, history trimming, answer policy injection |
| Prompt Templates | `core/prompt_templates.py` | Family-based prompt routing and rendering |
| Query Router | `core/router/query_router.py` | Routes to knowledge, project, document, or research collections |
| Distiller | `core/context/distiller.py` | Compresses evidence for models that benefit from it |
| Authority Memory | `core/memory/authority_memory.py` | Deterministic always-injected identity/profile storage |
| Citation Tracker | `core/citation_tracker.py` | Evidence usage logging |
| Runtime Replay | `core/runtime_replay.py` | Structured runtime traces |

## Prompt Construction

The prompt flow now has two distinct stages:

1. `core/context_builder.py`
   - builds the grounded system block
   - injects retrieved evidence
   - injects authority memory
   - trims conversation history against the token budget

2. `core/prompt_templates.py`
   - detects the active model family
   - renders the final prompt in the correct model format

This applies to:

- main chat generation
- query classification helper
- context distillation helper
- episodic summarization helper

## Active Model Detection

ENML no longer assumes one fixed chat model for all paths.

`core/llm_runtime.py` detects the active model from `llama-server` using `/v1/models`, and that result is reused by internal helper calls. This is important because the server may be running Gemma, Qwen, DeepSeek Coder, or another model selected at launch.

## Template Families

Current families:

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

The mapping is filename/model-id based, but it is designed to match the active server model discovered at runtime.

## Retrieval And Grounding

Retrieval is not one flat search.

The system:

- routes the query by intent
- applies a retrieval policy
- retrieves evidence
- reranks and groups evidence into an `EvidencePacket`
- injects answer-policy text and knowledge blocks into the grounded system prompt

Direct personal-memory questions also receive stricter grounding behavior, especially when evidence is missing.

## Storage Layers

Persistent runtime data:

- `memory/conversations/` for saved sessions
- `memory/authority/profile.json` for deterministic user/assistant/system state
- Qdrant collections for semantic retrieval
- `logs/` for system and evaluation logs

## Startup Scripts

Operational scripts:

- `setup.sh` initializes the environment
- `run_qdrant.sh` starts Qdrant
- `run_server.sh` starts `llama-server` and exposes the active model alias
- `run_web.sh` starts the web UI and syncs its model routing to the active server model

## Known Constraints

- prompt routing is family-based, so unusual GGUF conversions can still require special handling
- model quality depends on the GGUF conversion, quantization, and partial GPU offload
- some older stored memories may reflect pre-refactor extraction behavior, especially around preference verbs such as `like` vs `love`
