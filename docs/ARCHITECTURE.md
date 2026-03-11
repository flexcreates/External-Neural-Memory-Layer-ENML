# ENML Architecture

This document describes the current runtime architecture.

## End-To-End Flow

```text
User input
  -> Orchestrator
  -> Memory extraction / preference capture
  -> Memory validation (validators.py)
  -> Query router
  -> Retrieval policy engine
  -> Vector retrieval + local fallback records
  -> Evidence packet
  -> Context builder
  -> Hallucination guard (self-reference queries)
  -> Authority memory injection
  -> Model-specific prompt template
  -> llama-server generation
  -> Citation tracking
  -> Runtime replay logging
```

## Main Runtime Components

| Component | File | Purpose |
|---|---|---|
| Orchestrator | `core/orchestrator.py` | Main request pipeline and generation orchestration |
| Memory Manager | `core/memory_manager.py` | Memory updates, retrieval, local record fallback, authority integration |
| Memory Extractor | `core/memory/extractor.py` | Multi-layer fact extraction with LLM, rule, and regex passes |
| Memory Validator | `core/memory/validators.py` | Validates facts before storage, blocks noise and injection |
| Memory Garbage Collector | `core/memory/garbage_collector.py` | Periodic cleanup of deprecated memories |
| Context Builder | `core/context_builder.py` | Builds grounded prompts, trims history, and formats evidence |
| Hallucination Guard | `core/hallucination_guard.py` | Prevents hallucination on self-referential questions |
| Prompt Templates | `core/prompt_templates.py` | Renders prompts for the active model family |
| Query Router | `core/router/query_router.py` | Chooses the primary collection for retrieval |
| Retrieval Policy | `core/retrieval/policy.py` | Decides grounding strictness and collection mix |
| Retriever | `core/vector/retriever.py` | Hybrid dense+sparse retrieval plus reranking |
| Authority Memory | `core/memory/authority_memory.py` | Deterministic profile and preference store |
| Runtime Replay | `core/runtime_replay.py` | Structured runtime telemetry |
| Citation Tracker | `core/citation_tracker.py` | Tracks which evidence items were actually cited |

## Memory Layers

### Authority memory

- file-backed JSON profile
- always injected into direct recall prompts
- stores user identity, assistant identity, and user preferences

### Local record repository

- JSON memory records managed by the memory subsystem
- used as a fallback for retrieval and lifecycle services

### Vector memory

Qdrant collections:

- `knowledge_collection`
- `episodic_collection`
- `document_collection`
- `project_collection`
- `research_collection`
- `conversation_collection`
- `profile_collection`

### Session storage

- date-partitioned JSON session files in `memory/conversations/`

## Extraction Path

The extractor is layered:

1. intent and content guards reject questions, commands, and document-like input
2. direct fact statements are fast-pathed
3. LLM extraction runs first
4. rule extraction and regex extraction provide fallback coverage
5. predicates are normalized before storage

This is why direct profile statements such as `my name is ...` and `my profession is ...` behave more reliably than earlier versions.

## Retrieval Path

Retrieval is not one flat vector search.

The current flow is:

1. query routing picks the narrowest likely collection
2. retrieval policy decides strictness and primary/secondary collections
3. authority identity items are added locally
4. local record items are added locally
5. vector retrieval is attempted when Qdrant is available
6. results are grouped into an evidence packet
7. context builder formats answer rules and evidence blocks

## Prompt Construction

Prompt construction has two stages:

1. [core/context_builder.py](/home/flex/Projects/enml/core/context_builder.py)
   Produces the grounded system block, evidence sections, authority sections, and history selection.

2. [core/prompt_templates.py](/home/flex/Projects/enml/core/prompt_templates.py)
   Converts normalized messages into the exact family-specific prompt format.

This same template stack is reused by:

- main chat generation
- query classification helper calls
- document summarization helpers
- context distillation helpers

## Active Model Detection

ENML does not assume one fixed chat model.

[core/llm_runtime.py](/home/flex/Projects/enml/core/llm_runtime.py) detects the active server model from `/v1/models`, and the prompt/template layer uses that result to render the correct format.

## Startup Scripts

- [setup.sh](/home/flex/Projects/enml/setup.sh): bootstraps the environment
- [run_qdrant.sh](/home/flex/Projects/enml/run_qdrant.sh): starts local Qdrant via Docker
- [run_server.sh](/home/flex/Projects/enml/run_server.sh): chooses a model, plans context/gpu fit, launches `llama-server`
- [run_web.sh](/home/flex/Projects/enml/run_web.sh): launches Flask UI and aligns ENML’s model env with the active server model

## Current Constraints

- quality still depends on the GGUF conversion and quantization
- vector memory depends on Qdrant availability
- retrieval strictness is stronger for personal/document recall than for general chat
- partial GPU offload and large context windows change latency characteristics significantly

## Security & Validation

### Memory Validation Layer (`core/memory/validators.py`)

The memory validator prevents low-quality or malicious content from being stored:

- **Conversational Noise Filter**: Blocks greetings like "hi jarvis how are you buddy"
- **Prompt Injection Detection**: Blocks attempts to store malicious commands
- **Content Length Checks**: Rejects too-short or too-long content
- **Semantic Claim Validation**: Ensures claims aren't questions or commands

### Hallucination Guard (`core/hallucination_guard.py`)

Prevents the model from hallucinating about itself:

- **Self-Reference Detection**: Detects questions about training, origin, capabilities
- **Forced Factual Answers**: Injects system context for factual responses
- **Training Data Guard**: Blocks claims about web scraping or crowdsourcing

Example questions that trigger the guard:
- "How were you trained?"
- "What model are you?"
- "Who created you?"
- "Do you have internet access?"

### Memory Garbage Collector (`core/memory/garbage_collector.py`)

Periodic cleanup of memory stores:

- **Superseded Memory Cleanup**: Removes old versions of facts
- **Age-Based Cleanup**: Removes memories older than 90 days
- **Low-Confidence Cleanup**: Removes memories below 0.3 confidence
- **Session Cleanup**: Manages old conversation sessions

Run garbage collection periodically:

```python
from core.memory.garbage_collector import get_garbage_collector
gc = get_garbage_collector()
stats = gc.run_cleanup(dry_run=False)  # Actually delete
```
