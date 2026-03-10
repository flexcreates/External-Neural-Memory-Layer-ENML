# ENML Core — Architectural Process Documentation

This directory houses the foundational backbone of the **Infinite Learning v2.0** engine. The core is organized into specialized subsystems, each responsible for a distinct phase of the memory pipeline.

---

## Module Overview

| Module | File | Purpose |
|---|---|---|
| **Orchestrator** | `orchestrator.py` | Central pipeline controller — coordinates extraction, context building, and LLM calls |
| **Memory Manager** | `memory_manager.py` | Memory operations coordinator — routes between storage backends |
| **Knowledge Graph** | `knowledge_graph.py` | Entity linking, contradiction detection, and temporal fact versioning |
| **Context Builder** | `context_builder.py` | Grounded prompt construction with token budget enforcement |
| **Memory Feedback** | `memory_feedback.py` | Retrieval quality tracking for future memory pruning |
| **Project Manager** | `project_manager.py` | Code snapshot versioning and execution logging |
| **Logger** | `logger.py` | Centralized logging (console, rotating file, JSON audit trail) |

## Subsystem Directories

| Directory | Purpose | Details |
|---|---|---|
| `memory/` | Fact extraction, triple storage, authority profile | [README](memory/README.md) |
| `vector/` | Embedding, Qdrant connection, semantic retrieval | [README](vector/README.md) |
| `router/` | Intent classification and collection routing | [README](router/README.md) |
| `storage/` | JSON session persistence | [README](storage/README.md) |

---

## Data Flow Pipeline

```
User Message
    │
    ▼
┌─────────────────┐
│   Orchestrator   │  ← Entry point for all interactions
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
 Extract    Build Context
    │            │
    ▼            ▼
┌────────┐  ┌──────────────┐
│Extractor│  │ContextBuilder│
│(LLM)   │  │              │
└───┬────┘  └──────┬───────┘
    │              │
    ▼              ├──► QueryRouter → Collection Selection
┌──────────┐       ├──► Retriever  → Qdrant Semantic Search
│EntityLink│       └──► AuthorityMemory → Profile Injection
│  er      │
└───┬──────┘
    │
    ├──► Knowledge Graph (JSON Ledger)
    └──► Qdrant (Vector Embedding)
```

---

## 1. Fact Extraction (`memory/extractor.py`)

Before the Orchestrator generates a response, it intercepts the user's input and fires a hidden, small-context LLaMA 3 inference loop. This loop acts strictly as a data miner:

- **Extracts** factual data (identities, system setups, relationships, hobbies).
- **Refuses** to extract conversational pleasantries, questions, or temporary context.
- **Formats** output exclusively as Semantic Triples:
  ```json
  [{"subject": "user", "predicate": "has_hobby", "object": "vibe coding", "fact_type": "interest", "confidence": 0.92}]
  ```
- **Confidence thresholds** are dynamically adjusted per fact type (identity: 0.80, interest: 0.70, general: 0.75).

## 2. Entity Linking & Versioning (`knowledge_graph.py`)

The `EntityLinker` provides two critical services:

### Entity Resolution
Text mentions (e.g., "Flex", "flex", "my name") are resolved to canonical entity IDs using exact + alias matching. New entities are created automatically.

### Temporal Fact Versioning
Instead of dropping contradictions, the system **versions** them:
- **Single-value predicates** (name, age): new values supersede old ones.
- **Multi-value predicates** (hobbies, pets): new values are added alongside existing ones.
- Superseded facts retain their history with `superseded_by` pointers.

## 3. Triple Storage (`memory/triple_memory.py`)

Extracted facts are converted into `MemoryTriple` dataclasses:
```python
MemoryTriple(subject="user", predicate="has_hobby", object="vibe coding",
             confidence=0.92, fact_type="interest", source="user")
```

The `natural_sentence` property (`"user has_hobby vibe coding."`) is what gets embedded as a vector.

## 4. Qdrant Ingestion & Retrieval (`vector/`)

Every Triple is passed through the `EmbeddingService` (all-MiniLM-L6-v2, 384 dimensions) and committed to the `knowledge_collection`. The `Retriever` performs:
1. **Broad vector similarity** search (2× limit for re-ranking headroom)
2. **Local re-ranking** with entity keyword matching boosts
3. **Status filtering** — superseded facts are automatically excluded

## 5. Query Routing (`router/query_router.py`)

The `QueryRouter` classifies user intent and directs queries to the appropriate Qdrant collection:
- **Identity queries** ("what is my name") → `knowledge_collection`
- **Project queries** ("how is the function implemented") → `project_collection`
- **Research queries** ("explain transformers") → `research_collection`
- **General** → `conversation_collection`

Self-referential queries (containing "my", "I", "me") are force-routed to `knowledge_collection`.

## 6. Grounded Context Building (`context_builder.py`)

Rather than appending raw conversation history, the Context Builder constructs a strict system prompt:

```
Relevant Known Facts:
- user has_name Flex.
- user has_hobby paragliding.

Only answer using the provided facts where applicable.
If no relevant fact exists, say you don't know.
```

This multi-layered approach guarantees:
- The AI never guesses or hallucinates personal parameters.
- The database scales infinitely without exceeding the LLM's context window.
- **Token budget enforcement** trims older history when approaching context limits.

## 7. Authority Memory (`memory/authority_memory.py`)

A deterministic JSON profile (`memory/authority/profile.json`) that is **always** injected into the system prompt, regardless of vector search results. This guarantees core identity (name, system specs) is never lost.

## 8. Memory Feedback (`memory_feedback.py`)

Tracks retrieval-vs-usage statistics for each fact:
- `retrieved_count`: how many times a fact was surfaced
- `used_in_response`: how many times it was actually relevant
- `user_corrected`: whether the user flagged it as wrong

Quality score: `(usefulness × 0.6) + (accuracy × 0.4)` — enables future pruning of low-quality facts.
