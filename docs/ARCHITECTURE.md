# ENML System Architecture

Technical deep-dive into the External Neural Memory Layer architecture.

---

## System Overview

ENML is composed of three major layers:

```
┌──────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
│                                                                  │
│  chat.py · ingest_*.py · tools/file_tool.py                      │
├──────────────────────────────────────────────────────────────────┤
│                     COGNITIVE CORE                               │
│                                                                  │
│  Orchestrator ──► Extractor ──► EntityLinker                     │
│       │                              │                           │
│       ├──► ContextBuilder            ├──► Knowledge Graph        │
│       │       │                      └──► Qdrant Upsert          │
│       │       ├──► QueryRouter                                   │
│       │       ├──► Retriever                                     │
│       │       └──► AuthorityMemory                               │
│       │                                                          │
│       └──► LLM (llama-server via OpenAI API)                     │
├──────────────────────────────────────────────────────────────────┤
│                     STORAGE LAYER                                │
│                                                                  │
│  Qdrant (Docker) · JSON Files · Authority Profile                │
└──────────────────────────────────────────────────────────────────┘
```

---

## Message Processing Pipeline

When a user sends a message, the following happens in order:

### Phase 1: Fact Extraction (Background)
```
User Message ──► MemoryExtractor
                      │
                      ├──► LLM Inference (temperature=0, max_tokens=500)
                      │    Prompt: "Extract facts as JSON triples..."
                      │
                      ├──► RobustJSONParser
                      │    (handles markdown fences, nested JSON, etc.)
                      │
                      ├──► Confidence Filtering
                      │    (per-type thresholds: identity=0.80, interest=0.70)
                      │
                      └──► Verified Facts List
```

### Phase 2: Fact Storage
```
Verified Facts ──► MemoryManager.update_profile()
                        │
                        ├──► EntityLinker.store_fact()
                        │    ├── Check exact duplicates
                        │    ├── Multi-value? → Add alongside existing
                        │    └── Single-value? → Contradiction detection
                        │         └── Supersede old if contradiction found
                        │
                        └──► Retriever.add_memory()
                             └── Embed text ──► Qdrant upsert
```

### Phase 3: Context Building
```
User Query ──► ContextBuilder.build_context()
                    │
                    ├──► QueryRouter.route()
                    │    └── identity / project / research / conversation
                    │
                    ├──► Retriever.search()
                    │    ├── Vector similarity (2× limit)
                    │    ├── Recency boost (+0.08 for <30m, for context queries)
                    │    ├── Entity re-ranking (+0.2 subject, +0.1 predicate)
                    │    └── Status filter (exclude superseded)
                    │
                    ├──► Fallback Search
                    │    └── If primary collection yields 0 results, fallback to other content collections
                    │
                    ├──► AuthorityMemory.get_injected_prompt()
                    │    └── Always inject identity + system specs
                    │
                    ├──► Token Budget Enforcement
                    │    └── Trim history from oldest if exceeding 2800 tokens
                    │
                    └──► Final System Prompt + Trimmed History
```

### Phase 4: Response Generation
```
System Prompt + History + User Message ──► LLM (streaming)
                                                │
                                                └──► Token stream ──► User
```

---

## Data Storage Architecture

### Qdrant Collections

| Collection | Payload Schema | Source |
|---|---|---|
| `knowledge_collection` | `{subject, predicate, object, confidence, status, timestamp, text}` | Auto-extraction |
| `conversation_collection` | `{type, session_id, importance}` | `ingest_conversation.py` |
| `research_collection` | `{type, source/source_url, topic, chunk_index}` | `ingest_research.py`, `WebIngestor`, `document_ingester.py` |
| `project_collection` | `{type, file, module, language, chunk_index}` | `ingest_project.py`, `document_ingester.py` |
| `document_collection` | `{type, heading, timestamp, text}` | `document_ingester.py` |
| `profile_collection` | `{type, ...}` | Reserved for future use |

### JSON Files

| File | Purpose |
|---|---|
| `memory/authority/identity.json` | Permanent identity profile (AI auto-aging, User specs) |
| `memory/graph/entities.json` | Entity registry (IDs, canonical names, aliases) |
| `memory/graph/facts_ledger.json` | Fact version history with supersession chains |
| `memory/graph/feedback_stats.json` | Retrieval quality statistics |
| `memory/conversations/YYYY/MM/*.json` | Session transcripts |

---

## Singleton Services

| Service | Purpose | Thread-Safe |
|---|---|---|
| `EmbeddingService` | SentenceTransformer model (90 MB) | ✅ |
| `QdrantManager` | Qdrant client connection | ✅ |
| `_LoggerConfigurator` | Logging handler setup | ✅ |

---

## Security Model

| Layer | Protection |
|---|---|
| **File I/O** | `FileTool` restricts to `ALLOWED_PATHS` |
| **Web Requests** | `WebIngestor` blocks private IPs (SSRF) |
| **LLM API** | Uses placeholder API key (`sk-proj-no-key`) |
| **Data** | All storage is local (Qdrant Docker, JSON files) |
| **Credentials** | `.env` is gitignored |
