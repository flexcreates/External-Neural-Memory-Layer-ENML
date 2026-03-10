# Router Subsystem

The router subsystem determines which Qdrant collection should be searched for a given user query.

## Files

### `query_router.py` — QueryRouter

**Intent Classification** uses keyword matching to route queries:

| Priority | Intent | Keywords | Target Collection |
|---|---|---|---|
| 1 (Highest) | Identity/Profile | "my name", "who am i", "my pc", "what is my" | `knowledge_collection` |
| 2 | Project/Code | "codebase", "function", "class", ".py" | `project_collection` |
| 3 | Research | "explain", "how does", "what is", "theory" | `research_collection` |
| 4 (Fallback) | General | Everything else | `conversation_collection` |

**Self-Referential Override:** The `MemoryManager` applies a secondary check — if a query routed to `conversation_collection` contains self-referential words ("my", "I", "me"), it is re-routed to `knowledge_collection`.

**Extending the Router:**
To add a new collection, update:
1. `core/config.py` — add the collection name constant
2. `core/vector/qdrant_client.py` — add to the `collections` list
3. `core/router/query_router.py` — add keyword patterns
