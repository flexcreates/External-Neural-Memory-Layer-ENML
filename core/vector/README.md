# Vector Subsystem

The vector subsystem manages embedding generation, Qdrant connection lifecycle, and semantic retrieval with hybrid re-ranking.

## Files

### `embeddings.py` — EmbeddingService (Singleton)
Wraps `SentenceTransformer` (`all-MiniLM-L6-v2`) as a thread-safe singleton. The model (~90 MB) is loaded once and shared across all callers.

**Usage:**
```python
from core.vector.embeddings import EmbeddingService
service = EmbeddingService()  # Same instance every time
vector = service.embed("user has_name Flex.")  # Returns list[float] of 384 dims
```

---

### `qdrant_client.py` — QdrantManager (Singleton)
Manages the Qdrant client connection and ensures all required collections exist. Thread-safe singleton — one connection per process.

**Collections managed:**
| Collection | Purpose |
|---|---|
| `knowledge_collection` | Semantic triple facts (primary memory) |
| `conversation_collection` | Ingested conversation transcripts |
| `research_collection` | Web/document research chunks |
| `project_collection` | Ingested code file chunks |
| `profile_collection` | User profile embeddings |

---

### `retriever.py` — Hybrid Semantic Search
The `Retriever` class handles both insertions and search with a two-stage retrieval pipeline:

1. **Broad Vector Search** — queries Qdrant for 2× the requested limit
2. **Local Re-Ranking** — applies keyword matching boosts:
   - Subject match: +0.20
   - Predicate match: +0.10
   - Object match: +0.15
3. **Status Filtering** — superseded facts are automatically excluded
4. **Filter Support** — accepts both raw `FieldCondition` lists and simple `filter_dict` key-value pairs

**Search Example:**
```python
results = retriever.search(
    collection="knowledge_collection",
    query="what is the user's name",
    limit=5,
    filter_dict={"subject": "user", "predicate": "has_name"}
)
```
