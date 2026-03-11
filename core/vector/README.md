# Vector Subsystem

The vector subsystem manages embeddings, Qdrant availability, and hybrid retrieval.

## Main Files

| File | Purpose |
|---|---|
| `embeddings.py` | dense and sparse embedding services |
| `qdrant_client.py` | Qdrant singleton manager and collection lifecycle |
| `retriever.py` | insertions, hybrid search, reranking, and query expansion |

## Current Embeddings

- dense embeddings: `BAAI/bge-base-en-v1.5`
- sparse embeddings: `Qdrant/bm25`
- reranker: `BAAI/bge-reranker-base`

These run on CPU in the current architecture so the GGUF server can keep the GPU.

## Collections

The current Qdrant manager ensures these collections exist:

- `research_collection`
- `project_collection`
- `conversation_collection`
- `episodic_collection`
- `profile_collection`
- `knowledge_collection`
- `document_collection`

## Current Retrieval Behavior

The retriever:

1. performs hybrid sparse+dense retrieval
2. optionally expands the query into additional search variants
3. reranks candidates with the cross-encoder
4. applies recency and lightweight entity boosts
5. filters out superseded facts

## Degraded Mode

If Qdrant is unreachable:

- startup does not fail
- inserts are skipped with warnings
- vector search returns no vector results
- authority memory and local record fallback still work elsewhere in the system

Primary check:

```bash
curl http://localhost:6333/readyz
```
