# Router Subsystem

The router subsystem decides where retrieval or model-selection logic should go next.

## Main Files

| File | Purpose |
|---|---|
| `query_router.py` | routes user queries to the most relevant collection |
| `model_router.py` | lightweight heuristic model selector for helper LLM calls |
| `model_profiles.py` | prompt-budget/model-profile definitions |

## Query Routing

`query_router.py` combines heuristics with an LLM fallback classifier.

Current target collections:

- `knowledge_collection`
- `project_collection`
- `document_collection`
- `research_collection`

Important current behavior:

- explicit personal phrasing is routed to `knowledge_collection`
- many general world-knowledge prompts are routed to `research_collection`
- some short general tasks still bypass broad research routing and stay on the knowledge path because the retrieval policy may later decide memory should not be forced

## Model Routing

`model_router.py` is used mainly for helper calls such as summarization/classification.

It chooses between:

- `FAST_CHAT_MODEL`
- `CODING_CHAT_MODEL`
- `REASONING_CHAT_MODEL`
- `DEFAULT_CHAT_MODEL`

This is separate from the active `llama-server` model-detection path used by the main prompt template layer.
