# ENML Web Connectivity

Current status: ENML is designed to run fully locally. Internet-backed research is not part of the default runtime path.

## What Exists Today

- local web UI in `web_server.py`
- local document ingestion
- local Qdrant retrieval
- local `llama-server` generation

## What Does Not Exist By Default

- automatic live web search during chat
- automatic remote retrieval fallback
- built-in search engine API integration

## If You Add Web Research Later

Keep these constraints:

- local memory must remain the first retrieval layer
- web results should be explicitly marked as external evidence
- citations and runtime logs should record when external data was used
- web failures must degrade cleanly back to local-only behavior

## Suggested Integration Point

If web research is added, the correct place is after retrieval policy resolution and before final context building:

```text
User Query
  -> Query Router
  -> Retrieval Policy
  -> Local Retrieval
  -> Optional Web Retrieval
  -> Evidence Packet
  -> Context Builder
```

## Operational Note

The current project documentation, setup flow, and runtime scripts assume offline-first operation. If you add internet access later, update:

- `README.md`
- `docs/USER_GUIDE.md`
- `docs/DEVELOPMENT.md`
- `.env.example`
- any setup or deployment scripts that need API keys or extra packages
