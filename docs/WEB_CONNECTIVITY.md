# ENML Web Connectivity

ENML is currently offline-first.

## What Exists Today

- local `llama-server` integration
- local Flask web UI
- manual web ingestion through the research module
- local document, project, and research retrieval

## What Does Not Happen Automatically

- live web search during ordinary chat
- automatic search engine calls
- automatic remote retrieval fallback when local evidence is missing

## Current Web-Related Pieces

- [web_server.py](/home/flex/Projects/enml/web_server.py): browser chat UI and debug endpoints
- [run_web.sh](/home/flex/Projects/enml/run_web.sh): startup wrapper for the web UI
- [research/web_ingestor.py](/home/flex/Projects/enml/research/web_ingestor.py): manual web page fetch/clean/chunk/store path

## If You Add Live Web Retrieval Later

Keep these rules:

- local memory remains the first retrieval layer
- external evidence must be clearly distinguishable from local memory
- runtime logs and citation logs must record external evidence use
- failures must degrade cleanly back to local-only behavior

Suggested insertion point:

```text
User query
  -> Query router
  -> Retrieval policy
  -> Local retrieval
  -> Optional external retrieval
  -> Evidence packet
  -> Context builder
```
