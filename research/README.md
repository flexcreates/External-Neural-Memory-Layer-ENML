# Research Module

The research module currently provides manual web-page ingestion, not live search during ordinary chat.

## Main File

| File | Purpose |
|---|---|
| `web_ingestor.py` | fetch, clean, chunk, and optionally store remote web content |

## Current Pipeline

```text
URL
  -> safety check
  -> HTTP fetch
  -> HTML cleanup
  -> text extraction
  -> chunking
  -> optional storage in research memory
```

## Current Safeguards

- blocks private/local targets
- uses request timeouts
- strips obvious scripts and layout noise

## Current Limitation

This module is not automatically called during normal chat. It is a manual ingestion path for building local research memory.
