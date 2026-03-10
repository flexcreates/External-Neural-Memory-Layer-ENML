# Web Connectivity — Internet Integration Instructions

> **Status:** This document provides instructions for connecting ENML to the internet for automated research, web data extraction, and fact gathering. These capabilities are NOT yet active — this file serves as a roadmap and instruction set for future integration.

---

## Overview

ENML is designed to operate fully offline, but the architecture supports internet connectivity through the `research/web_ingestor.py` module. This document covers:

1. [Manual Web Ingestion (Available Now)](#1-manual-web-ingestion-available-now)
2. [Automated Web Research Pipeline](#2-automated-web-research-pipeline)
3. [Search Engine Integration](#3-search-engine-integration)
4. [Wikipedia / Knowledge Base APIs](#4-wikipedia--knowledge-base-apis)
5. [Real-Time Fact Extraction from Web](#5-real-time-fact-extraction-from-web)
6. [Scheduled Research Automation](#6-scheduled-research-automation)
7. [Safety & Rate Limiting](#7-safety--rate-limiting)

---

## 1. Manual Web Ingestion (Available Now)

The `WebIngestor` class can already fetch and parse web pages:

```python
from research.web_ingestor import WebIngestor
from core.vector.retriever import Retriever

# Initialize with Qdrant storage
ingestor = WebIngestor(retriever=Retriever())

# Ingest a web page (fetches, cleans, chunks, stores in research_collection)
chunks = ingestor.ingest_url("https://en.wikipedia.org/wiki/Python_(programming_language)", topic="python")
print(f"Ingested {len(chunks)} chunks")
```

You can then query this knowledge in chat:
```
You: explain Python programming language
AI: Based on the research material, Python is a high-level...
```

---

## 2. Automated Web Research Pipeline

### Architecture for Internet-Connected Research

```
User Query ──► Orchestrator
                   │
                   ├──► QueryRouter identifies "research" intent
                   │
                   ├──► Check local Qdrant first
                   │    └── If insufficient results (score < threshold)
                   │
                   ├──► WebResearchAgent (NEW)
                   │    ├── Search engine API call
                   │    ├── Fetch top N URLs
                   │    ├── WebIngestor.ingest_url() for each
                   │    └── Re-query Qdrant with fresh data
                   │
                   └──► ContextBuilder builds response
```

### Implementation Steps

**Step 1:** Create `research/web_research_agent.py`:

```python
"""
Web Research Agent — Automatic internet fact gathering.

This module connects ENML to the internet for real-time research.
It should be instantiated by the Orchestrator when research queries
return insufficient local results.
"""

import os
from typing import List, Optional
from research.web_ingestor import WebIngestor
from core.vector.retriever import Retriever
from core.config import QDRANT_RESEARCH_COLLECTION
from core.logger import get_logger

logger = get_logger(__name__)


class WebResearchAgent:
    def __init__(self):
        self.retriever = Retriever()
        self.ingestor = WebIngestor(retriever=self.retriever)
        self.search_api_key = os.getenv("SEARCH_API_KEY", "")
        self.search_engine_id = os.getenv("SEARCH_ENGINE_ID", "")

    def research(self, query: str, max_urls: int = 3) -> List[str]:
        """
        1. Search the web for relevant URLs
        2. Ingest top results into Qdrant
        3. Return the ingested chunks for immediate use
        """
        urls = self._search_web(query, max_urls)
        all_chunks = []
        for url in urls:
            chunks = self.ingestor.ingest_url(url, topic=query)
            if chunks:
                all_chunks.extend(chunks)
        return all_chunks

    def _search_web(self, query: str, max_results: int) -> List[str]:
        """Search using Google Custom Search API or similar."""
        # TODO: Implement actual search API call
        # See Section 3 below for API options
        raise NotImplementedError("Search API not yet configured")
```

**Step 2:** Add environment variables to `.env`:

```env
# Web Research (Optional — enables internet connectivity)
SEARCH_API_KEY=your_google_api_key_here
SEARCH_ENGINE_ID=your_custom_search_engine_id
WEB_RESEARCH_ENABLED=false
```

**Step 3:** Wire into Orchestrator:

```python
# In core/orchestrator.py, add after context retrieval:
if web_research_enabled and insufficient_local_results:
    from research.web_research_agent import WebResearchAgent
    agent = WebResearchAgent()
    fresh_data = agent.research(user_input)
    # Re-run context building with fresh data available
```

---

## 3. Search Engine Integration

### Option A: Google Custom Search API (Recommended)
- **Free tier:** 100 queries/day
- **Setup:** https://developers.google.com/custom-search/v1/overview
- **Cost:** $5 per 1000 queries beyond free tier

```python
import requests

def google_search(query: str, api_key: str, engine_id: str, num: int = 5) -> list:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {"key": api_key, "cx": engine_id, "q": query, "num": num}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return [item["link"] for item in resp.json().get("items", [])]
```

### Option B: DuckDuckGo (No API Key Required)
- **Free:** No cost, no API key needed
- **Library:** `pip install duckduckgo-search`

```python
from duckduckgo_search import DDGS

def ddg_search(query: str, max_results: int = 5) -> list:
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        return [r["href"] for r in results]
```

### Option C: SearXNG (Self-Hosted, Privacy-First)
- **Free:** Self-hosted metasearch engine
- **Docker:** `docker run -d -p 8888:8080 searxng/searxng`
- **No tracking:** Perfect for privacy-focused setups

```python
def searxng_search(query: str, instance: str = "http://localhost:8888") -> list:
    resp = requests.get(f"{instance}/search", params={"q": query, "format": "json"}, timeout=10)
    return [r["url"] for r in resp.json().get("results", [])[:5]]
```

---

## 4. Wikipedia / Knowledge Base APIs

### Wikipedia API (No Key Required)

```python
import requests

def wikipedia_summary(topic: str) -> str:
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic}"
    resp = requests.get(url, timeout=10)
    if resp.status_code == 200:
        return resp.json().get("extract", "")
    return ""
```

### arXiv API (Research Papers)

```python
import requests
import xml.etree.ElementTree as ET

def arxiv_search(query: str, max_results: int = 3) -> list:
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
    resp = requests.get(url, timeout=15)
    root = ET.fromstring(resp.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    papers = []
    for entry in root.findall("atom:entry", ns):
        papers.append({
            "title": entry.find("atom:title", ns).text.strip(),
            "summary": entry.find("atom:summary", ns).text.strip(),
            "url": entry.find("atom:id", ns).text.strip(),
        })
    return papers
```

---

## 5. Real-Time Fact Extraction from Web

Combine `WebIngestor` + `MemoryExtractor` to extract facts from web pages:

```python
from research.web_ingestor import WebIngestor
from core.memory.extractor import MemoryExtractor
from core.memory_manager import MemoryManager

ingestor = WebIngestor()
extractor = MemoryExtractor()
memory = MemoryManager()

# Fetch and clean a web page
text = ingestor.ingest_url("https://example.com/about-flex")

# Extract facts from the cleaned text
if text:
    for chunk in text:
        facts = extractor.extract_facts(chunk)
        for fact in facts:
            fact["source"] = "web"
            # Store in knowledge collection
            memory.update_profile(chunk)
```

---

## 6. Scheduled Research Automation

### Cron-Based Research

Create a script `scripts/scheduled_research.py`:

```python
#!/usr/bin/env python3
"""Run periodically via cron to keep research collection updated."""

import sys
sys.path.insert(0, "/home/flex/Ai-Models/models/ENML")

from research.web_ingestor import WebIngestor
from core.vector.retriever import Retriever

TOPICS = [
    ("AI safety research", "https://arxiv.org/list/cs.AI/recent"),
    ("Python releases", "https://www.python.org/downloads/"),
]

ingestor = WebIngestor(retriever=Retriever())
for topic, url in TOPICS:
    print(f"Ingesting: {topic}")
    ingestor.ingest_url(url, topic=topic)
```

### Cron Setup

```bash
# Edit crontab
crontab -e

# Run daily at 3 AM
0 3 * * * cd /home/flex/Ai-Models/models/ENML && .venv/bin/python3 scripts/scheduled_research.py >> logs/research_cron.log 2>&1
```

---

## 7. Safety & Rate Limiting

### Current Protections
- **SSRF blocking**: Private IPs (10.x, 172.16.x, 192.168.x, 127.x) are blocked
- **Request timeout**: 15 seconds per request
- **User-Agent**: Identifies as `ENML-Research-Bot/2.0`

### Recommended Additional Safety

When enabling internet access, add these protections:

```python
# Add to .env
WEB_RESEARCH_ENABLED=false          # Master switch
WEB_RATE_LIMIT_PER_MINUTE=10       # Max requests per minute
WEB_MAX_PAGE_SIZE_MB=5              # Reject pages larger than 5MB
WEB_BLOCKED_DOMAINS=facebook.com,twitter.com  # Domain blocklist
```

```python
# Rate limiter (add to WebIngestor or WebResearchAgent)
import time
from collections import deque

class RateLimiter:
    def __init__(self, max_per_minute: int = 10):
        self.max_per_minute = max_per_minute
        self.timestamps = deque()

    def wait_if_needed(self):
        now = time.time()
        while self.timestamps and self.timestamps[0] < now - 60:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.max_per_minute:
            sleep_time = 60 - (now - self.timestamps[0])
            time.sleep(max(0, sleep_time))
        self.timestamps.append(time.time())
```

### Data Provenance
When ingesting from the web, always store the `source_url` in the Qdrant payload so you can trace where information came from.

---

## Summary of Required Dependencies

When implementing internet connectivity, add these to `requirements.txt`:

```
# Already included:
requests>=2.31.0
beautifulsoup4>=4.12.0

# Optional (for search integration):
duckduckgo-search>=4.0  # If using DuckDuckGo
```

---

> **Reminder:** ENML is designed to work fully offline. Internet connectivity is an optional enhancement. Always verify that `WEB_RESEARCH_ENABLED=true` is set in `.env` before any internet features are activated.
