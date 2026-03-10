# Research Module

The research module provides web content ingestion capabilities for building a local knowledge base from internet sources.

## Files

### `web_ingestor.py` — WebIngestor

Fetches, cleans, chunks, and optionally stores web page content into Qdrant's `research_collection`.

**Pipeline:**
```
URL → Safety Check → HTTP Fetch → HTML Parse → Text Clean → Chunk → Qdrant Store
```

**Safety Features:**
- **SSRF Protection**: Blocks requests to private IP ranges (10.x, 172.16.x, 192.168.x, 127.x)
- **Timeout**: 15-second request timeout
- **Content Filtering**: Strips scripts, styles, navigation, headers, footers

**Usage (Standalone):**
```python
from research.web_ingestor import WebIngestor
from core.vector.retriever import Retriever

# Without storage (returns cleaned text chunks)
ingestor = WebIngestor()
chunks = ingestor.ingest_url("https://example.com/article", topic="AI")

# With Qdrant storage
ingestor = WebIngestor(retriever=Retriever())
chunks = ingestor.ingest_url("https://example.com/article", topic="AI")
# Chunks are automatically stored in research_collection
```

**Usage (CLI):**
```bash
python3 ingest_research.py /path/to/saved_article.txt --topic "machine learning"
```

**Chunking:**
- Default chunk size: 800 characters
- 10% overlap between consecutive chunks for context preservation

> **Note:** See [docs/WEB_CONNECTIVITY.md](../docs/WEB_CONNECTIVITY.md) for instructions on connecting ENML to the internet for automated research.
