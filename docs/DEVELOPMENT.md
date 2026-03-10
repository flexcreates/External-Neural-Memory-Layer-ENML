# ENML Development Guide

How to extend, modify, and develop the External Neural Memory Layer.

---

## Development Setup

```bash
# 1. Clone and set up
git clone https://github.com/flexcreates/ENML.git
cd ENML
chmod +x setup.sh && ./setup.sh

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Verify imports
python3 -c "from core.orchestrator import Orchestrator; print('✅ OK')"
```

---

## Adding a New Qdrant Collection

To add a new domain-specific collection (e.g., `notes_collection`):

### Step 1: Add config constant

```python
# core/config.py
QDRANT_NOTES_COLLECTION = os.getenv("QDRANT_NOTES_COLLECTION", "notes_collection")
```

### Step 2: Register in QdrantManager

```python
# core/vector/qdrant_client.py — inside __new__()
instance.collections = [
    # ... existing collections ...
    QDRANT_NOTES_COLLECTION,
]
```

### Step 3: Add routing rules

```python
# core/router/query_router.py — inside route()
notes_keywords = ["note", "memo", "reminder", "todo"]
if any(kw in query_lower for kw in notes_keywords):
    return QDRANT_NOTES_COLLECTION
```

### Step 4: Create ingestion script

```python
# ingest_notes.py
from core.vector.retriever import Retriever
from core.config import QDRANT_NOTES_COLLECTION

def ingest_note(text: str, tags: list[str]):
    retriever = Retriever()
    payload = {"type": "note", "tags": tags}
    retriever.add_memory(QDRANT_NOTES_COLLECTION, text, payload)
```

### Step 5: Update .env.example

```env
QDRANT_NOTES_COLLECTION=notes_collection
```

---

## Adding New Predicate Types

### Multi-Value Predicates (list semantics)

Add to `MULTI_VALUE_PREDICATES` in `core/knowledge_graph.py`:
```python
MULTI_VALUE_PREDICATES: Set[str] = {
    # ... existing ...
    'has_note', 'has_reminder',  # New additions
}
```

### Single-Value Predicates (replace semantics)

Add to `SINGLE_VALUE_PREDICATES` in `core/knowledge_graph.py`:
```python
SINGLE_VALUE_PREDICATES: Set[str] = {
    # ... existing ...
    'has_birthday', 'has_email',  # New additions
}
```

### Extraction Hints

Update the `EXTRACTION_PROMPT` in `core/memory/extractor.py` if you want the LLM to recognize new fact types.

---

## Adding a New Tool

Tools live in `tools/` and provide capabilities the AI can invoke.

### Step 1: Create the tool

```python
# tools/web_search_tool.py
from core.logger import get_logger

logger = get_logger(__name__)

class WebSearchTool:
    def search(self, query: str) -> list[dict]:
        """Search the web for information."""
        # Implementation here
        pass
```

### Step 2: Register it

Add to `tools/__init__.py`:
```python
from .web_search_tool import WebSearchTool
```

### Step 3: Wire into Orchestrator

Inside `core/orchestrator.py`, instantiate and use the tool:
```python
from tools.web_search_tool import WebSearchTool

class Orchestrator:
    def __init__(self):
        # ... existing init ...
        self.web_search = WebSearchTool()
```

---

## Modifying the Extraction Prompt

The extraction prompt in `core/memory/extractor.py` controls what facts the LLM extracts. Key guidelines:

1. **Keep it short** — the extraction runs in a small context window
2. **Be explicit** about the JSON format — lower-capability models need rigid structure
3. **List negative examples** — e.g., "DO NOT extract facts from questions"
4. **Use temperature 0** — ensures deterministic extractions

---

## Project Structure Conventions

| Convention | Description |
|---|---|
| `core/` | Framework internals only — no user-facing logic |
| `tools/` | Sandboxed utility classes |
| `research/` | Web and document ingestion |
| `memory/` | Runtime data directory (gitignored) |
| `docs/` | User and developer documentation |
| `logs/` | Runtime logs (gitignored) |
| Singletons | `EmbeddingService`, `QdrantManager` — use `__new__` pattern |
| Logging | Always use `get_logger(__name__)` |
| Config | All env vars go through `core/config.py` |

---

## Running Tests

```bash
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run diagnostics
python3 chat.py --diagnose

# Syntax check all modules
python3 -m py_compile core/config.py
python3 -m py_compile core/orchestrator.py
# ... etc
```

---

## Common Patterns

### Using the Retriever to Store Facts Programmatically

```python
from core.vector.retriever import Retriever
from core.config import QDRANT_KNOWLEDGE_COLLECTION

retriever = Retriever()
retriever.add_memory(
    collection=QDRANT_KNOWLEDGE_COLLECTION,
    text="user has_skill python.",
    payload={
        "subject": "user",
        "predicate": "has_skill",
        "object": "python",
        "confidence": 1.0,
        "status": "active",
        "timestamp": "2026-02-22T12:00:00"
    }
)
```

### Querying the Knowledge Graph

```python
from core.knowledge_graph import EntityLinker
from core.vector.embeddings import EmbeddingService

linker = EntityLinker(embedding_service=EmbeddingService())

# Get all active facts for a subject
entity = linker.resolve_or_create("user")
facts = linker.get_current_facts(entity.id)
for fact in facts:
    print(f"{fact.predicate}: {fact.object_literal}")
```
