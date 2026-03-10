# ENML User Guide

Complete guide for using the External Neural Memory Layer system — from first-time setup to advanced workflows.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Teaching Facts to Your AI](#teaching-facts-to-your-ai)
3. [Recalling Information](#recalling-information)
4. [Session Management](#session-management)
5. [Ingesting External Data](#ingesting-external-data)
6. [System Management](#system-management)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites Checklist

- [ ] Python 3.12+ installed
- [ ] Docker installed and running
- [ ] llama.cpp built (need `llama-server` binary)
- [ ] GGUF model downloaded (recommend: `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf`)
- [ ] ~4 GB free VRAM (GPU) or ~8 GB free RAM (CPU-only)

### First-Time Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/flexcreates/ENML.git
cd ENML

# 2. Run automated setup
chmod +x setup.sh
./setup.sh

# 3. Edit .env with your paths
nano .env
# Set MODEL_PATH and LLAMA_SERVER to your actual locations
```

### Starting ENML

Open **three terminals** in the ENML directory:

```bash
# Terminal 1: Start the dynamic LLM server (auto-detects models and VRAM)
./run_server.sh

# Terminal 2: Start the vector database
./run_qdrant.sh

# Terminal 3: Start the chat
source .venv/bin/activate
python3 chat.py
```

You should see:
```
Initializing ENML Orchestrator...
Scanning for models in /home/flex/ai-models...
[Select your model...]
--- Chat Started (Session: session_20260222_190000) ---
Type 'exit' to quit, '/remember <text>' to save a fact.

You: _
```

---

## Teaching Facts to Your AI

ENML learns automatically from natural conversation. Simply tell it things:

### Identity Information
```
You: my name is Flex
You: I'm a software engineer
You: I'm 25 years old
```

### System Specs
```
You: my PC is a Lenovo LOQ with i5-12450HX and RTX 3050
You: I have 16GB of RAM
You: I'm running Ubuntu 24.04
```

### Interests and Hobbies
```
You: I like vibe coding and creating art
You: I enjoy playing chess
You: I love watching sci-fi movies
```
> Interests/hobbies stack — saying "I like chess" won't overwrite "I like vibe coding".

### Relationships
```
You: my father's name is John
You: my best friend is Alex
You: I have a dog named Bruno
```

### Force-Saving Facts
If you want to explicitly save something without relying on automatic extraction:
```
You: /remember I have a medical condition called ADHD
✅ Memory saved: 'I have a medical condition called ADHD'
```

### What NOT to Do
- Don't expect the AI to learn from its own responses — only **your** messages are extracted.
- ENML uses **Semantic Intent Classification**. It will intentionally *ignore* messages it classifies as:
  - **Questions:** `"what is my name?"`
  - **Exploration/Discussion:** `"can we discuss how Rust works?"`
  - **Hypotheticals:** `"what if we used Python instead?"`

---

## Recalling Information

Simply ask the AI about things you've told it:

```
You: what is my name?
AI: Your name is Flex.

You: what are my PC specs?
AI: You have a Lenovo LOQ with an i5-12450HX processor and RTX 3050 GPU.

You: what are my hobbies?
AI: You enjoy vibe coding and creating art.
```

### How It Works (The 5-Step Recall)
1. **Query Intent Routing**: The system classifies your query to search the right memory silo (`knowledge`, `project`, `research`, etc.).
2. **Hybrid Search**: The system queries Qdrant using both dense semantic vectors (`BGE-base-en-v1.5`) and sparse keyword matching (`BM25`) to find up to 20 candidate memories.
3. **Memory Aging**: Older facts with low confidence naturally decay in score over time, while highly confident facts remain resilient.
4. **Cross-Encoder Reranking**: The retrieved memories are strictly re-scored alongside your query using `BAAI/bge-reranker-base` to surface the absolute most relevant points.
5. **Context Distillation**: If multiple fragmented memories are found, they are routed to the `ContextDistiller` to be compressed into a dense bullet-point summary before being injected into the final prompt.

If no relevant fact exists that passes the confidence thresholds, the AI will say "I don't know" instead of hallucinating.

---

## Session Management

### Automatic Session Saving
When you type `exit`, the current session is automatically saved to:
```
memory/conversations/YYYY/MM/session_YYYYMMDD_HHMMSS.json
```

### Resuming a Session
```bash
python3 chat.py --session session_20260222_184853
```
This loads the previous conversation history, so the AI has the full context.

### Finding Session IDs
Sessions are named with timestamps. Check your conversations directory:
```bash
ls memory/conversations/2026/02/
```

---

## Ingesting External Data

### Conversation Logs
Import a previously saved conversation into the vector database for long-term retrieval:
```bash
source .venv/bin/activate
python3 ingest_conversation.py memory/conversations/2026/02/session_xyz.json --importance 0.8
```

### Code Files
Make your AI aware of a codebase:
```bash
python3 ingest_project.py /path/to/project/main.py --module "MyProject" --language python
python3 ingest_project.py /path/to/project/utils.py --module "MyProject" --language python
```

Then ask coding questions:
```
You: what functions are in the MyProject module?
AI: Based on the ingested code, ...
```

### Research Documents
Ingest a text file (article, paper, documentation):
```bash
python3 ingest_research.py /path/to/paper.txt --topic "transformer architecture"
```

Then query it:
```
You: explain the transformer architecture
AI: Based on the research material, ...
```

### Web Pages
Use the WebIngestor programmatically:
```python
from research.web_ingestor import WebIngestor
from core.vector.retriever import Retriever

ingestor = WebIngestor(retriever=Retriever())
ingestor.ingest_url("https://example.com/article", topic="AI safety")
```

---

## System Management

### Running Diagnostics
```bash
python3 chat.py --diagnose
```
Tests: JSON parsing, Entity Linker versioning/contradiction detection.

### Checking Qdrant Status
Visit `http://localhost:6333/dashboard` in your browser to inspect collections, point counts, and storage.

### Viewing Logs
```bash
# Human-readable log
tail -f logs/memory_system.log

# Structured JSON audit trail
tail -f logs/audit.jsonl
```

### Full System Reset
```bash
chmod +x reset_memory.sh
./reset_memory.sh
```
> ⚠️ This permanently deletes ALL memory, sessions, profile data, and Qdrant collections.

---

## Troubleshooting

### "Connection refused" on chat startup
**Cause:** Qdrant or llama-server not running.
**Fix:**
```bash
# Check Qdrant
curl http://localhost:6333/health
# If it fails: ./run_qdrant.sh

# Check llama-server
curl http://localhost:8080/health
# If it fails: ./run_server.sh
```

### AI says "I don't know" for things you told it
**Possible causes:**
1. **Extraction failed** — check `logs/memory_system.log` for `❌ Rejected` lines
2. **Low confidence** — the LLM may have given a confidence below the threshold
3. **Wrong routing** — the query may be going to the wrong collection

**Debug:**
```bash
# Check what's stored in Qdrant
source .venv/bin/activate
python3 -c "
from core.vector.retriever import Retriever
r = Retriever()
results = r.search('knowledge_collection', 'user name', limit=10)
for res in results:
    print(res['payload'])
"
```

### AI hallucinating its own identity
**Cause:** Authority memory not properly loaded.
**Fix:** Check `memory/authority/identity.json` and verify `AI_NAME` in `.env`.

### Out of memory / slow responses
**Causes:**
- VRAM exhausted — ENML automatically reserves 500MB as a hard buffer and 2GB for future automation systems. If you still run OOM, ensure no other background applications are dominating the GPU.
- Context too large — the system auto-trims and uses Context Distillation to shrink memories, but very long copy-pasted blocks can be heavy.
- Embedding module loading — The first run downloads the `bge-base-en` and reranker models (~350MB). They run purely on CPU to save VRAM for the main LLM.

### Docker permission errors
```bash
sudo usermod -aG docker $USER
# Log out and log back in
```
