<p align="center">
  <h1 align="center">🧠 ENML — External Neural Memory Layer</h1>
  <p align="center">
    <em>Infinite memory for local AI systems. Your AI remembers everything, forever.</em>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-features">Features</a> •
    <a href="#-architecture">Architecture</a> •
    <a href="#-web-ui">Web UI</a> •
    <a href="#-configuration">Configuration</a> •
    <a href="docs/USER_GUIDE.md">User Guide</a> •
    <a href="docs/DEVELOPMENT.md">Dev Guide</a>
  </p>
</p>

---

## What is ENML?

**ENML** gives your local LLM (Llama 3, Mistral, etc.) persistent, long-term memory that survives across sessions. It builds a personal **Knowledge Graph** from your conversations — your name, hobbies, pets, hardware specs, projects — and recalls them instantly, forever.

No cloud. No API keys. **100% local.** Your data stays on your machine.

```
Session 1:  "My name is Flex, I have a pet lizard named Colu"
            → Extracted & stored as semantic triples

Session 2:  "What's my pet's name?"
            → "Your pet lizard's name is Colu."  ✅
```

---

## ⚡ Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime |
| Docker | 20+ | Qdrant vector database |
| llama.cpp | Latest | Local LLM inference server |
| GGUF Model | Any | Recommended: `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` |

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/flexcreates/ENML.git
cd ENML

# 2. Run the setup script
chmod +x setup.sh
./setup.sh

# 3. Configure your environment
nano .env
# ↳ Set MODEL_PATH, LLAMA_SERVER, and AI_NAME

# 4. Start the services
./run_server.sh        # Terminal 1: Start Llama server
./run_qdrant.sh        # Terminal 2: Start Qdrant (if not started by setup)

# 5a. Start chatting (CLI)!
source .venv/bin/activate
python3 chat.py

# 5b. Or launch the Web UI
./run_web.sh           # Opens at http://localhost:5000
```

---

## ✨ Features

### 🧠 Real-Time Fact Extraction
Every message is analyzed by the LLM to extract semantic triples:
- `user has_name Flex` · `user has_pet lizard` · `user has_interest vibe-coding`
- Multi-value support: hobbies, pets, and projects stack without overwriting
- **Capped extraction**: max 10 facts per message to prevent memory overload

### 📄 Document Ingestion & Summarization Pipeline
Paste large documents (README files, documentation, code) and ENML handles them intelligently:
- **Auto-detection**: inputs over 500 chars or with markdown structure are classified as documents
- **Section chunking**: documents split by headings/paragraphs for focused processing
- **LLM-powered categorization & summarization**: each section is summarized by the LLM, and the document is classified into `project`, `research`, or `document` categories.
- **Categorized dual-layer storage**: summaries stored in categorized collections for domain-specific retrieval + facts extracted into `knowledge_collection`
- **Confidence-scored retrieval**: retrieved items carry semantic similarity scores; only items above threshold are injected
- **Noise filtering**: code blocks, ASCII art, URLs, and file paths are stripped before processing
- **Caps**: max 15 summaries + 25 facts per document to prevent memory explosion

```
You: [paste a 300-line README.md]
📄 Large input detected (9378 chars, 281 lines) — ingesting as document...
✅ Document ingested: 29 sections, 12 summaries, 25 facts extracted
```

### 🌐 Web UI
A beautiful dark-themed chat interface at `http://localhost:5000`:
- Full ENML pipeline — same memory, extraction, and knowledge graph as CLI
- SSE streaming for real-time response display
- Document paste support with ingestion feedback
- Session management across browser tabs

### 🔍 Smart Query Routing & Confidence Retrieval
The context builder searches Qdrant for relevant memories and injects them into the system prompt — so the AI always has context about you.
- **Confidence scoring**: every retrieved item carries a semantic similarity score
- **Threshold filtering**: only items above minimum confidence (0.30) are injected
- **Hybrid & Fallback retrieval**: queries target specific domains (e.g., project code), falling back to other document collections if needed, alongside knowledge facts
- **Recency boosting**: recently ingested documents receive a score boost, helping the AI understand context-dependent phrases like "explain this project"
- **Memory depth**: retrieves up to 10 document summary chunks to reconstruct full context
- **Memory cap**: max 8 scored items injected per query
- **Time Awareness**: Agent knows the exact real-world time at the moment of query processing, preventing chronological hallucination.
- **Deduplication**: redundant memories are filtered before injection
- **Token budget**: 6000-token context window for rich responses

### 🛡️ Identity Separation & Auto-Aging
AI identity and user identity are managed through a deterministic `identity.json` file that supersedes vector memory, guaranteeing zero identity drift.
- `"My name is Flex"` → stored as `user.name = Flex` (Identity Module & Knowledge Graph)
- `"Your mood is angry"` → stored as `assistant.personality_mood = angry` (Identity Module)
- **Auto-Aging:** The AI intuitively calculates its exact age globally across instances (`(now - creation_date).days + 1`).
- **Prompt Routing:** Users can map custom prompt engineering parameters seamlessly in `identity.json`.

### 📊 Knowledge Graph
Facts are stored as semantic triples with contradiction detection:
- `user has_name Flex` (active) → `user has_name David` (supersedes old value)
- Multi-value predicates (hobbies, pets) allow multiple active values
- **50+ multi-value predicates** including singular and plural forms

### 🧹 Intelligent Filtering
- **Question pre-check**: questions and commands skip extraction entirely
- **Document detection**: structured content (markdown, tables, code) bypasses real-time extraction
- **Noise filter**: greetings, filler, and structural predicates are rejected before storage
- **Name guard**: device/brand names can't overwrite your identity
- **Predicate normalization**: `uses Ubuntu` → `uses_os Ubuntu`, `has_hobbies` → `has_hobby`

### 🔒 100% Local & Private
- All processing happens on your machine
- No cloud APIs, no telemetry, no data leaves your system
- Qdrant runs in a local Docker container
- Embedding model runs on CPU (VRAM reserved for LLM)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              chat.py (CLI)  ·  web_server.py (Browser)      │
│                    ↓ InputClassifier ↓                      │
│              ┌─────────────┬──────────────────┐             │
│              │ Conversation │    Document     │             │
│              │   (normal)   │  (batch ingest) │             │
│              └──────┬──────┴────────┬─────────┘             │
├─────────────────────┼───────────────┼───────────────────────┤
│                   core/orchestrator.py                      │
│        ┌──────────┬───────────────┬───────────────┐         │
│        │ Extractor│ MemoryManager │ ContextBuilder│         │
│        │(LLM-based│ (routes facts │ (builds prompt│         │
│        │ fact     │  to correct   │  with memory  │         │
│        │ mining)  │  storage)     │  context)     │         │
│        └────┬─────┴──────┬────────┴──────┬────────┘         │
│             │            │               │                  │
│   ┌─────────┴──┐  ┌──────┴─────┐  ┌──────┴──────┐           │
│   │ Knowledge  │  │ Authority  │  │   Qdrant    │           │
│   │   Graph    │  │  Memory    │  │  (Vector    │           │
│   │(Triples +  │  │ (JSON for  │  │   Search)   │           │
│   │Enrichment) │  │ AI name/   │  │             │           │
│   │            │  │ role)      │  │             │           │
│   └────────────┘  └────────────┘  └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Llama.cpp Server (localhost:8080) │ Qdrant (localhost:6333)│
└─────────────────────────────────────────────────────────────┘
```

### Core Modules

| Module | Purpose |
|---|---|
| `chat.py` | CLI interface with multi-line paste support and input classification |
| `web_server.py` | Flask web UI with SSE streaming and full ENML pipeline |
| `core/orchestrator.py` | Main pipeline: extract → store → build context → stream response |
| `core/memory/extractor.py` | LLM-based fact extraction with document detection and noise filtering |
| `core/memory/document_ingester.py` | LLM-summarized document ingestion: chunk → clean → summarize → store |
| `core/memory_manager.py` | Routes facts to Knowledge Graph, Authority Memory, or Qdrant |
| `core/knowledge_graph.py` | Semantic triple storage with contradiction detection (50+ multi-value predicates) |
| `core/memory/authority_memory.py` | Deterministic JSON profile for AI and user identity |
| `core/context_builder.py` | Builds LLM prompt with confidence-scored memory injection |
| `core/config.py` | Centralized `.env` loader with configurable limits |
| `core/vector/retriever.py` | Qdrant semantic search with re-ranking |
| `core/vector/embeddings.py` | Thread-safe singleton embedding service (CPU-optimized) |
| `core/vector/qdrant_client.py` | Thread-safe singleton Qdrant connection manager |

---

## 🌐 Web UI

ENML includes a built-in web chat interface powered by Flask:

```bash
# Start the web server
./run_web.sh
# Open http://localhost:5000 in your browser
```

**Features:**
- Dark-themed modern interface with Inter & JetBrains Mono fonts
- Real-time SSE streaming responses
- Paste documents directly — auto-detected and batch-processed
- Same ENML memory pipeline as CLI (extraction, knowledge graph, authority memory)
- Session management with conversation history
- Health check endpoint at `/api/health`

The web UI port is configurable via `WEB_SERVER_PORT` in `.env` (default: `5000`).

---

## ⚙️ Configuration

All configuration is managed through `.env`. No hardcoded values.

### Key Settings

```bash
# Your AI's name
AI_NAME=Jarvis

# LLM model and server
MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf
LLAMA_SERVER=/path/to/llama.cpp/build/bin/llama-server
LLAMA_SERVER_URL=http://localhost:8080

# Qdrant vector database
QDRANT_URL=http://localhost:6333

# Embedding model
EMBED_MODEL=all-MiniLM-L6-v2
EMBED_DIM=384

# Context & Extraction Limits (v3.0)
CONTEXT_SIZE=4096                # LLM context window size
MAX_REALTIME_INPUT_CHARS=500     # Threshold for document detection
MAX_FACTS_PER_EXTRACTION=10      # Max facts per single extraction call
MAX_DOCUMENT_FACTS=25            # Max facts per document ingestion
MAX_DOCUMENT_SUMMARIES=15        # Max LLM-generated summaries per document
MIN_RETRIEVAL_CONFIDENCE=0.30    # Minimum confidence score for memory injection

# Web UI
WEB_SERVER_PORT=5000             # Web chat UI port
```

See [`.env.example`](.env.example) for the complete configuration reference.

---

## 📁 Project Structure

```
ENML/
├── chat.py                 # CLI chat interface (paste-safe, input classification)
├── web_server.py           # Flask web chat UI with SSE streaming
├── setup.sh                # One-command installer
├── run_server.sh           # Llama.cpp server launcher (dynamic GPU-aware VRAM)
├── run_qdrant.sh           # Qdrant Docker manager
├── run_web.sh              # Web UI startup script
├── reset_memory.sh         # Memory wipe utility
├── requirements.txt        # Python dependencies (includes Flask)
├── .env.example            # Configuration template
│
├── templates/              # Web UI
│   └── chat.html           # Dark-themed chat interface
│
├── core/                   # Core engine
│   ├── config.py           # .env loader with configurable limits
│   ├── orchestrator.py     # Main pipeline
│   ├── memory_manager.py   # Fact routing
│   ├── context_builder.py  # Prompt builder (capped, deduplicated)
│   ├── knowledge_graph.py  # Triple store (50+ multi-value predicates)
│   ├── logger.py           # Logging system
│   ├── memory/             # Memory subsystem
│   │   ├── extractor.py    # LLM fact extraction + document detection
│   │   ├── document_ingester.py  # Batch document ingestion pipeline
│   │   ├── authority_memory.py   # JSON identity store
│   │   └── triple_memory.py      # Triple data class
│   ├── vector/             # Vector subsystem
│   │   ├── retriever.py    # Qdrant search
│   │   ├── embeddings.py   # Embedding service (CPU-optimized)
│   │   └── qdrant_client.py # Connection manager
│   ├── router/             # Query routing
│   │   └── query_router.py
│   └── storage/            # Session storage
│       └── json_storage.py
│
├── research/               # Web research module
│   └── web_ingestor.py
├── tools/                  # File operations
│   └── file_tool.py
├── docs/                   # Documentation
│   ├── USER_GUIDE.md
│   ├── ARCHITECTURE.md
│   ├── DEVELOPMENT.md
│   └── WEB_CONNECTIVITY.md
│
├── ingest_conversation.py  # Batch conversation import
├── ingest_project.py       # Project codebase import
└── ingest_research.py      # Research data import
```

---

## 🧪 Chat Commands

| Command | Description |
|---|---|
| `/remember <text>` | Manually save a fact to memory |
| `exit` | End the session and save conversation |
| `Ctrl+C` | Force quit (session still saves) |

**Paste support**: Multi-line content pasted in the terminal is automatically buffered into a single message. Documents are detected and ingested through the batch pipeline.

---

## 🔧 Shell Scripts

| Script | What it does |
|---|---|
| `setup.sh` | Complete installation: venv, deps, .env, dirs, Qdrant |
| `run_server.sh` | Starts llama-server with dynamic GPU-aware VRAM and configurable context size (default 4096) |
| `run_qdrant.sh` | Manages Qdrant Docker container lifecycle |
| `run_web.sh` | Starts the ENML web chat UI (Flask) |
| `reset_memory.sh` | Wipes all memory, sessions, and vector collections |

---

## 📖 Documentation

| Document | Description |
|---|---|
| [User Guide](docs/USER_GUIDE.md) | Step-by-step usage workflows |
| [Architecture](docs/ARCHITECTURE.md) | System design & data flow diagrams |
| [Development Guide](docs/DEVELOPMENT.md) | How to extend ENML |
| [Web Connectivity](docs/WEB_CONNECTIVITY.md) | Internet research integration |

---

## 🛣️ Roadmap

- [x] Real-time fact extraction from conversations
- [x] Persistent knowledge graph with contradiction detection
- [x] Multi-value predicate support (hobbies, pets, projects)
- [x] AI/user identity separation
- [x] Question/command pre-filtering
- [x] Predicate normalization (singular/plural, content-based)
- [x] Document ingestion pipeline (batch processing for large inputs)
- [x] Web chat UI with full ENML pipeline
- [x] Multi-line paste handling (terminal & browser)
- [x] Memory injection limits and deduplication
- [x] Configurable context window (default 4096 tokens)
- [x] CPU-optimized embedding model (VRAM reserved for LLM)
- [x] Document summarization pipeline (LLM-powered section summaries)
- [x] Confidence-scored retrieval with threshold filtering
- [ ] Web research ingestion pipeline
- [ ] Multi-modal memory (images, documents)
- [ ] Conversation summarization
- [ ] Memory decay and confidence aging
- [ ] Plugin system for custom extractors

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

**Created by [Flex](https://github.com/flexcreates)** · 2024–2026

---

<p align="center">
  <em>ENML — Because your AI should remember who you are.</em>
</p>
