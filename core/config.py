
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Root Paths (from .env or defaults)
ENML_ROOT = Path(os.getenv("ENML_ROOT", Path(__file__).resolve().parent.parent))
MEMORY_ROOT = Path(os.getenv("MEMORY_ROOT", ENML_ROOT / "memory"))

# Sub-paths
CONVERSATIONS_DIR = MEMORY_ROOT / "conversations"
PROJECTS_DIR = MEMORY_ROOT / "projects"
RESEARCH_DIR = MEMORY_ROOT / "research"
GRAPH_DIR = MEMORY_ROOT / "graph"

# Ensure directories exist
for d in [CONVERSATIONS_DIR, PROJECTS_DIR, RESEARCH_DIR, GRAPH_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Security
ALLOWED_PATHS = [Path(p.strip()) for p in os.getenv("ALLOWED_PATHS", "").split(",") if p.strip()]
if not ALLOWED_PATHS:
    ALLOWED_PATHS = [ENML_ROOT, Path.home() / "Projects", Path.home() / "Research"]

# Vector DB (Qdrant) Configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_RESEARCH_COLLECTION = os.getenv("QDRANT_RESEARCH_COLLECTION", "research_collection")
QDRANT_PROJECT_COLLECTION = os.getenv("QDRANT_PROJECT_COLLECTION", "project_collection")
QDRANT_CONVERSATION_COLLECTION = os.getenv("QDRANT_CONVERSATION_COLLECTION", "conversation_collection")
QDRANT_PROFILE_COLLECTION = os.getenv("QDRANT_PROFILE_COLLECTION", "profile_collection")
QDRANT_KNOWLEDGE_COLLECTION = os.getenv("QDRANT_KNOWLEDGE_COLLECTION", "knowledge_collection")
QDRANT_DOCUMENT_COLLECTION = os.getenv("QDRANT_DOCUMENT_COLLECTION", "document_collection")

# Model configuration
EMBEDDING_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM = int(os.getenv("EMBED_DIM", 384))
LLAMA_SERVER_URL = os.getenv("LLAMA_SERVER_URL", "http://localhost:8080")
AI_NAME = os.getenv("AI_NAME", "ENML Assistant")
AI_HINT = os.getenv("AI_HINT", "running on local hardware")

# Input & Extraction Limits
MAX_REALTIME_INPUT_CHARS = int(os.getenv("MAX_REALTIME_INPUT_CHARS", 500))
MAX_FACTS_PER_EXTRACTION = int(os.getenv("MAX_FACTS_PER_EXTRACTION", 10))
MAX_DOCUMENT_FACTS = int(os.getenv("MAX_DOCUMENT_FACTS", 25))
MAX_DOCUMENT_SUMMARIES = int(os.getenv("MAX_DOCUMENT_SUMMARIES", 15))
MIN_RETRIEVAL_CONFIDENCE = float(os.getenv("MIN_RETRIEVAL_CONFIDENCE", 0.30))
CONTEXT_SIZE = int(os.getenv("CONTEXT_SIZE", 4096))

# Web Server
WEB_SERVER_PORT = int(os.getenv("WEB_SERVER_PORT", 5000))

# Debug Mode (set ENML_DEBUG=1 to enable verbose console logging)
ENML_DEBUG = os.getenv("ENML_DEBUG", "0") == "1"
