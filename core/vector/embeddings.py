import threading
from sentence_transformers import SentenceTransformer
from core.config import EMBEDDING_MODEL
from core.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()


class EmbeddingService:
    """Singleton wrapper around SentenceTransformer.
    
    The model is loaded once on CPU and shared across all callers (Retriever,
    EntityLinker, ingestion scripts) to avoid duplicating the ~90 MB
    in-memory model. CPU is used intentionally because the LLM server
    already occupies most VRAM.
    
    local_files_only=True prevents network calls on every startup.
    The model must be cached in the default HuggingFace hub directory
    (~/.cache/huggingface/) from a prior run or manual download.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    logger.info(f"Loading embedding model: {EMBEDDING_MODEL} (on CPU)")
                    try:
                        instance.model = SentenceTransformer(
                            EMBEDDING_MODEL,
                            device="cpu",
                            local_files_only=True,
                        )
                    except OSError:
                        # First run: download and cache the model
                        logger.info(f"Model not cached locally, downloading once: {EMBEDDING_MODEL}")
                        instance.model = SentenceTransformer(
                            EMBEDDING_MODEL,
                            device="cpu",
                        )
                    cls._instance = instance
        return cls._instance

    def embed(self, text: str) -> list[float]:
        """Generates a dense vector embedding for the given text."""
        return self.model.encode(text).tolist()

class SparseEmbeddingService:
    """Singleton wrapper around fastembed BM25 SparseTextEmbedding."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    from fastembed import SparseTextEmbedding
                    instance = super().__new__(cls)
                    logger.info("Loading sparse embedding model: Qdrant/bm25")
                    instance.model = SparseTextEmbedding("Qdrant/bm25")
                    cls._instance = instance
        return cls._instance

    def embed(self, text: str):
        """Generates a sparse vector embedding (indices & values)."""
        result = list(self.model.embed([text]))[0]
        return result.indices.tolist(), result.values.tolist()
