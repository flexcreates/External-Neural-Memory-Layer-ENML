import threading
import urllib.error
import urllib.request
from qdrant_client import QdrantClient
from qdrant_client.http import models

from core.config import (
    QDRANT_URL, QDRANT_API_KEY,
    QDRANT_RESEARCH_COLLECTION, QDRANT_PROJECT_COLLECTION, QDRANT_CONVERSATION_COLLECTION,
    QDRANT_PROFILE_COLLECTION, QDRANT_KNOWLEDGE_COLLECTION, QDRANT_DOCUMENT_COLLECTION,
    QDRANT_EPISODIC_COLLECTION, EMBED_DIM
)
from core.logger import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()


class QdrantManager:
    """Singleton manager for Qdrant connections and collection lifecycle.
    
    Ensures only one QdrantClient connection is created and all required
    collections exist with the correct vector dimensions.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            with _lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)
                    instance.available = True
                    instance.collections = [
                        QDRANT_RESEARCH_COLLECTION,
                        QDRANT_PROJECT_COLLECTION,
                        QDRANT_CONVERSATION_COLLECTION,
                        QDRANT_EPISODIC_COLLECTION,
                        QDRANT_PROFILE_COLLECTION,
                        QDRANT_KNOWLEDGE_COLLECTION,
                        QDRANT_DOCUMENT_COLLECTION,
                    ]
                    if instance._is_reachable():
                        instance._ensure_collections()
                    else:
                        instance.available = False
                        logger.warning(
                            f"Qdrant is not reachable at {QDRANT_URL}. "
                            "ENML will continue in degraded mode without vector memory until Qdrant is started."
                        )
                    cls._instance = instance
        return cls._instance

    def _is_reachable(self) -> bool:
        base_url = QDRANT_URL.rstrip("/")
        probe_urls = (f"{base_url}/readyz", f"{base_url}/health", f"{base_url}/")
        for probe_url in probe_urls:
            try:
                request = urllib.request.Request(probe_url, method="GET")
                with urllib.request.urlopen(request, timeout=1.5) as response:
                    if 200 <= getattr(response, "status", 0) < 300:
                        return True
            except (urllib.error.URLError, TimeoutError, ValueError) as exc:
                logger.debug(f"Qdrant probe failed for '{probe_url}': {exc}")
        return False

    def _ensure_collections(self):
        """Creates any missing Qdrant collections with dense and sparse configured."""
        failures = 0
        for collection_name in self.collections:
            try:
                if not self.client.collection_exists(collection_name):
                    logger.info(f"Creating missing Qdrant collection (Hybrid): {collection_name}")
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "dense": models.VectorParams(
                                size=EMBED_DIM,
                                distance=models.Distance.COSINE
                            )
                        },
                        sparse_vectors_config={
                            "sparse": models.SparseVectorParams()
                        }
                    )
                else:
                    # Check if the existing collection needs upgrading for Hybrid Search (Sparse Vectors)
                    col_info = self.client.get_collection(collection_name)
                    config = col_info.config.params
                    if getattr(config, "sparse_vectors_config", None) is None or "sparse" not in getattr(config, "sparse_vectors_config", {}):
                        logger.info(f"Upgrading existing Qdrant collection '{collection_name}' to support Hybrid Search...")
                        self.client.update_collection(
                            collection_name=collection_name,
                            sparse_vectors_config={
                                "sparse": models.SparseVectorParams()
                            }
                        )
            except Exception as e:
                failures += 1
                logger.error(f"Failed to ensure collection '{collection_name}': {e}")
        if failures:
            self.available = False
            logger.warning(
                f"Qdrant is unavailable ({failures}/{len(self.collections)} collection checks failed). "
                "ENML will continue in degraded mode without vector memory until Qdrant is reachable."
            )
