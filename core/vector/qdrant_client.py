import threading
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
                    instance.collections = [
                        QDRANT_RESEARCH_COLLECTION,
                        QDRANT_PROJECT_COLLECTION,
                        QDRANT_CONVERSATION_COLLECTION,
                        QDRANT_EPISODIC_COLLECTION,
                        QDRANT_PROFILE_COLLECTION,
                        QDRANT_KNOWLEDGE_COLLECTION,
                        QDRANT_DOCUMENT_COLLECTION,
                    ]
                    instance._ensure_collections()
                    cls._instance = instance
        return cls._instance

    def _ensure_collections(self):
        """Creates any missing Qdrant collections with dense and sparse configured."""
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
                logger.error(f"Failed to ensure collection '{collection_name}': {e}")
