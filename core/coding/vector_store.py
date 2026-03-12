from typing import List, Dict, Any, Optional
from ..vector.qdrant_client import QdrantManager
from ..vector.embeddings import EmbeddingService
from ..config import QDRANT_CODE_TASKS_COLLECTION, QDRANT_CODE_CONTEXT_COLLECTION, EMBED_DIM
from .models import CodingTask
from ..logger import get_logger
from qdrant_client.http import models

logger = get_logger(__name__)

class CodeVectorStore:
    def __init__(self):
        self.qdrant_manager = QdrantManager()
        self.embedding_service = EmbeddingService()
        self.collections_ensured = False
        self.ensure_collections()

    def ensure_collections(self) -> bool:
        if not getattr(self.qdrant_manager, "available", True):
            return False
            
        if self.collections_ensured:
            return True
            
        try:
            for coll in [QDRANT_CODE_TASKS_COLLECTION, QDRANT_CODE_CONTEXT_COLLECTION]:
                if not self.qdrant_manager.client.collection_exists(coll):
                    self.qdrant_manager.client.create_collection(
                        collection_name=coll,
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
                    logger.info(f"[CodeVectorStore] Created missing collection {coll}")
            self.collections_ensured = True
            return True
        except Exception as e:
            logger.warning(f"[CodeVectorStore] Failed to ensure Qdrant collections: {e}")
            return False

    def index_task(self, task: CodingTask) -> bool:
        if not self.ensure_collections():
            return False
            
        try:
            text = f"{task.title}. {task.description}"
            if task.implementation_plan:
                text += " Plan: " + " ".join(task.implementation_plan)
                
            vector = self.embedding_service.embed(text)
            
            self.qdrant_manager.client.upsert(
                collection_name=QDRANT_CODE_TASKS_COLLECTION,
                points=[
                    models.PointStruct(
                        id=task.task_id,
                        vector={"dense": vector},
                        payload=task.to_dict()
                    )
                ]
            )
            return True
        except Exception as e:
            logger.warning(f"[CodeVectorStore] Failed to index task: {e}")
            return False

    def index_code_snippet(self, code_id: str, code_text: str, metadata: Dict[str, Any]) -> bool:
        if not self.ensure_collections():
            return False
            
        try:
            vector = self.embedding_service.embed(code_text)
            payload = {"text": code_text}
            payload.update(metadata)
            
            self.qdrant_manager.client.upsert(
                collection_name=QDRANT_CODE_CONTEXT_COLLECTION,
                points=[
                    models.PointStruct(
                        id=code_id,
                        vector={"dense": vector},
                        payload=payload
                    )
                ]
            )
            return True
        except Exception as e:
            logger.warning(f"[CodeVectorStore] Failed to index code snippet: {e}")
            return False

    def search_similar_tasks(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.ensure_collections():
            return []
            
        try:
            vector = self.embedding_service.embed(query)
            results = self.qdrant_manager.client.search(
                collection_name=QDRANT_CODE_TASKS_COLLECTION,
                query_vector=("dense", vector),
                limit=limit
            )
            return [r.payload for r in results if r.payload]
        except Exception as e:
            logger.warning(f"[CodeVectorStore] Failed to search tasks: {e}")
            return []

    def search_code_context(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.ensure_collections():
            return []
            
        try:
            vector = self.embedding_service.embed(query)
            results = self.qdrant_manager.client.search(
                collection_name=QDRANT_CODE_CONTEXT_COLLECTION,
                query_vector=("dense", vector),
                limit=limit
            )
            # Standardize output format
            return [r.payload for r in results if r.payload]
        except Exception as e:
            logger.warning(f"[CodeVectorStore] Failed to search code context: {e}")
            return []
