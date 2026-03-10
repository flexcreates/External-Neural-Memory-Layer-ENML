from typing import Dict, Any, List, Optional
from datetime import datetime
from core.vector.qdrant_client import QdrantManager
from core.vector.embeddings import EmbeddingService
from qdrant_client.http import models
from core.logger import get_logger
import uuid

logger = get_logger(__name__)

class Retriever:
    """Handles insertions and semantic search across specific Qdrant collections."""
    def __init__(self):
        self.qdrant_manager = QdrantManager()
        self.embedding_service = EmbeddingService()

    def add_memory(self, collection: str, text: str, payload: Dict[str, Any], memory_id: Optional[str] = None):
        """Adds text context and its metadata to the specified collection."""
        if memory_id is None:
            memory_id = str(uuid.uuid4())
            
        vector = self.embedding_service.embed(text)
        
        # Add plain text for retrieval back to original form
        if "text" not in payload:
            payload["text"] = text
            
        self.qdrant_manager.client.upsert(
            collection_name=collection,
            points=[
                models.PointStruct(
                    id=memory_id,
                    vector=vector,
                    payload=payload
                )
            ]
        )
        logger.info(f"Added memory to {collection} with ID {memory_id}")

    def search(self, collection: str, query: str, limit: int = 5, 
               filter_conditions: Optional[list] = None,
               filter_dict: Optional[Dict[str, str]] = None) -> List[Dict[str, Any]]:
        """Hybrid Semantic Search against a specific collection.
        
        Args:
            collection: Qdrant collection name to search.
            query: Natural language query string.
            limit: Maximum number of results to return.
            filter_conditions: Raw Qdrant FieldCondition list (advanced).
            filter_dict: Simple key-value dict converted to exact-match filters.
        """
        logger.info(f"[RETRIEVE] Searching '{collection}' for: '{query[:80]}' (limit={limit})")
        query_vector = self.embedding_service.embed(query)
        logger.debug(f"[RETRIEVE] Embedding generated: {len(query_vector)}-dim vector")
        
        # We implicitly only ever want 'active' or 'alternative' facts, never 'superseded' ones
        must_not_conditions = [
            models.FieldCondition(
                key="status",
                match=models.MatchValue(value="superseded")
            )
        ]
        
        # Build must conditions from filter_dict and/or filter_conditions
        must_conditions = []
        if filter_conditions:
            must_conditions.extend(filter_conditions)
        if filter_dict:
            for key, value in filter_dict.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
        
        query_filter = models.Filter(
            must_not=must_not_conditions,
            must=must_conditions if must_conditions else None
        )
            
        # 1. Broad Vector Similarity Lookup
        try:
            response = self.qdrant_manager.client.query_points(
                collection_name=collection,
                query=query_vector,
                limit=limit * 2,  # Fetch double and rank locally
                query_filter=query_filter,
                with_payload=True
            )
            results = response.points
            logger.info(f"[RETRIEVE] Qdrant returned {len(results)} raw results from '{collection}'")
        except Exception as e:
            logger.error(f"[RETRIEVE] Qdrant search FAILED on '{collection}': {e}")
            return []
                
        # 2. Local Re-Ranking (Entity Match + Recency)
        query_lower = query.lower()
        
        scored_results = []
        for r in results:
            payload = r.payload
            
            # Base Vector Semantic Score (0.0 to 1.0)
            score = r.score 
            
            # Recency Boost (For context-dependent queries like "this project")
            timestamp_str = payload.get("timestamp")
            if timestamp_str:
                try:
                    ts = timestamp_str.replace("Z", "+00:00")
                    dt = datetime.fromisoformat(ts)
                    if dt.tzinfo is None:
                        age_seconds = (datetime.now() - dt).total_seconds()
                    else:
                        age_seconds = (datetime.now(dt.tzinfo) - dt).total_seconds()
                    
                    if age_seconds < 1800:       # < 30 mins
                        score += 0.08
                    elif age_seconds < 7200:     # < 2 hours
                        score += 0.04
                    elif age_seconds < 86400:    # < 1 day
                        score += 0.02
                except Exception:
                    pass
            
            # Entity keyword matching boost
            subject = payload.get("subject", "").lower()
            predicate = payload.get("predicate", "").lower()
            obj_val = payload.get("object", payload.get("object_literal", "")).lower()
            
            if subject and subject in query_lower:
                score += 0.2
            if predicate and predicate.replace('_', ' ') in query_lower:
                score += 0.1
            if obj_val and obj_val in query_lower:
                score += 0.15
                
            scored_results.append((score, {
                "id": r.id,
                "score": score,
                "payload": payload
            }))
            
        # Sort by hybrid score and limit
        scored_results.sort(key=lambda x: x[0], reverse=True)
        final_results = [item[1] for item in scored_results[:limit]]
        
        if final_results:
            logger.info(f"[RETRIEVE] Returning {len(final_results)} results (top score: {final_results[0]['score']:.3f})")
            for i, r in enumerate(final_results[:3]):
                p = r.get('payload', {})
                fact_str = p.get('text', f"{p.get('subject','')} {p.get('predicate','')} {p.get('object','')}")
                logger.debug(f"[RETRIEVE]   [{i+1}] score={r['score']:.3f} → {fact_str[:100]}")
        else:
            logger.warning(f"[RETRIEVE] No results found in '{collection}' for query: '{query[:60]}'")
                
        return final_results
