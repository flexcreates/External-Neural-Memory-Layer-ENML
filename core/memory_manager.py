from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import (CONVERSATIONS_DIR, QDRANT_CONVERSATION_COLLECTION, 
                     QDRANT_PROFILE_COLLECTION, QDRANT_KNOWLEDGE_COLLECTION,
                     QDRANT_DOCUMENT_COLLECTION, QDRANT_PROJECT_COLLECTION,
                     QDRANT_RESEARCH_COLLECTION)
from .storage.json_storage import JSONStorage
from .vector.retriever import Retriever
from .router.query_router import QueryRouter
from .memory.authority_memory import AuthorityMemory
from .memory.extractor import MemoryExtractor
from .memory.triple_memory import MemoryTriple
from .knowledge_graph import MULTI_VALUE_PREDICATES
from .logger import get_logger
from datetime import datetime

logger = get_logger(__name__)

class MemoryManager:
    def __init__(self):
        """
        Initializes the MemoryManager with Qdrant, JSON backends, and Feedback.
        """
        from .memory_feedback import MemoryFeedbackSystem
        
        self.json_storage = JSONStorage(sessions_dir=CONVERSATIONS_DIR)
        self.retriever = Retriever()
        self.query_router = QueryRouter()
        self.authority_memory = AuthorityMemory()
        self.extractor = MemoryExtractor()
        self.feedback = MemoryFeedbackSystem()

    def save_session(self, session_id: str, messages: List[Dict[str, Any]]) -> Path:
        """Saves a full conversation session."""
        return self.json_storage.save_session(session_id, messages)

    def retrieve_context(self, query: str, n_results: int = 5) -> dict:
        """Confidence-scored hybrid retrieval: searches summaries + facts, returns scored items."""
        from .config import MIN_RETRIEVAL_CONFIDENCE
        
        collection = self.query_router.route(query)
        original_collection = collection
        
        # Force knowledge collection for self-referential queries
        query_lower = query.lower()
        self_words = ["my", "i", "me", "i'm"]
        if collection == QDRANT_CONVERSATION_COLLECTION:
            if any(f" {w} " in f" {query_lower} " or query_lower.startswith(w) for w in self_words):
                collection = QDRANT_KNOWLEDGE_COLLECTION
                logger.info(f"[ROUTE] Self-referential override: {original_collection} → {collection}")
        
        logger.info(f"[ROUTE] Query '{query[:60]}' → collection: {collection}")
        
        # ── Search document summaries from appropriate collection(s) ──
        scored_items = []
        
        # Content collections that hold summaries
        content_collections = [
            QDRANT_PROJECT_COLLECTION, 
            QDRANT_RESEARCH_COLLECTION, 
            QDRANT_DOCUMENT_COLLECTION
        ]
        
        # Determine which collections to search for summaries
        search_collections = []
        if collection in content_collections:
            # Search the specifically routed one first
            search_collections = [collection] + [c for c in content_collections if c != collection]
        else:
            # If routed to knowledge/conversation, still do a light search across content for context
            search_collections = content_collections
            
        for c in search_collections:
            try:
                summary_results = self.retriever.search(c, query, limit=10)
                for r in summary_results:
                    payload = r.get("payload", {})
                    score = r.get("score", 0)
                    text = payload.get("text", "")
                    heading = payload.get("heading", "")
                    
                    if score >= MIN_RETRIEVAL_CONFIDENCE and text:
                        scored_items.append({
                            "text": text,
                            "heading": heading,
                            "score": round(score, 3),
                            "type": "summary",
                            "source_collection": c
                        })
            except Exception as e:
                logger.debug(f"[RETRIEVE] Document summary search failed on {c}: {e}")
                
        # Deduplicate summaries just in case
        seen_summaries = set()
        unique_summaries = []
        for item in scored_items:
            if item["text"] not in seen_summaries:
                seen_summaries.add(item["text"])
                unique_summaries.append(item)
        scored_items = unique_summaries
        
        summary_count = len(scored_items)
        if summary_count:
            logger.info(f"[RETRIEVE] Found {summary_count} document summaries across collections (scores: {[s['score'] for s in scored_items]})")
        
        # ── Search for facts in Knowledge Collection ──
        # Facts are always strictly in QDRANT_KNOWLEDGE_COLLECTION
        fact_threshold = MIN_RETRIEVAL_CONFIDENCE + 0.05  # Slightly stricter for sparse facts
        results = self.retriever.search(QDRANT_KNOWLEDGE_COLLECTION, query, limit=n_results)
        for r in results:
            payload = r.get("payload", {})
            score = r.get("score", 0)
            
            # Apply time-decay to multi-value predicates (goals, interests, etc)
            predicate = payload.get("predicate", "")
            if predicate in MULTI_VALUE_PREDICATES:
                timestamp_str = payload.get("timestamp")
                if timestamp_str:
                    try:
                        timestamp = datetime.fromisoformat(timestamp_str)
                        age_days = (datetime.now() - timestamp).total_seconds() / (24 * 3600)
                        # ~5% drop per day, bottoming out at 20%
                        decay_factor = max(0.2, 1.0 - (0.05 * age_days))
                        
                        # Apply decay to the base retrieval score
                        # Ensure we multiply by the internal confidence (if it was penalized from denial)
                        internal_conf = payload.get("confidence", 1.0)
                        score = score * decay_factor * internal_conf
                    except ValueError:
                        pass
            
            if score < fact_threshold:
                continue
            
            if "subject" in payload and "predicate" in payload and "object" in payload:
                fact_text = f"{payload.get('subject')} {payload.get('predicate')} {payload.get('object')}"
            else:
                fact_text = payload.get("text", "")
            
            if fact_text:
                scored_items.append({
                    "text": fact_text,
                    "heading": "",
                    "score": round(score, 3),
                    "type": "fact",
                })
        
        # Sort all items by score descending, cap at 8
        scored_items.sort(key=lambda x: x["score"], reverse=True)
        scored_items = scored_items[:8]
        
        # Build plain doc list for backward compatibility
        all_docs = []
        for item in scored_items:
            if item["type"] == "summary" and item["heading"]:
                all_docs.append(f"📄 [{item['heading']}]: {item['text']}")
            elif item["type"] == "fact":
                all_docs.append(f"📌 {item['text']}.")
            else:
                all_docs.append(item["text"])
        
        if all_docs:
            fact_count = sum(1 for i in scored_items if i["type"] == "fact")
            logger.info(f"[RETRIEVE] Returning {len(all_docs)} items ({summary_count} summaries + {fact_count} facts)")
            for i, item in enumerate(scored_items[:5]):
                logger.debug(f"[RETRIEVE]   [{i}] score={item['score']:.3f} type={item['type']} → {item['text'][:100]}")
        else:
            logger.warning(f"[RETRIEVE] Zero memories above confidence threshold for: '{query[:60]}'")
                
        return {
            "type": collection,
            "documents": all_docs,
            "scored_items": scored_items,
        }

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a full session log."""
        return self.json_storage.load_session(session_id)

    def get_profile_summary(self) -> str:
        # Phase 5 placeholder
        return ""
        
    def update_profile(self, user_interaction: str, conversation_history: list = None):
        """Extract semantic triples and route them based on subject:
        
        - subject='assistant' → stored in AuthorityMemory (deterministic JSON profile)
        - subject='user' or other → stored in Knowledge Graph + Qdrant
        
        This prevents the AI's identity from colliding with the user's identity.
        
        Args:
            user_interaction: The user's current message.
            conversation_history: Recent messages for pronoun/context resolution.
        """
        from .knowledge_graph import EntityLinker, MULTI_VALUE_PREDICATES
        
        # Instantiate Linker with the embedding service instance from retriever
        entity_linker = EntityLinker(embedding_service=self.retriever.embedding_service)
        
        # Build conversation context string from last 3 messages
        context_str = ""
        if conversation_history:
            recent = conversation_history[-3:]  # Last 3 messages
            context_lines = []
            for msg in recent:
                role = msg.get("role", "user").capitalize()
                content = msg.get("content", "")[:200]  # Truncate long messages
                context_lines.append(f"{role}: {content}")
            context_str = "\n".join(context_lines)
        
        facts = self.extractor.extract_facts(user_input=user_interaction, conversation_context=context_str)
        
        for fact in facts:
            subject = fact.get("subject", "user").lower()
            predicate = fact.get("predicate", "").lower().replace(' ', '_')
            obj = fact.get("object", "")
            confidence = float(fact.get("confidence", 0.0))
            
            if not predicate or not obj:
                continue
            
            # ── ROUTE 1: Assistant identity facts → Authority Memory ──
            # ONLY name and role go to authority memory. Everything else (specs, etc.)
            # gets reclassified as user facts because hardware belongs to the user.
            if subject == "assistant":
                if self._is_ai_identity_fact(predicate):
                    self._store_assistant_fact(predicate, obj)
                    continue
                else:
                    # Reclassify: "assistant has_processor X" → "user has_processor X"
                    logger.info(f"Reclassifying assistant fact to user: {predicate} {obj}")
                    subject = "user"
                    fact["subject"] = "user"

            # ── ROUTE 1.5: User identity core facts → Authority Memory ──
            # Ensure critical user traits (name, age) get absolute priority in IdentityModule
            if subject == "user" and self._is_user_identity_fact(predicate):
                self._store_user_fact(predicate, obj)
                # We still allow it to flow into the Knowledge Graph as a standard fact for retrieval/history

            
            # ── ROUTE 2: User/entity facts → Knowledge Graph + Qdrant ──
            # Check if this is a multi-value predicate
            is_multi_value = predicate in MULTI_VALUE_PREDICATES
            
            if is_multi_value:
                # Search for existing similar facts to avoid exact duplicates
                existing = self._find_existing_fact(subject, predicate, obj)
                if existing:
                    logger.debug(f"Skipping duplicate: {subject} {predicate} {obj}")
                    continue
                # For multi-value, we don't check for "contradictions" - we just add
                status = "active"
            else:
                # For single-value, use the entity linker's contradiction detection
                enriched_fact = entity_linker.store_fact({
                    "subject": subject,
                    "predicate": predicate,
                    "object": obj,
                    "confidence": confidence
                })
                status = enriched_fact.status
            
            # Build payload
            payload = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "status": status
            }
            payload["text"] = f"{subject} {predicate} {obj}."
            
            logger.info(f"MemoryManager: Processed Fact -> {subject} {predicate} {obj} [Status: {status}]")
            
            # Store in Qdrant with status metadata
            self.retriever.add_memory(
                collection=QDRANT_KNOWLEDGE_COLLECTION,
                text=payload["text"],
                payload=payload
            )
    
    # Predicates that genuinely describe the AI's identity (not hardware)
    _AI_IDENTITY_PREDICATES = {'has_name', 'is_named', 'preferred_name', 'name', 'called',
                                'has_role', 'is_a', 'is_type', 'personality', 'tone', 'mood',
                                'prompt_engineering', 'rules', 'instructions'}
    
    def _is_ai_identity_fact(self, predicate: str) -> bool:
        """Returns True if this predicate describes the AI's identity, not hardware."""
        return predicate.lower() in self._AI_IDENTITY_PREDICATES

    # Core user identity predicates
    _USER_IDENTITY_PREDICATES = {'has_name', 'is_named', 'preferred_name', 'name', 'age'}

    def _is_user_identity_fact(self, predicate: str) -> bool:
        """Returns True if this predicate describes the User's core identity."""
        return predicate.lower() in self._USER_IDENTITY_PREDICATES

    def _store_assistant_fact(self, predicate: str, value: str):
        """Route assistant identity facts to Authority Memory.
        
        This keeps AI identity completely separated from user identity.
        """
        # Map predicates to authority memory keys
        name_predicates = {'has_name', 'is_named', 'preferred_name', 'name', 'called'}
        role_predicates = {'has_role', 'is_a', 'is_type'}
        mood_predicates = {'personality', 'tone', 'mood'}
        prompt_predicates = {'prompt_engineering', 'rules', 'instructions'}
        
        if predicate in name_predicates:
            key = "name"
        elif predicate in role_predicates:
            key = "role" 
        elif predicate in mood_predicates:
            key = "personality_mood"
        elif predicate in prompt_predicates:
            key = "prompt_engineering"
        else:
            key = predicate
        
        changed = self.authority_memory.upsert_fact("assistant", key, value)
        if changed:
            logger.info(f"MemoryManager: AI Identity Updated -> assistant.{key} = {value}")
        else:
            logger.debug(f"MemoryManager: AI identity unchanged: assistant.{key} = {value}")

    def _store_user_fact(self, predicate: str, value: str):
        """Route core User identity facts to Authority Memory for absolute priority."""
        name_predicates = {'has_name', 'is_named', 'preferred_name', 'name'}
        
        if predicate in name_predicates:
            key = "name"
        elif predicate == "age":
            key = "age"
        else:
            key = predicate

        changed = self.authority_memory.upsert_fact("user", key, value)
        if changed:
            logger.info(f"MemoryManager: User Identity Updated -> user.{key} = {value}")
    
    def _find_existing_fact(self, subject: str, predicate: str, obj: str) -> Optional[Dict]:
        """Check if exact fact already exists to prevent duplicates."""
        try:
            # Search for subject+predicate combinations
            query = f"{subject} {predicate}"
            results = self.retriever.search(
                QDRANT_KNOWLEDGE_COLLECTION, 
                query, 
                limit=5,
                filter_dict={"subject": subject, "predicate": predicate}
            )
            
            for r in results:
                payload = r.get("payload", {})
                if payload.get("object") == obj:
                    return payload
            return None
        except Exception as e:
            logger.debug(f"Error checking for existing fact: {e}")
            return None