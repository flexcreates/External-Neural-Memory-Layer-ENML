from typing import List, Dict, Any, Optional
from pathlib import Path

from .config import (CONVERSATIONS_DIR, QDRANT_CONVERSATION_COLLECTION, 
                     QDRANT_PROFILE_COLLECTION, QDRANT_KNOWLEDGE_COLLECTION,
                     QDRANT_DOCUMENT_COLLECTION, QDRANT_PROJECT_COLLECTION,
                     QDRANT_RESEARCH_COLLECTION, QDRANT_EPISODIC_COLLECTION)
from .storage.json_storage import JSONStorage
from .vector.retriever import Retriever
from .router.query_router import QueryRouter
from .memory.authority_memory import AuthorityMemory
from .memory.extractor import MemoryExtractor
from .memory.consolidation_service import MemoryConsolidationService
from .memory.lifecycle_service import MemoryLifecycleService
from .memory.record_repository import MemoryRecordRepository
from .memory.types import EvidenceItem, EvidencePacket, MemoryRecord, MemoryType
from .retrieval import RetrievalPolicyEngine
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
        self.record_repository = MemoryRecordRepository()
        self.consolidator = MemoryConsolidationService(self.record_repository)
        self.lifecycle = MemoryLifecycleService(self.record_repository, self.feedback)
        self.retrieval_policy_engine = RetrievalPolicyEngine()

    def save_session(self, session_id: str, messages: List[Dict[str, Any]]) -> Path:
        """Saves a full conversation session."""
        return self.json_storage.save_session(session_id, messages)

    def retrieve_context(self, query: str, n_results: int = 5, model_profile=None) -> dict:
        """Policy-driven retrieval that returns both backward-compatible docs and an evidence packet."""
        from .config import MIN_RETRIEVAL_CONFIDENCE
        
        collection = self.query_router.route(query)
        original_collection = collection
        model_profile_name = getattr(model_profile, "name", "medium")
        
        # Force knowledge collection for self-referential queries
        query_lower = query.lower()
        self_words = ["my", "i", "me", "i'm"]
        if collection == QDRANT_CONVERSATION_COLLECTION:
            if any(f" {w} " in f" {query_lower} " or query_lower.startswith(w) for w in self_words):
                collection = QDRANT_KNOWLEDGE_COLLECTION
                logger.info(f"[ROUTE] Self-referential override: {original_collection} → {collection}")
        policy = self.retrieval_policy_engine.resolve(query, collection, model_profile_name)
        logger.info(f"[ROUTE] Query '{query[:60]}' → collection: {collection}")
        
        scored_items = []
        scored_items.extend(self._load_authority_identity_items(query))
        search_collections = list(policy.primary_collections)
        if model_profile_name == "medium":
            search_collections.extend(policy.secondary_collections)
        search_collections = list(dict.fromkeys(search_collections))

        for c in search_collections:
            try:
                per_collection_limit = max(policy.limits.values()) if policy.limits else 5
                summary_results = self.retriever.search(c, query, limit=per_collection_limit)
                for r in summary_results:
                    payload = r.get("payload", {})
                    score = r.get("score", 0)
                    quality_score = self.feedback.get_memory_quality_score(str(r.get("id", "")))
                    score = (score * 0.85) + (quality_score * 0.15)
                    text = payload.get("text", "")
                    heading = payload.get("heading", "")
                    memory_type = payload.get("memory_type", self._infer_memory_type(c, payload))
                    
                    if score >= MIN_RETRIEVAL_CONFIDENCE and text:
                        scored_items.append({
                            "id": str(r.get("id", "")),
                            "text": text,
                            "heading": heading,
                            "score": round(score, 3),
                            "type": memory_type,
                            "source_collection": c
                        })
            except Exception as e:
                logger.debug(f"[RETRIEVE] Search failed on {c}: {e}")
                
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
            logger.info(f"[RETRIEVE] Found {summary_count} candidate memories across policy collections")

        scored_items.sort(key=lambda x: x["score"], reverse=True)
        max_items = getattr(model_profile, "max_evidence_items", 8) if model_profile else 8
        scored_items = scored_items[:max_items]
        evidence_packet = self._build_evidence_packet(policy, scored_items)
        
        all_docs = []
        for item in scored_items:
            if item["type"] in {MemoryType.DOCUMENT.value, MemoryType.PROJECT.value, MemoryType.RESEARCH.value} and item["heading"]:
                all_docs.append(f"📄 [{item['heading']}]: {item['text']}")
            elif item["type"] in {MemoryType.FACT.value, MemoryType.IDENTITY.value, MemoryType.PREFERENCE.value, MemoryType.SEMANTIC_CLAIM.value}:
                all_docs.append(f"📌 {item['text']}.")
            elif item["type"] == MemoryType.EPISODIC.value:
                all_docs.append(f"🧠 {item['text']}")
            else:
                all_docs.append(item["text"])
        
        if all_docs:
            fact_count = sum(1 for i in scored_items if i["type"] == "fact")
            logger.info(f"[RETRIEVE] Returning {len(all_docs)} items ({summary_count} summaries + {fact_count} facts)")
            for i, item in enumerate(scored_items[:5]):
                logger.debug(f"[RETRIEVE]   [{i}] score={item['score']:.3f} type={item['type']} → {item['text'][:100]}")
        else:
            if policy.name == "general_chat":
                logger.info(f"[RETRIEVE] Skipped memory retrieval for general task: '{query[:60]}'")
            else:
                logger.warning(f"[RETRIEVE] Zero memories above confidence threshold for: '{query[:60]}'")
            evidence_packet.no_relevant_memory = True
                
        return {
            "type": collection,
            "documents": all_docs,
            "scored_items": scored_items,
            "evidence_packet": evidence_packet,
            "policy": policy,
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

        self._capture_conversation_preferences(user_interaction)
        
        facts = self.extractor.extract_facts(user_input=user_interaction, conversation_context=context_str)
        if not facts and self._should_store_semantic_claim(user_interaction):
            semantic_record = self._build_semantic_claim_record(user_interaction)
            self._store_memory_record(semantic_record)
            self.retriever.add_memory(
                collection=QDRANT_KNOWLEDGE_COLLECTION,
                text=semantic_record.claim_text,
                payload={
                    "text": semantic_record.claim_text,
                    "memory_type": semantic_record.memory_type,
                    "subject": semantic_record.subject,
                    "confidence": semantic_record.confidence,
                    "status": semantic_record.status,
                    "source": semantic_record.source_type,
                    "namespace": semantic_record.namespace,
                    "tags": semantic_record.tags,
                }
            )
            logger.info(f"Stored semantic claim: {semantic_record.claim_text}")
            return
        
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
            namespace = "assistant.identity" if subject == "assistant" else "user.identity" if self._is_user_identity_fact(predicate) else f"{subject}.memory"
            payload = {
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "created_at": datetime.now().isoformat(),
                "last_used": None,
                "status": status,
                "source": "user_message",
                "namespace": namespace,
            }
            payload["text"] = f"{subject} {predicate} {obj}."
            
            logger.info(f"Stored fact: {subject} {predicate} {obj}")
            logger.info(f"MemoryManager: Processed Fact -> {subject} {predicate} {obj} [Status: {status}]")
            self._store_memory_record(self._build_memory_record(
                subject=subject,
                predicate=predicate,
                obj=obj,
                confidence=confidence,
                namespace=namespace,
                status=status,
                source_type="user_message",
            ))
            
            # Store in Qdrant with status metadata
            self.retriever.add_memory(
                collection=QDRANT_KNOWLEDGE_COLLECTION,
                text=payload["text"],
                payload=payload
            )

    def _infer_memory_type(self, collection: str, payload: Dict[str, Any]) -> str:
        predicate = payload.get("predicate", "")
        if collection == QDRANT_EPISODIC_COLLECTION:
            return MemoryType.EPISODIC.value
        if collection == QDRANT_PROJECT_COLLECTION:
            return MemoryType.PROJECT.value
        if collection == QDRANT_RESEARCH_COLLECTION:
            return MemoryType.RESEARCH.value
        if collection == QDRANT_DOCUMENT_COLLECTION:
            return MemoryType.DOCUMENT.value
        if self._is_user_identity_fact(predicate) or payload.get("namespace") == "user.identity":
            return MemoryType.IDENTITY.value
        if predicate == "likes":
            return MemoryType.PREFERENCE.value
        if payload.get("memory_type"):
            return payload["memory_type"]
        return MemoryType.FACT.value

    def _load_authority_identity_items(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        if not any(token in query_lower for token in ["my ", "who am i", "my name", "my age", "your name", "who are you"]):
            return []

        profile = self.authority_memory.load()
        items: List[Dict[str, Any]] = []
        user_data = profile.get("user", {})
        assistant_data = profile.get("assistant", {})

        if any(token in query_lower for token in ["my name", "who am i"]) and user_data.get("name"):
            items.append({
                "id": "authority_user_name",
                "text": f"user name {user_data['name']}",
                "heading": "Authority Identity",
                "score": 1.0,
                "type": MemoryType.IDENTITY.value,
                "source_collection": "authority_memory",
                "confidence": 0.99,
            })
        if "my age" in query_lower and user_data.get("age"):
            items.append({
                "id": "authority_user_age",
                "text": f"user age {user_data['age']}",
                "heading": "Authority Identity",
                "score": 1.0,
                "type": MemoryType.IDENTITY.value,
                "source_collection": "authority_memory",
                "confidence": 0.99,
            })
        if any(token in query_lower for token in ["your name", "who are you"]) and assistant_data.get("name"):
            items.append({
                "id": "authority_assistant_name",
                "text": f"assistant name {assistant_data['name']}",
                "heading": "Authority Identity",
                "score": 1.0,
                "type": MemoryType.IDENTITY.value,
                "source_collection": "authority_memory",
                "confidence": 0.99,
            })
        return items

    def _build_evidence_packet(self, policy, scored_items: List[Dict[str, Any]]) -> EvidencePacket:
        packet = EvidencePacket(policy_name=policy.name, answer_policy=policy.answer_policy)
        for item in scored_items:
            evidence = EvidenceItem(
                memory_id=item.get("id", ""),
                memory_type=item["type"],
                text=item["text"],
                score=item["score"],
                confidence=float(item.get("confidence", item.get("score", 0.5))),
                collection=item.get("source_collection", ""),
                metadata={"heading": item.get("heading", "")},
            )
            if evidence.memory_type == MemoryType.IDENTITY.value:
                packet.identity_items.append(evidence)
            elif evidence.memory_type in {MemoryType.FACT.value, MemoryType.PREFERENCE.value}:
                packet.fact_items.append(evidence)
            elif evidence.memory_type == MemoryType.EPISODIC.value:
                packet.episodic_items.append(evidence)
            elif evidence.memory_type == MemoryType.PROJECT.value:
                packet.project_items.append(evidence)
            elif evidence.memory_type == MemoryType.RESEARCH.value:
                packet.research_items.append(evidence)
            elif evidence.memory_type == MemoryType.DOCUMENT.value:
                packet.document_items.append(evidence)
            else:
                packet.semantic_items.append(evidence)
            if evidence.memory_id:
                self.feedback.log_retrieval(evidence.memory_id, was_used=False)
        return packet

    def _build_memory_record(self, subject: str, predicate: str, obj: str, confidence: float, namespace: str, status: str, source_type: str, memory_type: Optional[str] = None) -> MemoryRecord:
        resolved_memory_type = memory_type or self._resolve_memory_type_from_fact(predicate)
        return MemoryRecord(
            memory_type=resolved_memory_type,
            subject=subject,
            predicate_canonical=predicate,
            predicate_surface=predicate,
            object_value=obj,
            claim_text=f"{subject} {predicate} {obj}.",
            tags=self._derive_tags(f"{predicate} {obj}"),
            entities=[subject, obj],
            source_type=source_type,
            confidence=confidence,
            salience=min(1.0, confidence + 0.1),
            status=status,
            namespace=namespace,
            metadata={"predicate": predicate},
        )

    def _build_semantic_claim_record(self, user_interaction: str) -> MemoryRecord:
        return MemoryRecord(
            memory_type=MemoryType.SEMANTIC_CLAIM.value,
            subject="user",
            predicate_canonical=None,
            predicate_surface="semantic_claim",
            object_value=None,
            claim_text=user_interaction.strip(),
            tags=self._derive_tags(user_interaction),
            entities=["user"],
            source_type="user_message",
            confidence=0.6,
            salience=0.55,
            status="provisional",
            namespace="user.semantic",
            metadata={"dynamic_schema": True},
        )

    def _derive_tags(self, text: str) -> List[str]:
        stopwords = {"that", "this", "with", "from", "have", "your", "about", "there", "their", "would", "should", "could"}
        tags = []
        for token in text.lower().replace(".", " ").replace(",", " ").split():
            if len(token) < 4 or token in stopwords:
                continue
            tags.append(token)
        return sorted(set(tags))[:8]

    def _resolve_memory_type_from_fact(self, predicate: str) -> str:
        if self._is_user_identity_fact(predicate):
            return MemoryType.IDENTITY.value
        if predicate == "likes":
            return MemoryType.PREFERENCE.value
        return MemoryType.FACT.value

    def _store_memory_record(self, record: MemoryRecord):
        self.consolidator.submit(record)

    def _should_store_semantic_claim(self, text: str) -> bool:
        lowered = text.strip().lower()
        if len(lowered.split()) < 4:
            return False
        if lowered.endswith("?"):
            return False
        blocked_starts = ("what", "why", "how", "can you", "please", "show", "list", "tell me")
        return not lowered.startswith(blocked_starts)

    def _capture_conversation_preferences(self, user_interaction: str):
        text = user_interaction.strip().lower()
        stored = []

        if "stop asking" in text and "how can you assist" in text:
            if self.authority_memory.upsert_user_preference(
                "conversation_closer",
                "Avoid asking 'How can I assist you?' unless the user explicitly asks for help."
            ):
                stored.append(("conversation_closer", "avoid_assist_closer"))

        if "just talk to me" in text or "answer questions and learn things" in text:
            if self.authority_memory.upsert_user_preference(
                "conversation_style",
                "Keep the conversation natural, direct, and non-repetitive. Answer questions without repetitive help offers."
            ):
                stored.append(("conversation_style", "natural_direct"))

        if ("stop asking question" in text or "stop asking me questions" in text or "dont ask question" in text or "don't ask question" in text):
            if self.authority_memory.upsert_user_preference(
                "no_follow_up_questions",
                "Avoid repetitive or unnecessary follow-up questions. Ask them only when they add clear conversational value."
            ):
                stored.append(("no_follow_up_questions", "enabled"))

        for key, value in stored:
            self._store_memory_record(MemoryRecord(
                memory_type=MemoryType.PREFERENCE.value,
                subject="user",
                predicate_canonical=key,
                predicate_surface=key,
                object_value=value,
                claim_text=f"user preference {key} {value}.",
                tags=self._derive_tags(f"{key} {value}"),
                entities=["user"],
                source_type="conversation_preference",
                confidence=0.98,
                salience=0.95,
                namespace="user.preferences",
                metadata={"authority": True},
            ))
    
    # Predicates that genuinely describe the AI's identity (not hardware)
    _AI_IDENTITY_PREDICATES = {'has_name', 'is_named', 'preferred_name', 'name', 'called',
                                'age', 'created_date', 'creation_date',
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
        age_predicates = {'age'}
        creation_predicates = {'created_date', 'creation_date'}
        
        if predicate in name_predicates:
            key = "name"
        elif predicate in role_predicates:
            key = "role" 
        elif predicate in mood_predicates:
            key = "personality_mood"
        elif predicate in prompt_predicates:
            key = "prompt_engineering"
        elif predicate in age_predicates:
            key = "assistant_age_days"
        elif predicate in creation_predicates:
            key = "creation_date"
        else:
            key = predicate
        
        changed = self.authority_memory.upsert_fact("assistant", key, value)
        if changed:
            logger.info(f"MemoryManager: AI Identity Updated -> assistant.{key} = {value}")
            self._store_memory_record(MemoryRecord(
                memory_type=MemoryType.IDENTITY.value,
                subject="assistant",
                predicate_canonical=key,
                predicate_surface=predicate,
                object_value=value,
                claim_text=f"assistant {key} {value}.",
                tags=self._derive_tags(f"{key} {value}"),
                entities=["assistant", value],
                source_type="authority_memory",
                confidence=0.99,
                salience=1.0,
                namespace="assistant.identity",
                metadata={"authority": True},
            ))
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
            self._store_memory_record(MemoryRecord(
                memory_type=MemoryType.IDENTITY.value,
                subject="user",
                predicate_canonical=key,
                predicate_surface=predicate,
                object_value=value,
                claim_text=f"user {key} {value}.",
                tags=self._derive_tags(f"{key} {value}"),
                entities=["user", value],
                source_type="authority_memory",
                confidence=0.99,
                salience=1.0,
                namespace="user.identity",
                metadata={"authority": True},
            ))
    
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
