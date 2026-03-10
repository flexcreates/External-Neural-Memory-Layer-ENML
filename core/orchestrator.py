
from typing import List, Dict, Any, Generator
from datetime import datetime
import time
from openai import OpenAI
import threading
from .logger import get_logger
from .config import LLAMA_SERVER_URL, EMBEDDING_MODEL, QDRANT_EPISODIC_COLLECTION
from .citation_tracker import CitationTracker
from .memory_manager import MemoryManager
from .context_builder import ContextBuilder
from .memory.types import MemoryRecord, MemoryType
from .runtime_replay import RuntimeReplayLogger
from .router.model_router import ModelRouter
from .llm_runtime import detect_server_model
from .prompt_templates import build_chat_prompt_from_messages

logger = get_logger(__name__)

class Orchestrator:
    def __init__(self):
        self.client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key")
        self.memory_manager = MemoryManager()
        self.context_builder = ContextBuilder(self.memory_manager)
        self.model_router = ModelRouter(fixed_model=detect_server_model(self.client))
        self.citation_tracker = CitationTracker()
        self.runtime_replay_logger = RuntimeReplayLogger()

    def _response_max_tokens(self, user_input: str, model_name: str) -> int:
        text = (user_input or "").lower().strip()
        word_count = len(text.split())
        if word_count <= 8 and any(token in text for token in ["my ", "name", "age", "color", "weight", "pet", "who am i"]):
            return 96
        if word_count <= 10 and any(token in text for token in ["hi", "hello", "hey", "how are you"]):
            return 64
        if any(token in text for token in ["code", "python", "rust", "bug", "stack trace", "function", "class", "script"]):
            return 320 if "deepseek-coder" in model_name.lower() else 400
        if any(token in text for token in ["derive", "explain", "solve", "compare", "architecture", "tradeoff"]):
            return 400
        return 180

    def _tuned_temperature(self, user_input: str, model_name: str, base_temperature: float) -> float:
        text = (user_input or "").lower()
        if "deepseek-coder" in model_name.lower():
            if any(token in text for token in ["code", "python", "rust", "bug", "script", "function", "class"]):
                return min(base_temperature, 0.3)
            return min(base_temperature, 0.2)
        return base_temperature
        
    def process_message(self, 
                        user_input: str, 
                        session_id: str, 
                        history: List[Dict[str, str]],
                        system_prompt: str = "You are a helpful AI.",
                        skip_extraction: bool = False) -> Generator[str, None, None]:
        """
        Processes a user message through the ENML pipeline.
        
        Flow:
        1. Validate Input
        2. Retrieve Context (Memory, Profile) -> via ContextBuilder
        3. Build Prompt
        4. Call LLM (Stream)
        5. Store Result
        
        Args:
            skip_extraction: If True, skip fact extraction (used when document
                           ingestion was already handled by the caller).
        """
        logger.info(f"Processing message for session {session_id}")
        request_start = time.perf_counter()
        
        # 1. Knowledge Graph Query (Future)
        # 2. Tool Validation (Future)
        
        # 3. Build Context
        # We don't verify token limits here yet (TODO in Phase 5), but ContextBuilder has the logic placeholders
        # Add user input to history temporarily for context building if needed, 
        # but usually we want to pass the *previous* history + current input.
        
        # 1. Update Profile Immediately (Real-time Learning)
        # Pass recent conversation context so the extractor can resolve pronouns
        # like "its", "that", "this" (e.g., "its David" refers to the pet turtle)
        if not skip_extraction:
            extraction_start = time.perf_counter()
            self.memory_manager.update_profile(user_input, conversation_history=history)
            extraction_ms = (time.perf_counter() - extraction_start) * 1000
        else:
            extraction_ms = 0.0

        # 2. Build Context
        # For this implementation, we assume 'history' contains the conversation SO FAR.
        # We append the NEW user message to the context sent to LLM.
        model_name = self.model_router.route(user_input)
        model_profile = self.model_router.get_profile(model_name)
        context_start = time.perf_counter()
        prompt_text, temperature = self.context_builder.build_context(
            user_input,
            history,
            system_prompt=system_prompt,
            model_profile=model_profile,
            model_name=model_name,
        )
        context_ms = (time.perf_counter() - context_start) * 1000
        temperature = self._tuned_temperature(user_input, model_name, temperature)
        max_tokens = self._response_max_tokens(user_input, model_name)
        
        # 4. Call LLM
        logger.info(
            f"[LLM] Calling model={model_name}, temp={temperature}, max_tokens={max_tokens}, prompt_chars={len(prompt_text)}"
        )
        logger.debug(f"[LLM] Prompt preview: {prompt_text[:400]}...")
        try:
            llm_start = time.perf_counter()
            stream = self.client.completions.create(
                model=model_name,
                prompt=prompt_text,
                stream=True,
                temperature=temperature,
                top_p=0.9,
                max_tokens=max_tokens
            )
            
            full_response = ""
            for chunk in stream:
                content = chunk.choices[0].text or ""
                if content:
                    yield content
                    full_response += content

            evidence_packet = self.context_builder.last_evidence_packet
            llm_ms = (time.perf_counter() - llm_start) * 1000
            if evidence_packet is not None:
                cited = self.citation_tracker.track(session_id, user_input, full_response, evidence_packet)
                for item in cited:
                    if item["memory_id"]:
                        self.memory_manager.feedback.log_retrieval(item["memory_id"], was_used=True)
                self.runtime_replay_logger.log({
                    "session_id": session_id,
                    "query": user_input,
                    "model_name": model_name,
                    "model_profile": model_profile.name,
                    "policy_name": getattr(self.context_builder.last_retrieval_policy, "name", None),
                    "strict_grounding": getattr(model_profile, "strict_grounding", False) or getattr(getattr(self.context_builder, "last_retrieval_policy", None), "strict_grounding", False),
                    "evidence_count": len(evidence_packet.all_items()),
                    "evidence_items": [item.to_dict() for item in evidence_packet.all_items()],
                    "citations": cited,
                    "response_preview": full_response[:500],
                    "timings_ms": {
                        "extraction": round(extraction_ms, 3),
                        "context": round(context_ms, 3),
                        "llm": round(llm_ms, 3),
                        "total": round((time.perf_counter() - request_start) * 1000, 3),
                    },
                    "unsupported_claim_estimate": max(0, self._estimate_unsupported_claims(full_response, cited)),
                })
                    
            # 5. Post-Processing
            # Check if history is long enough to trigger an episodic summary chunk
            # history passed in doesn't include the current response yet, so if it's 19 items, 
            # with the current prompt and response it becomes 21. 
            if len(history) > 0 and len(history) % 20 == 0:
                # Fire off async summarization so it doesn't block the next user input
                threading.Thread(
                    target=self._summarize_and_store_episodic,
                    args=(history[-20:], session_id),
                    daemon=True
                ).start()
            if len(history) > 0 and len(history) % 10 == 0:
                threading.Thread(
                    target=self.memory_manager.lifecycle.run_once,
                    daemon=True
                ).start()
            
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            yield f"Error: {str(e)}"

    def _estimate_unsupported_claims(self, response_text: str, cited: List[Dict[str, Any]]) -> int:
        response_tokens = [token for token in response_text.split() if len(token) > 4]
        cited_text = " ".join(item.get("text", "") for item in cited).lower()
        unsupported = 0
        for token in response_tokens:
            cleaned = token.strip(".,!?;:()[]{}\"'").lower()
            if cleaned and cleaned not in cited_text:
                unsupported += 1
        return min(unsupported, 5)

    def _summarize_and_store_episodic(self, recent_history: List[Dict[str, str]], session_id: str):
        """Summarizes a chunk of conversation and stores it as an episodic memory event."""
        logger.info(f"[EPISODIC] Triggering conversation summarization for {len(recent_history)} messages.")
        
        convo_text = ""
        for msg in recent_history:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if role in ["user", "assistant"]:
                convo_text += f"{role.capitalize()}: {content}\n"
                
        prompt = (
            "Summarize the following conversation segment concisely. "
            "Focus strictly on the main topics discussed, decisions made, or important context shared. "
            "Do not include conversational filler. Keep it under 3-4 sentences.\n\n"
            f"Conversation:\n{convo_text}"
        )
        
        try:
            model_name = self.model_router.route(prompt)
            rendered_prompt = build_chat_prompt_from_messages(
                model_name,
                [{"role": "user", "content": prompt}],
            )
            response = self.client.completions.create(
                model=model_name,
                prompt=rendered_prompt,
                temperature=0.0,
                max_tokens=200
            )
            summary = response.choices[0].text.strip()
            
            payload = {
                "session_id": session_id,
                "importance": 0.5,
                "type": "episodic_summary",
                "text": summary
            }
            
            self.memory_manager.retriever.add_memory(
                collection=QDRANT_EPISODIC_COLLECTION,
                text=summary,
                payload=payload
            )
            self.memory_manager._store_memory_record(MemoryRecord(
                memory_type=MemoryType.EPISODIC.value,
                subject="conversation",
                predicate_canonical="summary",
                predicate_surface="summary",
                object_value=session_id,
                claim_text=summary,
                tags=self.memory_manager._derive_tags(summary),
                entities=["conversation", session_id],
                source_type="episodic_summary",
                source_ref=session_id,
                confidence=0.75,
                salience=0.7,
                namespace="conversation.episodic",
                metadata={"session_id": session_id},
            ))
            logger.info("[EPISODIC] Successfully stored conversation summary.")
        except Exception as e:
            logger.error(f"[EPISODIC] Failed to summarize conversation: {e}")

    def save_session(self, session_id: str, messages: List[Dict[str, str]]):
        return self.memory_manager.save_session(session_id, messages)
