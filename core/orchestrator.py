
from typing import List, Dict, Any, Generator
from datetime import datetime
from openai import OpenAI
import threading
from .logger import get_logger
from .config import LLAMA_SERVER_URL, EMBEDDING_MODEL, QDRANT_CONVERSATION_COLLECTION
from .memory_manager import MemoryManager
from .context_builder import ContextBuilder

logger = get_logger(__name__)

class Orchestrator:
    def __init__(self):
        self.client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key")
        self.memory_manager = MemoryManager()
        self.context_builder = ContextBuilder(self.memory_manager)
        
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
            self.memory_manager.update_profile(user_input, conversation_history=history)

        # 2. Build Context
        # For this implementation, we assume 'history' contains the conversation SO FAR.
        # We append the NEW user message to the context sent to LLM.
        
        full_context, temperature = self.context_builder.build_context(user_input, history, system_prompt=system_prompt)
        
        # Append current user message if not already in history
        # (It shouldn't be in history yet if we want to save it strictly after)
        full_context.append({"role": "user", "content": user_input})
        
        # 4. Call LLM
        logger.info(f"[LLM] Calling model=Meta-Llama-3-8B-Instruct, temp={temperature}, messages={len(full_context)}")
        if full_context and full_context[0].get("role") == "system":
            logger.debug(f"[LLM] System prompt preview: {full_context[0]['content'][:400]}...")
        try:
            stream = self.client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                messages=full_context,
                stream=True,
                temperature=temperature,
                top_p=0.9,
                max_tokens=1000
            )
            
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    yield content
                    full_response += content
                    
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
            
        except Exception as e:
            logger.error(f"LLM Call Failed: {e}")
            yield f"Error: {str(e)}"

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
            response = self.client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )
            summary = response.choices[0].message.content.strip()
            
            payload = {
                "session_id": session_id,
                "importance": 0.5,
                "type": "episodic_summary",
                "text": summary
            }
            
            self.memory_manager.retriever.add_memory(
                collection=QDRANT_CONVERSATION_COLLECTION,
                text=summary,
                payload=payload
            )
            logger.info("[EPISODIC] Successfully stored conversation summary.")
        except Exception as e:
            logger.error(f"[EPISODIC] Failed to summarize conversation: {e}")

    def save_session(self, session_id: str, messages: List[Dict[str, str]]):
        return self.memory_manager.save_session(session_id, messages)
