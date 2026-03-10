from typing import List, Dict, Any, Optional, Tuple
import re
from .memory_manager import MemoryManager
from .time_provider import TimeProvider
from .logger import get_logger
from .context.distiller import ContextDistiller

logger = get_logger(__name__)

class ContextBuilder:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.distiller = ContextDistiller()
        
    def build_context(self, 
                      user_input: str, 
                      history: List[Dict[str, str]], 
                      system_prompt: str = "You are a persistent AI assistant named Jarvis.",
                      max_context_tokens: int = 3000) -> Tuple[List[Dict[str, str]], float]:
        """
        Builds the context and returns messages & temperature based on query mode.
        """
        logger.info(f"[INJECT] Building context for: '{user_input[:60]}'")
        
        # 1. Routing & Retrieval (confidence-scored)
        retrieval_data = self.memory_manager.retrieve_context(user_input, n_results=5)
        mode = retrieval_data["type"]
        docs = retrieval_data["documents"]
        
        logger.info(f"[INJECT] Retrieved {len(docs)} docs from mode='{mode}'")
        
        # Deduplicate docs by content to avoid redundant memory injection
        seen = set()
        unique_docs = []
        for doc in docs:
            doc_key = doc.strip().lower()
            if doc_key not in seen:
                seen.add(doc_key)
                unique_docs.append(doc)
        
        pre_dedup = len(docs)
        docs = unique_docs[:10]  # Hard cap at 10 memories
        if pre_dedup != len(docs):
            logger.debug(f"[INJECT] Deduplicated: {pre_dedup} → {len(docs)} unique docs")
            
        # Distill Context
        if docs:
            logger.info("[INJECT] Distilling retrieved memories...")
            distilled_summary = self.distiller.distill(user_input, docs)
            if distilled_summary:
                docs = [distilled_summary]
            else:
                docs = [] # Erase docs if completely irrelevant
        
        temperature = 0.6
        effective_system_prompt = system_prompt
        
        # 2. Mode-Specific Prompts
        if mode == "research_collection":
            temperature = 0.0
            context_str = "\n".join(docs)
            effective_system_prompt = (
                "You are a factual AI assistant.\n"
                "Use ONLY the context below.\n"
                "If answer not present, say you do not know.\n\n"
                f"Context:\n{context_str}\n\n"
                f"Question:\n{user_input}"
            )
        elif mode == "project_collection":
            temperature = 0.2
            context_str = "\n".join(docs)
            effective_system_prompt = (
                "You are an AI engineering assistant.\n"
                "Use the following project context to answer.\n\n"
                f"Context:\n{context_str}"
            )
        else: # Conversation / General Semantic Profile
            temperature = 0.6
        
        # 3. Dynamic Knowledge Sufficiency Feedback & System Time
        system_time = f"System Time: {TimeProvider.formatted()}"
        sufficiency_feedback = "Local Knowledge Confidence: HIGH" if docs and len(docs) > 0 else "Local Knowledge Confidence: LOW\nWeb Research Allowed: TRUE"
        
        # Merge retrieved memory (summaries + facts) into the system prompt
        # Items are already confidence-scored and sorted by memory_manager
        if docs:
            formatted_docs = "\n".join(docs)
            effective_system_prompt += (
                f"\n\nRelevant Graph Memory & Context:\n{formatted_docs}\n\n"
                f"{system_time}\n"
                f"{sufficiency_feedback}\n\n"
                f"IMPORTANT: Answer using ONLY the information provided above when applicable. "
                f"Items marked 📄 are detailed document summaries. "
                f"Items marked 📌 are remembered facts about the user/system. "
            )
            logger.info(f"[INJECT] ✅ Injected {len(docs)} confidence-scored items into system prompt")
            # Log individual scores from scored_items if available
            scored_items = retrieval_data.get("scored_items", [])
            for i, item in enumerate(scored_items[:5]):
                logger.debug(f"[INJECT]   [{i}] score={item.get('score', '?')} type={item.get('type', '?')} → {item.get('text', '')[:80]}")
        else:
            effective_system_prompt += (
                f"\n\n{system_time}\n"
                f"{sufficiency_feedback}\n\n"
                f"No specific Graph memories located. Answer using standard knowledge."
            )
            logger.warning(f"[INJECT] ⚠ No memories above confidence threshold (mode='{mode}')")
            
        # 4. Authority Identity Module Injection (Absolute Highest Priority)
        # This injects the permanent AI/User identity.json strings BEFORE the history.
        effective_system_prompt = self.memory_manager.authority_memory.get_injected_prompt(effective_system_prompt)
        logger.debug(f"[INJECT] Authority memory injected into prompt")
        messages = [{"role": "system", "content": effective_system_prompt}]
        
        # 4. Append Conversation History (with token budget enforcement)
        SLIDING_WINDOW_COUNT = 12
        recent_history = history[-SLIDING_WINDOW_COUNT:] if len(history) > SLIDING_WINDOW_COUNT else history
        
        # Calculate remaining token budget after system prompt
        system_tokens = self.estimate_tokens(effective_system_prompt)
        user_tokens = self.estimate_tokens(user_input)
        remaining_budget = max_context_tokens - system_tokens - user_tokens - 100  # 100 token safety margin
        
        # Trim history from oldest if it exceeds budget
        trimmed_history = []
        running_tokens = 0
        for msg in reversed(recent_history):
            msg_tokens = self.estimate_tokens(msg.get("content", ""))
            if running_tokens + msg_tokens > remaining_budget:
                break
            trimmed_history.insert(0, msg)
            running_tokens += msg_tokens
        
        messages.extend(trimmed_history)
        
        logger.info(f"[PROMPT] Final context: {len(messages)} messages, ~{system_tokens + running_tokens + user_tokens} tokens, temp={temperature}")
        logger.debug(f"[PROMPT] System prompt preview: {effective_system_prompt[:300]}...")
        
        return messages, temperature

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~1.3 tokens per whitespace-delimited word."""
        if not text:
            return 0
        return int(len(text.split()) * 1.3)
