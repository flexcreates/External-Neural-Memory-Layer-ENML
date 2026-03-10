from typing import List, Dict, Any, Optional, Tuple
import re
from .memory_manager import MemoryManager
from .time_provider import TimeProvider
from .logger import get_logger
from .context.distiller import ContextDistiller
from .context.prompt_budget_manager import PromptBudgetManager
from .memory.types import EvidencePacket, MemoryType
from .prompt_templates import build_chat_prompt, get_model_template_info

logger = get_logger(__name__)

class ContextBuilder:
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.distiller = ContextDistiller()
        self.last_evidence_packet: Optional[EvidencePacket] = None
        self.last_retrieval_policy = None
        
    def build_context(self, 
                      user_input: str, 
                      history: List[Dict[str, str]], 
                      system_prompt: str = "You are a persistent AI assistant named Jarvis.",
                      max_context_tokens: int = 3000,
                      model_profile = None,
                      model_name: str = "") -> Tuple[str, float]:
        """
        Builds the grounded prompt string and returns it with temperature.
        """
        logger.info(f"[INJECT] Building context for: '{user_input[:60]}'")
        budget_manager = PromptBudgetManager(max_context_tokens)
        template_info = get_model_template_info(model_name)
        is_small_model = template_info.size_b is not None and template_info.size_b <= 3.5
        is_personal_query = self._is_personal_query(user_input)
        
        retrieval_data = self.memory_manager.retrieve_context(user_input, n_results=5, model_profile=model_profile)
        mode = retrieval_data["type"]
        evidence_packet: EvidencePacket = retrieval_data["evidence_packet"]
        self.last_evidence_packet = evidence_packet
        self.last_retrieval_policy = retrieval_data.get("policy")
        policy_name = getattr(self.last_retrieval_policy, "name", "")
        docs = self._render_evidence(evidence_packet)
        
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
        allow_distillation = getattr(model_profile, "allow_distillation", True)
        if docs and allow_distillation and len(docs) > 1:
            logger.info("[INJECT] Distilling retrieved memories...")
            distilled_summary = self.distiller.distill(user_input, docs)
            if distilled_summary:
                docs = [distilled_summary] + docs[:2]
            else:
                docs = []
        docs = budget_manager.trim_items(docs, self.estimate_tokens)
        if is_small_model and len(docs) > 2:
            docs = docs[:2]
            logger.info("[PROMPT] Small-model compact mode: limiting evidence injection to 2 items")

        preference_intent = self._detect_preference_intent(user_input)
        preference_evidence_hint = ""
        if preference_intent and docs:
            evidence_text = "\n".join(docs).lower()
            has_likes = " likes " in evidence_text
            has_loves = " loves " in evidence_text
            if preference_intent == "love" and not has_loves and has_likes:
                preference_evidence_hint = (
                    "\n\n<preference_resolution>\n"
                    "The user asked about things they love, but memory evidence only states things they like. "
                    "Answer with 'like' and clarify that no 'love' memory was found.\n"
                    "</preference_resolution>\n"
                )
            elif preference_intent == "like" and not has_likes and has_loves:
                preference_evidence_hint = (
                    "\n\n<preference_resolution>\n"
                    "The user asked about things they like, but memory evidence only states things they love. "
                    "Answer with 'love' and clarify that no 'like' memory was found.\n"
                    "</preference_resolution>\n"
                )
        
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
            temperature = 0.5 if policy_name == "conversation_policy" else 0.6
        
        # 3. Dynamic Knowledge Sufficiency Feedback & System Time
        system_time = f"System Time: {TimeProvider.formatted()}"
        strict_grounding = getattr(model_profile, "strict_grounding", False) or retrieval_data["policy"].strict_grounding
        sufficiency_feedback = "Local Knowledge Confidence: HIGH" if docs and len(docs) > 0 else "Local Knowledge Confidence: LOW\nWeb Research Allowed: TRUE"
        user_pref_rules = self._build_user_preference_rules()

        if docs:
            formatted_docs = "\n".join(docs)
            factual_recall_rule = ""
            if policy_name == "personal_memory":
                factual_recall_rule = (
                    "Treat direct user-stated memories as authoritative unless conflicting evidence is present. "
                    "Answer in plain declarative sentences without unnecessary qualifiers."
                )
            effective_system_prompt += (
                f"\n\n<knowledge>\n{formatted_docs}\n</knowledge>\n"
                f"\n<answer_policy>\n" + "\n".join(f"- {line}" for line in evidence_packet.answer_policy) + "\n</answer_policy>\n\n"
                f"\n<conversation_rules>\n{user_pref_rules}\n</conversation_rules>\n\n"
                f"{system_time}\n"
                f"{sufficiency_feedback}\n\n"
                f"IMPORTANT: {'Answer only from memory evidence when applicable.' if strict_grounding else 'Prefer memory evidence when applicable.'} "
                f"Keep distinctions between exact memory, episodic summary, and general knowledge explicit. "
                f"Avoid repetitive filler closers and do not ask 'How can I assist you?' unless the user asks for help or next steps. "
                f"If this is a memory recall reply, keep it compact and factual. "
                f"{factual_recall_rule}"
            )
            if preference_evidence_hint:
                effective_system_prompt += preference_evidence_hint
            logger.info(f"[INJECT] ✅ Injected {len(docs)} confidence-scored items into system prompt")
            # Log individual scores from scored_items if available
            scored_items = retrieval_data.get("scored_items", [])
            for i, item in enumerate(scored_items[:5]):
                logger.debug(f"[INJECT]   [{i}] score={item.get('score', '?')} type={item.get('type', '?')} → {item.get('text', '')[:80]}")
        else:
            if is_personal_query:
                temperature = min(temperature, 0.2)
            effective_system_prompt += (
                "\n\n<knowledge>\nNo relevant memories found.\n</knowledge>\n"
                "\n<answer_policy>\n- No relevant memories were retrieved.\n- Do not fabricate personal facts.\n- If the answer depends on memory, say you do not know.\n</answer_policy>\n"
                f"\n<conversation_rules>\n{user_pref_rules}\n</conversation_rules>\n"
                f"\n\n{system_time}\n"
                f"{sufficiency_feedback}\n\n"
                f"No specific Graph memories located. Answer using standard knowledge. Avoid repetitive filler closers."
            )
            if is_personal_query:
                effective_system_prompt += (
                    "\n\n<personal_memory_guard>\n"
                    "The user asked about personal facts. Since no memory evidence was retrieved, "
                    "do not guess or infer. Say you do not know and ask a brief clarification if needed.\n"
                    "</personal_memory_guard>\n"
                )
            logger.warning(f"[INJECT] ⚠ No memories above confidence threshold (mode='{mode}')")
            
        # 4. Authority Identity Module Injection (Absolute Highest Priority)
        # This injects the permanent AI/User identity.json strings BEFORE the history.
        effective_system_prompt = self.memory_manager.authority_memory.get_injected_prompt(effective_system_prompt)
        logger.debug(f"[INJECT] Authority memory injected into prompt")
        # 5. Append Conversation History (with token budget enforcement)
        SLIDING_WINDOW_COUNT = 6 if is_small_model else 12
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
        
        prompt_text = build_chat_prompt(
            model_name=model_name,
            system_prompt=effective_system_prompt,
            conversation_history=trimmed_history,
            user_message=user_input,
        )

        logger.info(
            f"[PROMPT] Final prompt: {1 + len(trimmed_history) + 1} messages, "
            f"~{system_tokens + running_tokens + user_tokens} source tokens, temp={temperature}"
        )
        logger.debug(f"[PROMPT] System prompt preview: {effective_system_prompt[:300]}...")
        logger.debug(f"[PROMPT] Final prompt preview: {prompt_text[:400]}...")
        
        return prompt_text, temperature

    def _build_user_preference_rules(self) -> str:
        profile = self.memory_manager.authority_memory.load()
        prefs = profile.get("user", {}).get("preferences", {})
        if not prefs:
            return "- Keep responses concise and natural."

        lines = []
        conversation_style = prefs.get("conversation_style")
        if conversation_style:
            lines.append(f"- {conversation_style}")
        closer = prefs.get("conversation_closer")
        if closer:
            lines.append(f"- {closer}")
        no_questions = prefs.get("no_follow_up_questions")
        if no_questions:
            lines.append(f"- {no_questions}")
        return "\n".join(lines) if lines else "- Keep responses concise and natural."

    def _is_personal_query(self, text: str) -> bool:
        lowered = (text or "").lower()
        if "my " in lowered or "mine" in lowered or "myself" in lowered:
            return True
        if lowered.startswith(("what is my", "what's my", "who am i", "do i", "am i", "is my")):
            return True
        return False

    def _detect_preference_intent(self, text: str) -> str:
        lowered = (text or "").lower()
        if "love" in lowered:
            return "love"
        if "like" in lowered:
            return "like"
        return ""

    def _render_evidence(self, evidence_packet: EvidencePacket) -> List[str]:
        sections: List[str] = []
        if evidence_packet.identity_items:
            sections.append(self._format_section("identity", evidence_packet.identity_items))
        if evidence_packet.fact_items:
            sections.append(self._format_section("retrieved_facts", evidence_packet.fact_items))
        if evidence_packet.semantic_items:
            sections.append(self._format_section("semantic_claims", evidence_packet.semantic_items))
        if evidence_packet.episodic_items:
            sections.append(self._format_section("episodic_context", evidence_packet.episodic_items))
        if evidence_packet.project_items:
            sections.append(self._format_section("project_context", evidence_packet.project_items))
        if evidence_packet.document_items:
            sections.append(self._format_section("document_context", evidence_packet.document_items))
        if evidence_packet.research_items:
            sections.append(self._format_section("research_context", evidence_packet.research_items))
        return sections

    def _format_section(self, name: str, items) -> str:
        lines = [f"<{name}>"]
        for item in items:
            lines.append(
                f"[id={item.memory_id}] {item.text} | confidence={item.confidence:.2f} | score={item.score:.2f} | source={item.collection}"
            )
        lines.append(f"</{name}>")
        return "\n".join(lines)

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate: ~1.3 tokens per whitespace-delimited word."""
        if not text:
            return 0
        return int(len(text.split()) * 1.3)
