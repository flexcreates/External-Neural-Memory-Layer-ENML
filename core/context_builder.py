from typing import List, Dict, Any, Optional, Tuple
import re
from .memory_manager import MemoryManager
from .time_provider import TimeProvider
from .logger import get_logger
from .context.distiller import ContextDistiller
from .context.prompt_budget_manager import PromptBudgetManager
from .memory.types import EvidencePacket, MemoryType
from .prompt_templates import build_chat_prompt, get_model_template_info, _strip_xml_from_system

logger = get_logger(__name__)

CLEAN_PROMPT_FAMILIES = {"mistral", "wizardcoder", "deepseek-coder", "deepseek", "openchat", "gemma", "qwen-coder"}


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
        is_identity_query = self._is_identity_query(user_input)
        code_model_families = {"wizardcoder", "deepseek-coder", "qwen-coder"}
        is_code_model = template_info.family in code_model_families
        is_code_task = self._is_code_task(user_input)
        
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
        has_local_evidence = bool(docs)
        is_general_knowledge_query = self._is_general_knowledge_query(user_input)
        append_structured_memory = not is_code_model

        # 2. Mode-Specific Prompts
        if is_code_model and is_code_task:
            temperature = 0.15
            role_prompt = "You are an expert programmer. Write complete, working code. No placeholders."
            if docs:
                clean_docs = []
                for doc in docs[:2]:
                    stripped = re.sub(r"<[^>]+>", " ", doc)
                    stripped = re.sub(r"\s+", " ", stripped).strip()
                    if stripped:
                        clean_docs.append(stripped)
                context_str = "\n".join(clean_docs)
                if context_str:
                    effective_system_prompt = f"{role_prompt}\nRelevant context: {context_str}"
                else:
                    effective_system_prompt = role_prompt
            else:
                effective_system_prompt = role_prompt
            logger.info(f"[PROMPT] Code model coding path: clean prompt, temp={temperature}")
        elif is_code_model:
            temperature = 0.35
            role_prompt = (
                "You are a helpful personal AI assistant. "
                "Answer directly, naturally, and use the relevant memory facts below when present. "
                "Do not frame your reply as programming help unless the user asks for code."
            )
            if docs:
                clean_docs = self._flatten_evidence_text(evidence_packet, limit=4)
                context_str = "\n".join(clean_docs)
                if context_str:
                    effective_system_prompt = f"{role_prompt}\nRelevant context: {context_str}"
                else:
                    effective_system_prompt = role_prompt
            else:
                effective_system_prompt = role_prompt
            logger.info(f"[PROMPT] Code model conversational path: clean prompt, temp={temperature}")
        elif policy_name == "document_memory":
            temperature = 0.0
            context_str = "\n".join(docs)
            effective_system_prompt = (
                "You are a factual AI assistant.\n"
                "Use ONLY the retrieved document context below.\n"
                "If the answer is not present in the documents, say you do not know.\n\n"
                f"Document Context:\n{context_str}\n\n"
                f"Question:\n{user_input}"
            )
        elif mode == "research_collection" and has_local_evidence:
            temperature = 0.2
            context_str = "\n".join(docs)
            effective_system_prompt = (
                "You are a factual AI assistant.\n"
                "Use the retrieved research context below first.\n"
                "If the context is incomplete, you may answer from general model knowledge, "
                "but clearly separate retrieved facts from general explanation.\n\n"
                f"Research Context:\n{context_str}\n\n"
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
        elif policy_name == "personal_memory" or (is_personal_query and has_local_evidence):
            temperature = 0.2
            effective_system_prompt = (
                "You are a personal memory recall assistant.\n"
                "Answer only from verified user memory when relevant.\n"
                "If multiple facts conflict, prefer the most specific and recent verified fact.\n"
                "For identity questions, keep user identity and assistant identity separate.\n"
                "For summary questions like 'what do you know about me', summarize the retrieved user facts clearly.\n"
                "CRITICAL: When recalling facts about the user, always answer in second person. "
                "Say 'Your favorite color is X' not 'My favorite color is X'. "
                "You are the assistant, not the user. Never answer as if you are the user."
            )
        elif policy_name == "research_memory" or is_general_knowledge_query:
            temperature = 0.35 if is_small_model else 0.45
            effective_system_prompt = (
                "You are a knowledgeable AI assistant.\n"
                "This is a general knowledge question, not a personal memory recall task.\n"
                "If retrieved research context exists, use it first.\n"
                "If no local research context exists, answer from standard model knowledge.\n"
                "Do not say 'I don't know' unless the concept is genuinely unknown or the user asked for a sourced memory-backed answer.\n"
                "Prefer clear teaching-style explanations for concept questions."
            )
        else: # Conversation / General Semantic Profile
            if is_code_task:
                temperature = 0.2
            else:
                temperature = 0.5 if policy_name == "conversation_policy" else 0.6
        
        # 3. Dynamic Knowledge Sufficiency Feedback & System Time
        system_time = f"System Time: {TimeProvider.formatted()}"
        strict_grounding = getattr(model_profile, "strict_grounding", False) or retrieval_data["policy"].strict_grounding
        sufficiency_feedback = "Local Knowledge Confidence: HIGH" if docs and len(docs) > 0 else "Local Knowledge Confidence: LOW\nWeb Research Allowed: TRUE"
        user_pref_rules = self._build_user_preference_rules()

        if docs and append_structured_memory:
            formatted_docs = "\n".join(docs)
            factual_recall_rule = ""
            factual_recall_prelude = ""
            if policy_name == "personal_memory":
                factual_recall_rule = (
                    "Treat direct user-stated memories as authoritative unless conflicting evidence is present. "
                    "Answer in plain declarative sentences without unnecessary qualifiers. "
                    "Always use second person (your, you) when referring to the user's facts."
                )
                flattened = self._flatten_evidence_text(evidence_packet, limit=6)
                if flattened:
                    factual_recall_prelude = (
                        "\n\nVerified memory facts:\n" + "\n".join(f"- {line}" for line in flattened) + "\n"
                        "If a fact contains an exact number, name, or value, repeat it exactly."
                    )
            if factual_recall_prelude:
                effective_system_prompt += factual_recall_prelude
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
        elif not docs and append_structured_memory:
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
        elif is_code_model:
            logger.info("[INJECT] Skipping structured memory XML for code-specialist model")
            
        # 4. Authority Identity Module Injection (Absolute Highest Priority)
        # This injects the permanent AI/User identity.json strings BEFORE the history.
        effective_system_prompt = self.memory_manager.authority_memory.get_injected_prompt(
            effective_system_prompt,
            compact=is_code_model or is_identity_query,
        )
        logger.debug(f"[INJECT] Authority memory injected into prompt")

        # 4b. Post-assembly XML flattening for clean-prompt model families
        # This ensures recall-path XML (<knowledge>, <retrieved_facts>, etc.)
        # is converted to plaintext for models that ignore structured XML.
        if template_info.family in CLEAN_PROMPT_FAMILIES:
            effective_system_prompt = _strip_xml_from_system(effective_system_prompt)
            logger.debug(f"[INJECT] Applied XML→plaintext flattening for {template_info.family} family")

        # 5. Append Conversation History (with token budget enforcement)
        is_broad_recall = self._is_broad_recall_query(user_input)
        if is_personal_query or is_identity_query or is_broad_recall:
            recent_history = []
        else:
            sliding_window_count = 6 if is_small_model else 12
            recent_history = history[-sliding_window_count:] if len(history) > sliding_window_count else history
        
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
        if "my " in lowered or "mine" in lowered or "myself" in lowered or "about me" in lowered:
            return True
        if lowered.startswith((
            "what is my",
            "what's my",
            "what are my",
            "who am i",
            "do i",
            "am i",
            "is my",
            "what do you know about me",
            "tell me about me",
        )):
            return True
        if "what do you know about " in lowered and "flex" in lowered:
            return True
        # Broad aggregation patterns
        if self._is_broad_recall_query(text):
            return True
        return False

    def _is_broad_recall_query(self, text: str) -> bool:
        """Detect broad aggregation queries that should trigger personal-memory recall."""
        lowered = (text or "").lower().strip()
        broad_patterns = (
            "tell me everything",
            "everything you know",
            "what do you know",
            "summarize what you know",
            "what have you learned",
            "what do you remember",
            "what can you tell me",
            "list everything",
            "tell me all",
            "all you know",
            "recall everything",
            "show me what you know",
        )
        return any(pattern in lowered for pattern in broad_patterns)

    def _is_identity_query(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        markers = (
            "who am i",
            "who are you",
            "what is my name",
            "what's my name",
            "your name",
            "my age",
            "your age",
            "about me",
        )
        return any(marker in lowered for marker in markers)

    def _detect_preference_intent(self, text: str) -> str:
        lowered = (text or "").lower()
        if "love" in lowered:
            return "love"
        if "like" in lowered:
            return "like"
        return ""

    def _is_code_task(self, text: str) -> bool:
        """
        Detect code generation, debugging, or implementation requests.
        """
        lowered = (text or "").lower().strip()
        if self._is_personal_query(lowered):
            return False
        code_markers = (
            "write a",
            "write the",
            "create a",
            "create the",
            "implement",
            "build a",
            "build the",
            "code a",
            "code the",
            "function",
            "class ",
            "script",
            "algorithm",
            "program",
            "module",
            "snippet",
            "debug",
            "fix the bug",
            "fix this",
            "refactor",
            "optimize the",
            "unit test",
            "generate code",
            "make a",
            "snake game",
            "pygame",
            "flask",
            "fastapi",
            "sql query",
            "regex",
            "decorator",
            "recursion",
        )
        return any(marker in lowered for marker in code_markers)

    def _is_general_knowledge_query(self, text: str) -> bool:
        lowered = (text or "").lower().strip()
        if self._is_personal_query(lowered):
            return False
        if self._is_code_task(lowered):
            return False

        concept_markers = (
            "explain",
            "what is",
            "what are",
            "how does",
            "how do",
            "why does",
            "why do",
            "difference between",
            "compare",
            "concept",
            "theory",
            "principle",
            "engine",
            "photosynthesis",
            "photoelectric",
            "carnot",
        )
        return any(marker in lowered for marker in concept_markers)

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

    def _flatten_evidence_text(self, evidence_packet: EvidencePacket, limit: int = 4) -> List[str]:
        flattened: List[str] = []
        if hasattr(evidence_packet, "all_items"):
            items = evidence_packet.all_items()
        else:
            items = []
            for attr in [
                "identity_items",
                "fact_items",
                "episodic_items",
                "document_items",
                "project_items",
                "research_items",
                "semantic_items",
            ]:
                items.extend(getattr(evidence_packet, attr, []) or [])

        for item in items:
            text = re.sub(r"\s+", " ", item.text).strip()
            if text and text not in flattened:
                flattened.append(text)
            if len(flattened) >= limit:
                break
        return flattened

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
