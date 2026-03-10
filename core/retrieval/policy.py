from dataclasses import dataclass, field
from typing import Dict, List

from core.config import (
    QDRANT_DOCUMENT_COLLECTION,
    QDRANT_EPISODIC_COLLECTION,
    QDRANT_KNOWLEDGE_COLLECTION,
    QDRANT_PROJECT_COLLECTION,
    QDRANT_RESEARCH_COLLECTION,
)


@dataclass
class RetrievalPolicy:
    name: str
    primary_collections: List[str]
    secondary_collections: List[str] = field(default_factory=list)
    limits: Dict[str, int] = field(default_factory=dict)
    answer_policy: List[str] = field(default_factory=list)
    strict_grounding: bool = False


class RetrievalPolicyEngine:
    def resolve(self, query: str, routed_collection: str, model_profile_name: str) -> RetrievalPolicy:
        text = query.lower()
        is_personal = any(token in f" {text} " for token in [" my ", " me ", " i ", " i'm ", " i am "]) or text.startswith("what is my")
        is_general_task = any(token in text for token in ["count to", "multiply", "reverse", "write a poem", "tell a joke", "continue rest", "continue the rest"])
        is_conversation = any(token in text for token in [
            "let's talk", "lets talk", "talk about", "what do you understand", "tell me about",
            "i am from", "i'm from", "stop asking", "just talk to me", "don't ask", "dont ask",
            "can just stop asking", "i am frustrated", "i'm frustrated"
        ])

        if is_general_task:
            return RetrievalPolicy(
                name="general_chat",
                primary_collections=[],
                limits={},
                answer_policy=[
                    "This is a general task, not a memory recall request.",
                    "Do not force memory retrieval when the task is purely computational or conversational.",
                ],
                strict_grounding=False,
            )

        if is_conversation and not text.startswith("what is my") and not text.startswith("what animal do i") and "my " not in text:
            return RetrievalPolicy(
                name="conversation_policy",
                primary_collections=[],
                secondary_collections=[],
                limits={},
                answer_policy=[
                    "This is ordinary conversation, not a memory recall task.",
                    "Respond naturally and directly.",
                    "Avoid repetitive filler closers like 'How can I assist you?' or 'Would you like to discuss...'.",
                    "Ask follow-up questions only when they add clear value to the conversation.",
                ],
                strict_grounding=False,
            )

        if is_personal or routed_collection == QDRANT_KNOWLEDGE_COLLECTION:
            return RetrievalPolicy(
                name="personal_memory",
                primary_collections=[QDRANT_KNOWLEDGE_COLLECTION],
                secondary_collections=[QDRANT_EPISODIC_COLLECTION],
                limits={"identity": 3, "fact": 4, "semantic_claim": 2, "episodic": 2},
                answer_policy=[
                    "For personal memory questions, answer only from retrieved memory or say unknown.",
                    "Use a direct factual tone.",
                    "Keep the answer short, usually one or two sentences.",
                    "Do not hedge when the memory is a direct user-stated fact.",
                    "If conflicting memories exist, state the conflict instead of guessing.",
                    "Avoid unnecessary follow-up questions after a direct factual answer.",
                ],
                strict_grounding=True,
            )

        if routed_collection == QDRANT_PROJECT_COLLECTION:
            return RetrievalPolicy(
                name="project_memory",
                primary_collections=[QDRANT_PROJECT_COLLECTION],
                limits={"project": 5},
                answer_policy=[
                    "Prioritize project evidence and architecture notes.",
                    "Do not mix personal memory unless the query explicitly asks for it.",
                ],
            )

        if routed_collection == QDRANT_DOCUMENT_COLLECTION:
            return RetrievalPolicy(
                name="document_memory",
                primary_collections=[QDRANT_DOCUMENT_COLLECTION],
                limits={"document": 5},
                answer_policy=[
                    "Answer from retrieved document evidence when available.",
                    "If the answer is absent from retrieved documents, say so.",
                ],
                strict_grounding=True,
            )

        return RetrievalPolicy(
            name="research_memory",
            primary_collections=[QDRANT_RESEARCH_COLLECTION],
            limits={"research": 5},
            answer_policy=[
                "Use retrieved research context when present.",
                "If memory is insufficient, rely on model knowledge but avoid unsupported specifics.",
            ],
        )
