from core.config import (
    LLAMA_SERVER_URL,
    QDRANT_DOCUMENT_COLLECTION,
    QDRANT_KNOWLEDGE_COLLECTION,
    QDRANT_PROJECT_COLLECTION,
    QDRANT_RESEARCH_COLLECTION,
)
from core.logger import get_logger
from core.llm_runtime import detect_server_model
from core.prompt_templates import build_chat_prompt_from_messages, get_stop_sequences_for_model
from openai import OpenAI

logger = get_logger(__name__)


class QueryRouter:
    """Routes queries to the narrowest reasonable collection."""

    def __init__(self):
        self.client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key")
        self.model_name = detect_server_model(self.client)

    def route(self, query: str) -> str:
        heuristic = self._heuristic_route(query)
        if heuristic:
            logger.info(f"[ROUTE] Heuristic intent route → {heuristic}")
            return heuristic

        try:
            prompt = build_chat_prompt_from_messages(
                self.model_name,
                [
                    {
                        "role": "system",
                        "content": (
                            "Classify the query into exactly one tag: "
                            "personal_query, project_query, document_query, research_query, general_query. "
                            "Use personal_query only for questions about the user's identity, preferences, possessions, or past interactions. "
                            "Use general_query for ordinary world knowledge, explanations, science, math, definitions, or concept questions. "
                            "Reply with only the tag."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
            )
            response = self.client.completions.create(
                model=self.model_name,
                prompt=prompt,
                temperature=0.0,
                max_tokens=10,
                stop=get_stop_sequences_for_model(self.model_name),
            )
            intent = response.choices[0].text.strip().lower()
            mapped = {
                "personal_query": QDRANT_KNOWLEDGE_COLLECTION,
                "project_query": QDRANT_PROJECT_COLLECTION,
                "document_query": QDRANT_DOCUMENT_COLLECTION,
                "research_query": QDRANT_RESEARCH_COLLECTION,
                "general_query": QDRANT_RESEARCH_COLLECTION,
            }.get(intent)
            if mapped:
                logger.info(f"[ROUTE] LLM intent route → {intent} => {mapped}")
                return mapped
        except Exception as exc:
            logger.warning(f"[ROUTE] LLM classification failed: {exc}")

        return QDRANT_KNOWLEDGE_COLLECTION

    def _heuristic_route(self, query: str):
        text = query.lower().strip()

        general_task_patterns = [
            "count to", "multiply", "reverse", "continue rest", "continue the rest",
            "sum ", "subtract", "divide", "calculate", "math", "till 0",
        ]
        if any(pattern in text for pattern in general_task_patterns):
            return QDRANT_KNOWLEDGE_COLLECTION

        personal_markers = [
            "my ", "me ", "i ", "i'm", "i am", "who am i", "what is my", "remember",
            "do you know my", "about me",
        ]
        if any(marker in f" {text} " for marker in personal_markers):
            return QDRANT_KNOWLEDGE_COLLECTION

        if any(token in text for token in ["readme", "document", "docs", "file", "markdown", "notes", "pdf"]):
            return QDRANT_DOCUMENT_COLLECTION

        if any(token in text for token in ["project", "code", "repo", "repository", "function", "class", "module", "script", "architecture"]):
            return QDRANT_PROJECT_COLLECTION

        research_markers = [
            "research", "paper", "study", "citation", "source", "journal", "evidence",
            "explain", "what is", "what are", "how does", "how do", "compare", "framework",
            "theory", "principle", "concept",
        ]
        if any(token in text for token in research_markers):
            return QDRANT_RESEARCH_COLLECTION

        return None
