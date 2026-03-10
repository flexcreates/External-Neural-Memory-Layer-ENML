from core.config import (
    LLAMA_SERVER_URL,
    QDRANT_DOCUMENT_COLLECTION,
    QDRANT_KNOWLEDGE_COLLECTION,
    QDRANT_PROJECT_COLLECTION,
    QDRANT_RESEARCH_COLLECTION,
)
from core.logger import get_logger
from openai import OpenAI

logger = get_logger(__name__)


class QueryRouter:
    """Routes queries to the narrowest reasonable collection."""

    def __init__(self):
        self.client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key")

    def route(self, query: str) -> str:
        heuristic = self._heuristic_route(query)
        if heuristic:
            logger.info(f"[ROUTE] Heuristic intent route → {heuristic}")
            return heuristic

        try:
            response = self.client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Classify the query into exactly one tag: "
                            "personal_query, project_query, document_query, research_query. "
                            "Pure arithmetic, counting, continuation requests, and casual non-project tasks should default to personal_query fallback. "
                            "Reply with only the tag."
                        ),
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=10,
            )
            intent = response.choices[0].message.content.strip().lower()
            mapped = {
                "personal_query": QDRANT_KNOWLEDGE_COLLECTION,
                "project_query": QDRANT_PROJECT_COLLECTION,
                "document_query": QDRANT_DOCUMENT_COLLECTION,
                "research_query": QDRANT_RESEARCH_COLLECTION,
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

        if any(token in text for token in ["research", "explain", "what is", "how does", "compare", "framework"]):
            return QDRANT_RESEARCH_COLLECTION

        return None
