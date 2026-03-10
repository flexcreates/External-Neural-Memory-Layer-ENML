from typing import Dict, Any
from core.config import (
    QDRANT_RESEARCH_COLLECTION,
    QDRANT_PROJECT_COLLECTION,
    QDRANT_CONVERSATION_COLLECTION,
    QDRANT_KNOWLEDGE_COLLECTION,
    QDRANT_DOCUMENT_COLLECTION,
    LLAMA_SERVER_URL
)
from core.logger import get_logger
from openai import OpenAI

logger = get_logger(__name__)

class QueryRouter:
    """Routes queries to correct Qdrant collection based on LLM intent classification."""
    def __init__(self):
        self.client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key")

    def route(self, query: str) -> str:
        """
        Classify the query into one of 5 intents and return the corresponding collection.
        If classification fails, fallback to heuristics.
        """
        system_prompt = (
            "You are a sophisticated query intent classifier. Classify the user's query into EXACTLY ONE of the following tags:\n"
            "- 'personal_question': The user is asking about themselves, their identity, specs, hobbies, or the AI's identity.\n"
            "- 'project_question': The user is asking about a codebase, scripts, programming, architecture, or system.\n"
            "- 'document_question': The user is asking about specific documents, notes, readmes, or text files.\n"
            "- 'research_question': The user is asking about general concepts, academic topics, frameworks, or web knowledge.\n"
            "- 'general_knowledge': The user is asking a basic conversation question or something not related to personal memory, projects, or documents.\n\n"
            "Respond ONLY with the exact string tag and nothing else."
        )
        
        try:
            response = self.client.chat.completions.create(
                model="Meta-Llama-3-8B-Instruct",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query:\n{query}"}
                ],
                temperature=0.0,
                max_tokens=15
            )
            intent = response.choices[0].message.content.strip().lower()
            logger.info(f"[ROUTE] LLM classified intent: {intent}")
            
            if "personal_question" in intent:
                return QDRANT_KNOWLEDGE_COLLECTION
            elif "project_question" in intent:
                return QDRANT_PROJECT_COLLECTION
            elif "document_question" in intent:
                return QDRANT_DOCUMENT_COLLECTION
            elif "research_question" in intent:
                return QDRANT_RESEARCH_COLLECTION
            elif "general_knowledge" in intent:
                return QDRANT_KNOWLEDGE_COLLECTION # fallback
                
        except Exception as e:
            logger.warning(f"[ROUTE] LLM classification failed: {e}. Falling back to default.")

        query_lower = query.lower()
        project_keywords = ["code", "script", "project", "app"]
        if any(kw in query_lower for kw in project_keywords):
            return QDRANT_PROJECT_COLLECTION
            
        research_keywords = ["explain", "what is a", "how does"]
        if any(kw in query_lower for kw in research_keywords):
            return QDRANT_RESEARCH_COLLECTION

        return QDRANT_KNOWLEDGE_COLLECTION
