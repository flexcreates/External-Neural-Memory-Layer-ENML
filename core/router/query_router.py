from typing import Dict, Any
from core.config import (
    QDRANT_RESEARCH_COLLECTION,
    QDRANT_PROJECT_COLLECTION,
    QDRANT_CONVERSATION_COLLECTION,
    QDRANT_KNOWLEDGE_COLLECTION
)
from core.logger import get_logger

logger = get_logger(__name__)

class QueryRouter:
    """Routes queries to correct Qdrant collection or Identity memory."""
    def __init__(self):
        pass

    def route(self, query: str) -> str:
        """
        Determines the target collection or domain based on query analysis.
        Priority order: identity > document/content > project > research > conversation
        """
        query_lower = query.lower()
        
        # ── Priority 1: Identity / personal facts ──
        # Highest priority to prevent 'what is my' being hijacked by 'what is' research
        identity_keywords = [
            "who am i", "my name", "my pc", "my specs", "my father", 
            "my mother", "my condition", "what do i like", "do i have", 
            "my hobbies", "my friend", "my best friend", "what is my",
            "about me", "my pet", "my device", "my laptop", "my computer",
        ]
        for kw in identity_keywords:
            if kw in query_lower:
                logger.info(f"[ROUTE] Matched identity keyword '{kw}' → {QDRANT_KNOWLEDGE_COLLECTION}")
                return QDRANT_KNOWLEDGE_COLLECTION
        
        # ── Priority 2: Document / content queries ──
        # Queries ABOUT ingested documents should search knowledge (also triggers hybrid chunk search)
        content_keywords = [
            "folder structure", "project structure", "file structure",
            "how does it work", "how it works", "workflow", "features",
            "about the project", "about enml", "about this project",
            "tell me about", "facts about", "what does it do",
            "components", "overview", "summary", "describe",
        ]
        for kw in content_keywords:
            if kw in query_lower:
                logger.info(f"[ROUTE] Matched content keyword '{kw}' → {QDRANT_KNOWLEDGE_COLLECTION}")
                return QDRANT_KNOWLEDGE_COLLECTION
            
        # ── Priority 3: Project/code routing ──
        project_keywords = [
            "codebase", "implemented", "script", ".py", "function", "class", "module", 
            "source code", "git", "project", "app", "application", "repo", "repository", 
            "system", "architecture"
        ]
        for kw in project_keywords:
            if kw in query_lower:
                logger.info(f"[ROUTE] Matched project keyword '{kw}' → {QDRANT_PROJECT_COLLECTION}")
                return QDRANT_PROJECT_COLLECTION
            
        # ── Priority 4: Research routing ──
        # More specific patterns to avoid hijacking document queries
        research_keywords = [
            "explain the concept", "how does a", "what is a", "what is the",
            "documentation for", "article about", "theory of", "paper on",
            "research on", "study about", "research", "paper", "study", 
            "framework", "concept", "article"
        ]
        for kw in research_keywords:
            if kw in query_lower:
                logger.info(f"[ROUTE] Matched research keyword '{kw}' → {QDRANT_RESEARCH_COLLECTION}")
                return QDRANT_RESEARCH_COLLECTION
            
        # Fallback to knowledge (instead of conversation) for better hybrid retrieval
        logger.info(f"[ROUTE] No keyword match, fallback → {QDRANT_KNOWLEDGE_COLLECTION}")
        return QDRANT_KNOWLEDGE_COLLECTION
