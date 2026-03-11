"""
Hallucination Guard - Controls how the AI answers questions about itself.

Prevents the model from hallucinating about:
- Its training data
- Its creator/owner
- Its capabilities
- How it was built

Instead, it enforces answering only from the system context.
"""
import re
from typing import Optional, List, Tuple
from dataclasses import dataclass
from core.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HallucinationGuardResult:
    """Result of hallucination check."""
    is_system_query: bool
    guard_triggered: bool
    system_answer: Optional[str] = None


class HallucinationGuard:
    """Prevents hallucination on self-referential questions."""
    
    # Patterns that indicate the user is asking about the AI itself
    SELF_REFERENCE_PATTERNS = [
        # Training/origin questions
        r"how were you trained",
        r"what were you trained on",
        r"your training",
        r"your training data",
        r"who trained you",
        r"who created you",
        r"who made you",
        r"what (is|was) your training",
        r"where did you come from",
        r"what is your origin",
        r"your dataset",
        
        # Capabilities questions
        r"what (can|are) you capable of",
        r"your capabilities",
        r"your abilities",
        r"what (model|llm) (are|is) you",
        r"what model (are|is) you",
        
        # Identity questions that might lead to hallucination
        r"are you a (large language model|llm|chatbot)",
        r"are you an? (ai|人工智能)",
        r"are you (gpt|chatgpt|claude|gemini)",
        r"do you (have|contain) (web|internet) (access|scraping)",
        r"(web|internet) (scraping|access)",
        r"crowdsourcing",
    ]
    
    # System context facts that should be used instead of hallucinating
    SYSTEM_CONTEXT_FACTS = {
        "model": "Meta Llama 3 8B (or compatible local model)",
        "memory_system": "ENML (Enhanced Neural Memory Layer)",
        "creator": "Flex",
        "running": "locally",
        "architecture": "Qdrant-based hybrid RAG with semantic retrieval",
        "training": "This system was not trained on web scraping or crowdsourcing. It is a local AI assistant running on user hardware.",
    }
    
    def __init__(self):
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.SELF_REFERENCE_PATTERNS
        ]
    
    def check(self, user_input: str) -> HallucinationGuardResult:
        """
        Check if the user is asking a self-referential question.
        
        Args:
            user_input: The user's question
            
        Returns:
            HallucinationGuardResult with guard status and optional system answer
        """
        if not user_input or not isinstance(user_input, str):
            return HallucinationGuardResult(
                is_system_query=False,
                guard_triggered=False
            )
        
        lowered = user_input.lower()
        
        # Check if it's a self-reference query
        is_self_reference = any(
            pattern.search(lowered) for pattern in self._compiled_patterns
        )
        
        if not is_self_reference:
            return HallucinationGuardResult(
                is_system_query=False,
                guard_triggered=False
            )
        
        # Generate system answer based on what's being asked
        system_answer = self._generate_system_answer(lowered)
        
        return HallucinationGuardResult(
            is_system_query=True,
            guard_triggered=True,
            system_answer=system_answer
        )
    
    def _generate_system_answer(self, query: str) -> str:
        """Generate a factual answer based on the system context."""
        answers = []
        
        # Training/origin
        if any(p in query for p in ["train", "origin", "creator", "made", "dataset"]):
            answers.append(
                "I am running locally using Meta Llama 3 8B with ENML memory. "
                "I was not trained on web scraping or crowdsourcing. "
                "Flex is my creator."
            )
        
        # Model
        if "model" in query:
            answers.append(f"I run on {self.SYSTEM_CONTEXT_FACTS['model']}.")
        
        # Memory system
        if "memory" in query:
            answers.append(
                f"I use {self.SYSTEM_CONTEXT_FACTS['memory_system']} for persistent memory."
            )
        
        # Capabilities
        if "capabil" in query or "able" in query:
            answers.append(
                "I can answer questions, help with tasks, and maintain persistent memory "
                "of our conversations. I run locally on your machine."
            )
        
        # Default answer
        if not answers:
            answers.append(
                "I am a local AI assistant built by Flex using ENML memory architecture. "
                "I run on a local LLM model and do not have internet access or web scraping capabilities."
            )
        
        return " ".join(answers)
    
    def inject_guard_into_prompt(
        self, 
        system_prompt: str, 
        user_input: str,
        model_name: str = ""
    ) -> Tuple[str, bool]:
        """
        Inject hallucination guard into the system prompt if needed.
        
        Args:
            system_prompt: Current system prompt
            user_input: User's question
            model_name: Name of the model being used
            
        Returns:
            Tuple of (modified_prompt, guard_injected)
        """
        check_result = self.check(user_input)
        
        if not check_result.guard_triggered:
            return system_prompt, False
        
        # Add guard instruction to system prompt
        guard_instruction = (
            "\n\n<SELF_REFERENCE_GUARD>\n"
            f"IMPORTANT: The user is asking about you (the AI). "
            f"Answer using ONLY this system context: {check_result.system_answer}. "
            "Do NOT mention training data, web scraping, or crowdsourcing. "
            "Do NOT speculate about your training. "
            "Base your answer ONLY on what is stated in this system context.\n"
            "</SELF_REFERENCE_GUARD>\n"
        )
        
        return system_prompt + guard_instruction, True


# Global guard instance
_guard: Optional[HallucinationGuard] = None


def get_hallucination_guard() -> HallucinationGuard:
    """Get or create the global hallucination guard instance."""
    global _guard
    if _guard is None:
        _guard = HallucinationGuard()
    return _guard
