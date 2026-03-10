from core.config import CODING_CHAT_MODEL, DEFAULT_CHAT_MODEL, FAST_CHAT_MODEL, REASONING_CHAT_MODEL
from core.router.model_profiles import MEDIUM_MODEL_PROFILE, SMALL_MODEL_PROFILE


class ModelRouter:
    """Small heuristic router for chat model selection."""

    def route(self, user_input: str) -> str:
        text = user_input.lower()
        if any(token in text for token in ["code", "python", "bug", "stack trace", "refactor", "function", "class"]):
            return CODING_CHAT_MODEL
        if len(text.split()) <= 6:
            return FAST_CHAT_MODEL
        if any(token in text for token in ["why", "design", "architecture", "tradeoff", "reason"]):
            return REASONING_CHAT_MODEL
        return DEFAULT_CHAT_MODEL

    def get_profile(self, model_name: str):
        if model_name in {FAST_CHAT_MODEL}:
            return SMALL_MODEL_PROFILE
        if model_name in {CODING_CHAT_MODEL, REASONING_CHAT_MODEL, DEFAULT_CHAT_MODEL}:
            return MEDIUM_MODEL_PROFILE
        return MEDIUM_MODEL_PROFILE
