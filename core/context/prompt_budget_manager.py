from typing import Dict, List
from core.config import (
    PROMPT_BUDGET_DOCUMENTS,
    PROMPT_BUDGET_MEMORY,
    PROMPT_BUDGET_SYSTEM,
    PROMPT_BUDGET_USER,
)


class PromptBudgetManager:
    """Keeps prompt sections within a predictable token budget."""

    def __init__(self, max_context_tokens: int):
        self.max_context_tokens = max_context_tokens

    def allocate(self) -> Dict[str, int]:
        configured_total = PROMPT_BUDGET_SYSTEM + PROMPT_BUDGET_MEMORY + PROMPT_BUDGET_DOCUMENTS + PROMPT_BUDGET_USER
        if configured_total <= self.max_context_tokens:
            buffer = max(0, self.max_context_tokens - configured_total)
            return {
                "system": PROMPT_BUDGET_SYSTEM,
                "memory": PROMPT_BUDGET_MEMORY,
                "documents": PROMPT_BUDGET_DOCUMENTS,
                "user": PROMPT_BUDGET_USER,
                "buffer": buffer,
            }
        buffer_target = int(self.max_context_tokens * 0.05)
        available = max(0, self.max_context_tokens - buffer_target)
        scale = available / max(configured_total, 1)
        system = int(PROMPT_BUDGET_SYSTEM * scale)
        memory = int(PROMPT_BUDGET_MEMORY * scale)
        documents = int(PROMPT_BUDGET_DOCUMENTS * scale)
        user = int(PROMPT_BUDGET_USER * scale)
        buffer = max(0, self.max_context_tokens - (system + memory + documents + user))
        return {
            "system": system,
            "memory": memory,
            "documents": documents,
            "user": user,
            "buffer": buffer,
        }

    def trim_items(self, items: List[str], estimate_tokens) -> List[str]:
        budgets = self.allocate()
        limit = budgets["memory"] + budgets["documents"]
        kept: List[str] = []
        used = 0
        for item in items:
            item_tokens = estimate_tokens(item)
            if used + item_tokens > limit:
                break
            kept.append(item)
            used += item_tokens
        return kept
