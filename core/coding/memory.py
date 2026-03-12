import re
from typing import Optional, Dict, Any
from .models import CodingTask, TaskStatus
from .task_store import CodingTaskStore
from .context_builder import CodeContextBuilder
from .vector_store import CodeVectorStore
from ..logger import get_logger
from pathlib import Path
from ..config import ENML_ROOT

logger = get_logger(__name__)

class CodingMemory:
    def __init__(self, storage_path: Optional[str] = None):
        if storage_path is None:
            # Default to ENML memory root / coding
            storage_path = str(ENML_ROOT / "memory" / "coding")
            
        self.task_store = CodingTaskStore(storage_path)
        self.context_builder = CodeContextBuilder()
        self.vector_store = CodeVectorStore()
        
        try:
            self.vector_store.ensure_collections()
        except Exception as e:
            logger.warning(f"[CodingMemory] Could not ensure Qdrant collections during init: {e}")

    def get_prompt_injection(self, user_input: str, project_path: Optional[str] = None) -> str:
        """Entry point for the pipeline to get the structured coding context."""
        try:
            active_task = self.task_store.get_active_task()
            return self.context_builder.build_full_coding_context(active_task, project_path)
        except Exception as e:
            logger.warning(f"[CodingMemory] Failed to build prompt injection: {e}")
            return ""

    def create_task(self, title: str, description: str = "") -> CodingTask:
        task = CodingTask(title=title, description=description, status=TaskStatus.IN_PROGRESS)
        self.task_store.save(task)
        self.vector_store.index_task(task)
        logger.info(f"[CodingMemory] Created new task: {title}")
        return task

    def get_active_task(self) -> Optional[CodingTask]:
        return self.task_store.get_active_task()

    def advance_current_task(self) -> bool:
        task = self.get_active_task()
        if not task:
            return False
        return self.task_store.advance_step(task.task_id)

    def complete_current_task(self) -> bool:
        task = self.get_active_task()
        if not task:
            return False
        return self.task_store.update_status(task.task_id, TaskStatus.DONE)

    def parse_task_from_input(self, user_input: str, record: bool = True) -> Optional[CodingTask]:
        import logging
        logging.getLogger("enml.coding").debug(f"[CODING-PARSE] Checking input: {user_input[:80]}")
        
        # 1. Quick heuristic to decide if we need the parser LLM
        # Added broad range of heuristic hooks bypassing LLM latency
        if not user_input:
            return None
            
        lowered = user_input.lower().strip()
        
        if lowered.endswith("?"):
            return None
            
        if len(lowered.split()) < 4:
            return None
            
        greetings = {"hello", "hi", "thanks", "okay", "yes", "no"}
        if lowered in greetings:
            return None
            
        coding_keywords = [
            "implement", "fix the bug", "write a function", "refactor",
            "create a module", "build a", "create the", "code a",
            "add a", "add support for", "build the", "create a",
            "update the", "update my", "modify the", "extend the",
            "debug", "fix the", "fix my", "resolve the", "patch",
            "write a", "write the", "write me a",
            "traceback", "error:", "exception:"
        ]
        
        # Check for basic hit
        if not any(kw in lowered for kw in coding_keywords):
            return None
            
        # Try to roughly extract a title (first sentence or up to 60 chars)
        title = user_input.split(".")[0].split("\n")[0].strip()
        if len(title) > 60:
            title = title[:57] + "..."
            
        try:
            task = CodingTask(
                title=title,
                description=user_input,
                status=TaskStatus.IN_PROGRESS
            )
            if record:
                self.task_store.save(task)
                self.vector_store.index_task(task)
                logger.info(f"[CodingMemory] Auto-parsed and recorded coding task: '{title}'")
            return task
        except Exception as e:
            logger.warning(f"[CodingMemory] Failed to auto-parse task: {e}")
            return None
