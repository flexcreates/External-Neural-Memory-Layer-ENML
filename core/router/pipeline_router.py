from enum import Enum
from typing import Optional
from ..logger import get_logger
from ..config import ACTIVE_PIPELINE
from ..model_registry import get_active_model_record, ModelTier

logger = get_logger(__name__)

class PipelineMode(str, Enum):
    GENERAL = "GENERAL"
    CODER = "CODER"
    MID = "MID"

class PipelineRouter:
    @classmethod
    def classify(cls, user_input: str) -> PipelineMode:
        """Determines the appropriate pipeline mode for a given user input and active model."""
        if ACTIVE_PIPELINE.lower() != "auto":
            try:
                return PipelineMode(ACTIVE_PIPELINE.upper())
            except ValueError:
                logger.warning(f"[PipelineRouter] Invalid ACTIVE_PIPELINE '{ACTIVE_PIPELINE}', falling back to auto.")

        active_record = get_active_model_record()
        if not active_record:
            logger.debug("[PipelineRouter] Unknown active model. Defaulting to GENERAL.")
            return PipelineMode.GENERAL

        # Map ModelTier to PipelineMode directly
        if active_record.tier == ModelTier.CODER:
            return PipelineMode.CODER
        elif active_record.tier == ModelTier.MID:
            return PipelineMode.MID
        
        # We are on GENERAL tier
        # Check if the user is asking for code anyway. If so, warn, but don't silently reroute.
        # Create a temporary instance of CodingMemory just to use its heuristic parsing
        # without doing file I/O unless parse_task_from_input returns a valid task.
        # Wait, the heuristic in CodingMemory uses parse_task_from_input which actually CREATES a task.
        # We should decouple the check, but since we can't easily break the instructions,
        # we'll just check for coding keywords here.
        lowered = user_input.lower().strip()
        coding_keywords = [
            "implement", "fix the bug", "write a function", "refactor",
            "create a module", "build a", "create the", "code a"
        ]
        
        if any(kw in lowered for kw in coding_keywords):
            logger.warning("[PipelineRouter] User requested coding task, but a GENERAL tier model is active. Processing via GENERAL pipeline.")
            
        return PipelineMode.GENERAL

    @classmethod
    def get_pipeline_description(cls, mode: PipelineMode) -> str:
        descriptions = {
            PipelineMode.GENERAL: "General reasoning and memory recall pipeline",
            PipelineMode.CODER: "Coding assistant pipeline with persistent project/task tracking",
            PipelineMode.MID: "Fast low-latency mid-tier pipeline"
        }
        return descriptions.get(mode, "Unknown pipeline mode")
