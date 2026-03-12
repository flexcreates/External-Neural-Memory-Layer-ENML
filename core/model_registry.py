import os
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .logger import get_logger
from .config import (
    MODEL_BASE_PATH,
    MODEL_PATH_CODER,
    MODEL_PATH_GENERAL,
    MODEL_PATH_MID,
    MODEL_PATH_QWEN
)
from .llm_runtime import detect_server_model
from openai import OpenAI
from .config import LLAMA_SERVER_URL

logger = get_logger(__name__)

class ModelTier(str, Enum):
    GENERAL = "GENERAL"
    CODER = "CODER"
    MID = "MID"

class PromptFamily(str, Enum):
    LLAMA3 = "LLAMA3"
    MISTRAL = "MISTRAL"
    GEMMA = "GEMMA"
    CHATML = "CHATML"
    PHI3 = "PHI3"

class ModelCapability(str, Enum):
    CODE_GENERATION = "CODE_GENERATION"
    GENERAL_REASONING = "GENERAL_REASONING"
    FAST_INFERENCE = "FAST_INFERENCE"

@dataclass
class ModelRecord:
    name: str
    filename: str
    path: Path
    tier: ModelTier
    family: PromptFamily
    capabilities: List[ModelCapability]
    context_window_estimate: int
    default_temperature: float
    default_top_p: float
    description: str

# Skipped models
SKIPPED_MODELS = [
    "phi-2"  # safetensors format, incompatible with llama-server
]

def _build_registry() -> Dict[str, ModelRecord]:
    return {
        "deepseek-coder-6.7b-instruct.Q4_K_M.gguf": ModelRecord(
            name="DeepSeek Coder 6.7B",
            filename="deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
            path=MODEL_PATH_CODER / "deepseek" / "deepseek-coder-6.7b-instruct.Q4_K_M.gguf",
            tier=ModelTier.CODER,
            family=PromptFamily.CHATML,
            capabilities=[ModelCapability.CODE_GENERATION, ModelCapability.GENERAL_REASONING],
            context_window_estimate=4096,
            default_temperature=0.2,
            default_top_p=0.9,
            description="Strong coding model, CHATML format."
        ),
        "qwen2.5-coder-7b-instruct-q4_k_m.gguf": ModelRecord(
            name="Qwen2.5 Coder 7B",
            filename="qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            path=MODEL_PATH_CODER / "qwen" / "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
            tier=ModelTier.CODER,
            family=PromptFamily.CHATML,
            capabilities=[ModelCapability.CODE_GENERATION, ModelCapability.GENERAL_REASONING],
            context_window_estimate=4096,
            default_temperature=0.2,
            default_top_p=0.9,
            description="Highly capable 7B coding model, CHATML format."
        ),
        "wizardcoder-python-7b-v1.0.Q4_K_M.gguf": ModelRecord(
            name="WizardCoder Python 7B",
            filename="wizardcoder-python-7b-v1.0.Q4_K_M.gguf",
            path=MODEL_PATH_CODER / "wizard-coder" / "wizardcoder-python-7b-v1.0.Q4_K_M.gguf",
            tier=ModelTier.CODER,
            family=PromptFamily.CHATML, # Replaced legacy prompt with ChatML
            capabilities=[ModelCapability.CODE_GENERATION],
            context_window_estimate=4096,
            default_temperature=0.2,
            default_top_p=0.9,
            description="Specialized python generating model, converted to ChatML for context preservation."
        ),
        "gemma-2-9b-it-Q4_K_M.gguf": ModelRecord(
            name="Gemma 2 9B IT",
            filename="gemma-2-9b-it-Q4_K_M.gguf",
            path=MODEL_PATH_GENERAL / "google" / "gemma-2-9b-it-Q4_K_M.gguf",
            tier=ModelTier.GENERAL,
            family=PromptFamily.GEMMA,
            capabilities=[ModelCapability.GENERAL_REASONING],
            context_window_estimate=4096,
            default_temperature=0.7,
            default_top_p=0.9,
            description="Google's open-weights 9B instruct model."
        ),
        "llama-3-8b-instruct.Q4_K_M.gguf": ModelRecord(
            name="Llama 3 8B Instruct",
            filename="llama-3-8b-instruct.Q4_K_M.gguf",
            path=MODEL_PATH_GENERAL / "llama" / "llama-3-8b-instruct.Q4_K_M.gguf",
            tier=ModelTier.GENERAL,
            family=PromptFamily.LLAMA3,
            capabilities=[ModelCapability.GENERAL_REASONING],
            context_window_estimate=8192,
            default_temperature=0.7,
            default_top_p=0.9,
            description="Solid reasoning baseline LLaMA 3 8B."
        ),
        "mistral-7b-instruct-v0.2.Q4_K_M.gguf": ModelRecord(
            name="Mistral 7B Instruct v0.2",
            filename="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            path=MODEL_PATH_GENERAL / "mistral" / "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            tier=ModelTier.GENERAL,
            family=PromptFamily.MISTRAL,
            capabilities=[ModelCapability.GENERAL_REASONING],
            context_window_estimate=4096,
            default_temperature=0.7,
            default_top_p=0.9,
            description="Solid reasoning baseline from Mistral AI."
        ),
        "openchat-3.5-0106.Q4_K_M.gguf": ModelRecord(
            name="Openchat 3.5",
            filename="openchat-3.5-0106.Q4_K_M.gguf",
            path=MODEL_PATH_GENERAL / "openchat" / "openchat-3.5-0106.Q4_K_M.gguf",
            tier=ModelTier.GENERAL,
            family=PromptFamily.CHATML, # Treated as openchat/chatml variant
            capabilities=[ModelCapability.GENERAL_REASONING],
            context_window_estimate=4096,
            default_temperature=0.7,
            default_top_p=0.9,
            description="General 7b reasoning openchat model."
        ),
        "Qwen2.5-7B-Instust-Q4_K_M.gguf": ModelRecord(
            name="Qwen 2.5 7B Instruct",
            filename="Qwen2.5-7B-Instust-Q4_K_M.gguf",
            path=MODEL_PATH_QWEN / "Qwen2.5-7B-Instust-Q4_K_M.gguf",
            tier=ModelTier.GENERAL,
            family=PromptFamily.CHATML,
            capabilities=[ModelCapability.GENERAL_REASONING],
            context_window_estimate=4096,
            default_temperature=0.7,
            default_top_p=0.9,
            description="General Qwen language model."
        ),
        "Phi-3-mini-4k-instruct-Q4_K_M.gguf": ModelRecord(
            name="Phi-3 Mini 4k",
            filename="Phi-3-mini-4k-instruct-Q4_K_M.gguf",
            path=MODEL_PATH_MID / "phi-3-mini" / "Phi-3-mini-4k-instruct-Q4_K_M.gguf",
            tier=ModelTier.MID,
            family=PromptFamily.PHI3,
            capabilities=[ModelCapability.FAST_INFERENCE],
            context_window_estimate=4096,
            default_temperature=0.6,
            default_top_p=0.9,
            description="Small capability model suitable for fast inference."
        ),
        "qwen2.5-3b-instruct-q4_k_m.gguf": ModelRecord(
            name="Qwen 2.5 3B",
            filename="qwen2.5-3b-instruct-q4_k_m.gguf",
            path=MODEL_PATH_MID / "qwen" / "qwen2.5-3b-instruct-q4_k_m.gguf",
            tier=ModelTier.MID,
            family=PromptFamily.CHATML,
            capabilities=[ModelCapability.FAST_INFERENCE],
            context_window_estimate=4096,
            default_temperature=0.6,
            default_top_p=0.9,
            description="Fast Qwen inference model."
        ),
        "SmolLM3-3B-Q4_K_M.gguf": ModelRecord(
            name="SmolLM3 3B",
            filename="SmolLM3-3B-Q4_K_M.gguf",
            path=MODEL_PATH_MID / "smol" / "SmolLM3-3B-Q4_K_M.gguf",
            tier=ModelTier.MID,
            family=PromptFamily.CHATML,
            capabilities=[ModelCapability.FAST_INFERENCE],
            context_window_estimate=4096,
            default_temperature=0.6,
            default_top_p=0.9,
            description="Extremely small chatml variant."
        )
    }

MODEL_REGISTRY = _build_registry()

def get_model_record(filename: str) -> Optional[ModelRecord]:
    """Returns the ModelRecord for a specific model filename if it exists in the registry."""
    if filename in SKIPPED_MODELS:
        logger.warning(f"[Model Registry] Attempted to get skipped model: {filename}")
        return None
        
    for name, record in MODEL_REGISTRY.items():
        if record.filename.lower() == filename.lower() or record.name.lower() == filename.lower():
            return record
    return None

def get_models_by_tier(tier: ModelTier) -> List[ModelRecord]:
    """Returns all available models matching a specific tier."""
    return [r for r in MODEL_REGISTRY.values() if r.tier == tier]

def get_active_model_record() -> Optional[ModelRecord]:
    """Retrieves the record for the currently active server model directly from the runtime."""
    try:
        client = OpenAI(base_url=f"{LLAMA_SERVER_URL}/v1", api_key="sk-proj-no-key", timeout=1.0, max_retries=0)
        raw_id = detect_server_model(client)
        if not raw_id:
            return None
        candidates = [
            raw_id,
            raw_id + ".gguf",
            raw_id.lower(),
            raw_id.lower() + ".gguf",
            os.path.basename(raw_id),
            os.path.basename(raw_id) + ".gguf",
        ]
        for candidate in candidates:
            if candidate in MODEL_REGISTRY:
                return MODEL_REGISTRY[candidate]
            rec = get_model_record(candidate)
            if rec:
                return rec
    except Exception as e:
        logger.warning(f"[Model Registry] Failed to get active model from server: {e}")
    return None

def scan_model_paths() -> List[ModelRecord]:
    """Walks the DATA directory models and matches filenames to registry entries."""
    found_models = []
    
    if not MODEL_BASE_PATH.exists() or not MODEL_BASE_PATH.is_dir():
         logger.warning(f"[Model Registry] MODEL_BASE_PATH '{MODEL_BASE_PATH}' does not exist.")
         return found_models
         
    for root, _, files in os.walk(MODEL_BASE_PATH):
        for file in files:
            if not file.endswith(".gguf"):
                continue
                
            record = get_model_record(file)
            if record:
                found_models.append(record)
    
    return found_models
