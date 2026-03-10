from openai import OpenAI

from .config import DEFAULT_CHAT_MODEL
from .logger import get_logger

logger = get_logger(__name__)


def detect_server_model(client: OpenAI, fallback: str = DEFAULT_CHAT_MODEL) -> str:
    try:
        models = client.models.list()
        data = getattr(models, "data", None) or []
        if data:
            model_id = getattr(data[0], "id", None)
            if model_id:
                return model_id
    except Exception as exc:
        logger.debug(f"[LLM] Failed to detect server model: {exc}")
    return fallback
