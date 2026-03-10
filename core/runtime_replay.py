import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from core.logger import BASE_DIR, get_logger

logger = get_logger(__name__)


class RuntimeReplayLogger:
    """Writes end-to-end runtime traces for offline replay analysis."""

    def __init__(self):
        self.path = BASE_DIR / "logs" / "runtime_replay.jsonl"

    def log(self, entry: Dict[str, Any]):
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            **entry,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error(f"Failed to write runtime replay log: {exc}")
