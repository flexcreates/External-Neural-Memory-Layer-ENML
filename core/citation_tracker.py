import json
import re
from pathlib import Path
from typing import Dict, List

from core.logger import BASE_DIR, get_logger
from core.memory.types import EvidenceItem, EvidencePacket

logger = get_logger(__name__)


class CitationTracker:
    """Tracks which retrieved evidence items were actually used in a response."""

    def __init__(self):
        self.path = BASE_DIR / "logs" / "citations.jsonl"

    def track(self, session_id: str, user_input: str, response_text: str, evidence_packet: EvidencePacket) -> List[Dict]:
        cited = []
        for item in evidence_packet.all_items():
            if self._is_cited(item, response_text):
                cited.append({
                    "memory_id": item.memory_id,
                    "memory_type": item.memory_type,
                    "score": item.score,
                    "confidence": item.confidence,
                    "collection": item.collection,
                    "text": item.text,
                })

        entry = {
            "session_id": session_id,
            "query": user_input,
            "response_preview": response_text[:400],
            "citations": cited,
        }
        self._append(entry)
        logger.info(f"[CITE] Tracked {len(cited)} cited evidence items for session {session_id}")
        return cited

    def _append(self, entry: Dict):
        try:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error(f"Failed to write citation log: {exc}")

    def _is_cited(self, item: EvidenceItem, response_text: str) -> bool:
        response = response_text.lower()
        text = item.text.lower()
        if not response or not text:
            return False
        if text in response:
            return True

        evidence_tokens = self._normalize_tokens(self._tokens(text))
        if not evidence_tokens:
            return False
        response_tokens = self._normalize_tokens(self._tokens(response))
        overlap = [token for token in evidence_tokens if token in response_tokens]
        if item.memory_type == "identity":
            return len(overlap) >= 1
        return len(overlap) >= min(2, len(evidence_tokens))

    def _tokens(self, text: str) -> List[str]:
        stopwords = {"user", "assistant", "score", "confidence", "source", "with", "from", "that", "this", "there"}
        return [
            token for token in re.findall(r"[a-zA-Z0-9_+-]+", text.lower())
            if len(token) >= 3 and token not in stopwords
        ]

    def _normalize_tokens(self, tokens: List[str]) -> List[str]:
        normalized = []
        for token in tokens:
            normalized.append(token)
            if token.endswith("s") and len(token) > 3:
                normalized.append(token[:-1])
        return normalized
