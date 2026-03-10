from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

from core.memory.lifecycle_service import MemoryLifecycleService
from core.memory.record_repository import MemoryRecordRepository
from core.memory.types import MemoryRecord, MemoryStatus, MemoryType


class FakeFeedback:
    def __init__(self, scores):
        self.scores = scores

    def get_memory_quality_score(self, fact_id: str) -> float:
        return self.scores.get(fact_id, 0.5)


class MemoryLifecycleTest(unittest.TestCase):
    def test_lifecycle_archives_and_prunes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = MemoryRecordRepository()
            repo.path = Path(tmpdir) / "memory_records.json"
            repo._save([])

            old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat().replace("+00:00", "Z")
            records = [
                MemoryRecord(
                    memory_id="semantic_old",
                    memory_type=MemoryType.SEMANTIC_CLAIM.value,
                    subject="user",
                    claim_text="I enjoy robotics",
                    created_at=old,
                    reinforcement_count=1,
                    confidence=0.4,
                    status=MemoryStatus.PROVISIONAL.value,
                ),
                MemoryRecord(
                    memory_id="archived_old",
                    memory_type=MemoryType.EPISODIC.value,
                    subject="conversation",
                    claim_text="Old summary",
                    created_at=old,
                    reinforcement_count=1,
                    confidence=0.3,
                    status=MemoryStatus.ARCHIVED.value,
                ),
            ]
            repo._save([record.to_dict() for record in records])

            service = MemoryLifecycleService(repo, FakeFeedback({"semantic_old": 0.2, "archived_old": 0.1}))
            result = service.run_once()

            saved = repo.all()
            self.assertGreaterEqual(result["archived"], 1)
            self.assertGreaterEqual(result["pruned"], 1)
            self.assertTrue(any(record.memory_id == "semantic_old" and record.status == MemoryStatus.ARCHIVED.value for record in saved))
            self.assertTrue(all(record.memory_id != "archived_old" for record in saved))
