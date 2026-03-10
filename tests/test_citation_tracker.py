import tempfile
import unittest

from core.citation_tracker import CitationTracker
from core.memory.types import EvidenceItem, EvidencePacket


class CitationTrackerTest(unittest.TestCase):
    def test_citation_tracker_marks_used_evidence(self):
        tracker = CitationTracker()
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            tracker.path = Path(tmpdir) / "citations.jsonl"

            packet = EvidencePacket(
                policy_name="personal_memory",
                answer_policy=[],
                fact_items=[
                    EvidenceItem(
                        memory_id="fact_1",
                        memory_type="fact",
                        text="user likes turtles",
                        score=0.91,
                        confidence=0.92,
                        collection="knowledge_collection",
                    )
                ],
            )

            cited = tracker.track(
                session_id="s1",
                user_input="what do i like?",
                response_text="You like turtles.",
                evidence_packet=packet,
            )

            self.assertEqual(len(cited), 1)
            self.assertEqual(cited[0]["memory_id"], "fact_1")
            self.assertTrue(tracker.path.exists())
