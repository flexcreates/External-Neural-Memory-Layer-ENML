import unittest

from core.memory.extractor import MemoryExtractor
from core.memory.extractors.rule_extractor import RuleExtractor
from core.memory_manager import MemoryManager
from core.memory.types import MemoryRecord, MemoryType
from core.retrieval import RetrievalPolicyEngine


class DummyAuthorityMemory:
    def __init__(self):
        self.data = {
            "user": {
                "name": "Flex",
                "age": None,
                "profession": "vibecoding",
                "preferences": {"favorite_color": "scion blue"},
            },
            "assistant": {"name": "Jarvis"},
        }

    def load(self):
        return self.data


class DummyRecordRepository:
    def __init__(self, records):
        self._records = records

    def all(self):
        return self._records


class TranscriptRegressionTest(unittest.TestCase):
    def test_math_task_does_not_route_to_project_policy(self):
        engine = RetrievalPolicyEngine()
        policy = engine.resolve(
            "count to 100 but in reverse also multiply each number by pi",
            "knowledge_collection",
            "small",
        )
        self.assertEqual(policy.name, "general_chat")

    def test_conversation_policy_for_non_memory_chat(self):
        engine = RetrievalPolicyEngine()
        policy = engine.resolve(
            "lets talk about god",
            "knowledge_collection",
            "small",
        )
        self.assertEqual(policy.name, "conversation_policy")

    def test_introvert_preferences_extract(self):
        extractor = RuleExtractor()
        facts = extractor.extract(
            "i would like you to remember that i am an introvert type person i dont go out too much i like my own personal space"
        )
        predicates = {fact["predicate"]: fact["object"] for fact in facts}
        self.assertEqual(predicates.get("personality_type"), "introvert")
        self.assertEqual(predicates.get("likes"), "personal space")

    def test_direct_profile_statements_are_extracted_despite_typo(self):
        extractor = MemoryExtractor()
        facts = extractor.extract_facts("and my proffesion is vibecoding")
        predicates = {fact["predicate"]: fact["object"] for fact in facts}
        self.assertEqual(predicates.get("has_profession"), "vibecoding")

    def test_local_profile_facts_are_retrievable_without_qdrant(self):
        manager = MemoryManager.__new__(MemoryManager)
        manager.authority_memory = DummyAuthorityMemory()
        manager.record_repository = DummyRecordRepository([
            MemoryRecord(
                memory_type=MemoryType.FACT.value,
                subject="user",
                predicate_canonical="has_profession",
                object_value="vibecoding",
                claim_text="user has_profession vibecoding.",
                tags=["profession", "vibecoding"],
            ),
            MemoryRecord(
                memory_type=MemoryType.PREFERENCE.value,
                subject="user",
                predicate_canonical="has_favourite_color",
                object_value="scion blue",
                claim_text="user has_favourite_color scion blue.",
                tags=["favorite", "color", "scion", "blue"],
            ),
        ])

        authority_items = manager._load_authority_identity_items("what is my profession ?")
        self.assertTrue(any("profession vibecoding" in item["text"] for item in authority_items))

        color_items = manager._load_local_record_items("what is my fav color ?")
        self.assertTrue(any("scion blue" in item["text"] for item in color_items))
