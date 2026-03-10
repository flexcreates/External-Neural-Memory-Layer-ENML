import unittest

from core.memory.extractors.rule_extractor import RuleExtractor
from core.retrieval import RetrievalPolicyEngine


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
