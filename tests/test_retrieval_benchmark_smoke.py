import unittest

from core.retrieval import RetrievalPolicyEngine


class RetrievalPolicyEngineTest(unittest.TestCase):
    def test_retrieval_policy_engine_smoke(self):
        engine = RetrievalPolicyEngine()
        policy = engine.resolve("what is my age", "knowledge_collection", "small")
        self.assertEqual(policy.name, "personal_memory")
        self.assertTrue(policy.strict_grounding)
