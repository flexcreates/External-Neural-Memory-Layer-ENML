import json
from pathlib import Path
import unittest

from core.memory.extractors.rule_extractor import RuleExtractor
from core.retrieval import RetrievalPolicyEngine


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "replay_cases.json"


class ReplayCasesTest(unittest.TestCase):
    def test_replay_cases(self):
        data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        extractor = RuleExtractor()
        engine = RetrievalPolicyEngine()

        for case in data:
            if "input" in case:
                facts = extractor.extract(case["input"])
                self.assertTrue(facts, f"no facts for case {case['name']}")
                self.assertEqual(facts[0]["predicate"], case["expected_rule_predicate"])
                self.assertEqual(facts[0]["object"], case["expected_rule_object"])
            else:
                policy = engine.resolve(case["query"], case["collection"], case["profile"])
                self.assertEqual(policy.name, case["expected_policy"])
