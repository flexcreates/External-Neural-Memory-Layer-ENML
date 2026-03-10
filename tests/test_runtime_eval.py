import unittest

from tools.eval_runtime import evaluate


class RuntimeEvalTest(unittest.TestCase):
    def test_evaluate_runtime_metrics(self):
        entries = [
            {
                "evidence_count": 3,
                "citations": [{"memory_id": "1"}],
                "strict_grounding": True,
                "policy_name": "personal_memory",
                "timings_ms": {"total": 100.0},
                "unsupported_claim_estimate": 1,
            },
            {
                "evidence_count": 0,
                "citations": [],
                "strict_grounding": False,
                "policy_name": "research_memory",
                "timings_ms": {"total": 50.0},
                "unsupported_claim_estimate": 0,
            },
        ]
        result = evaluate(entries)
        self.assertEqual(result["total_requests"], 2)
        self.assertEqual(result["retrieval_hit_rate"], 0.5)
        self.assertEqual(result["strict_grounded_response_rate"], 0.5)
