import unittest

from core.context.prompt_budget_manager import PromptBudgetManager


class PromptBudgetManagerTest(unittest.TestCase):
    def test_prompt_budget_manager_respects_total_context(self):
        manager = PromptBudgetManager(3000)
        budgets = manager.allocate()
        self.assertLessEqual(sum(budgets.values()), 3000)
        self.assertGreater(budgets["system"], 0)
        self.assertGreater(budgets["memory"], 0)

    def test_prompt_budget_manager_trims_items(self):
        manager = PromptBudgetManager(100)
        kept = manager.trim_items(["a " * 20, "b " * 200], lambda text: len(text.split()))
        self.assertEqual(len(kept), 1)
