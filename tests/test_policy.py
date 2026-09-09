import unittest

import numpy as np

from frauddetectionreport.costs import Action, DEFAULT_PARAMS, expected_cost
from frauddetectionreport.evaluate import operational_metrics, fairness_check
from frauddetectionreport.policy import FullPolicy, WithReview


class PolicyTests(unittest.TestCase):
    def test_customer_impact_keeps_missing_and_small_groups(self):
        action = np.array([Action.BLOCK, Action.AUTH, Action.AUTH, Action.APPROVE])
        report = fairness_check(action, np.array([0, 0, 1, 1]),
                                np.array([None, None, "x", "x"]), DEFAULT_PARAMS)
        self.assertEqual(report.n.sum(), 4)
        missing = report.set_index("segment").loc["(missing)"]
        self.assertEqual(missing.n_legit, 2)
        self.assertAlmostEqual(missing.false_block_rate, 0.5)
        self.assertAlmostEqual(missing.expected_auth_failure_rate, 0.05)
        self.assertAlmostEqual(missing.expected_legit_denial_rate, 0.55)
        self.assertTrue(missing.small_sample)
        self.assertTrue(np.isnan(report.set_index("segment").loc["x", "expected_legit_denial_rate"]))

    def test_full_policy_is_pointwise_optimal_without_budget(self):
        params = DEFAULT_PARAMS.with_(review_budget=None)
        prob = np.linspace(0, 1, 10_001)
        amount = np.full_like(prob, 100.0)

        action = FullPolicy().decide(prob, amount, params)
        chosen = expected_cost(action, prob, amount, params)
        best = np.stack(
            [expected_cost(a, prob, amount, params) for a in Action]
        ).min(axis=0)

        np.testing.assert_allclose(chosen, best)

    def test_review_budget_is_applied_per_group(self):
        params = DEFAULT_PARAMS.with_(review_budget=1)
        prob = np.full(6, 0.2)
        amount = np.full(6, 100.0)
        day = np.array([1, 1, 1, 2, 2, 2])

        action = WithReview().decide(prob, amount, params, review_group=day)

        self.assertEqual(int((action == Action.REVIEW).sum()), 2)
        self.assertEqual(int((action[day == 1] == Action.REVIEW).sum()), 1)
        self.assertEqual(int((action[day == 2] == Action.REVIEW).sum()), 1)

    def test_operational_recall_uses_action_effectiveness(self):
        params = DEFAULT_PARAMS
        action = np.array([Action.AUTH, Action.REVIEW])
        is_fraud = np.array([1, 1])
        amount = np.array([100.0, 100.0])

        report = operational_metrics(action, is_fraud, amount, params)
        expected = ((1 - params.auth_pass_fraud) + params.review_catch_rate) / 2

        self.assertAlmostEqual(report["fraud_recall_count"], expected)
        self.assertAlmostEqual(report["fraud_recall_amount"], expected)


if __name__ == "__main__":
    unittest.main()
