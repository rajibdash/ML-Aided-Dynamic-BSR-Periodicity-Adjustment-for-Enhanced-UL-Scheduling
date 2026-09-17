import unittest

from ml_bsr.models import MovingAveragePredictor
from ml_bsr.policy import AdaptivePeriodicityPolicy, FixedPeriodicityPolicy, select_bsr_periodicity


class PolicyTests(unittest.TestCase):
    def test_select_bsr_periodicity_uses_allowed_values(self) -> None:
        self.assertEqual(select_bsr_periodicity(18.0), 16.0)
        self.assertEqual(select_bsr_periodicity(0.5), 1.0)

    def test_adaptive_policy_bootstrap_then_predict(self) -> None:
        policy = AdaptivePeriodicityPolicy(predictor=MovingAveragePredictor(window=2), bootstrap_periodicity_ms=10.0)
        self.assertEqual(policy.next_periodicity([]), 10.0)
        self.assertEqual(policy.next_periodicity([12.0, 18.0]), 10.0)

    def test_fixed_policy_is_constant(self) -> None:
        policy = FixedPeriodicityPolicy(periodicity_ms=20.0)
        self.assertEqual(policy.next_periodicity([5.0, 10.0]), 20.0)


if __name__ == '__main__':
    unittest.main()
