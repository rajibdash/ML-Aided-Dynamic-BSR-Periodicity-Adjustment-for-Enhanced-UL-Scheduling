import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_bsr.models import ModelDependencyError, MovingAveragePredictor, OptionalBackendPredictor
from ml_bsr.policy import AdaptivePeriodicityPolicy, FixedPeriodicityPolicy, select_bsr_periodicity


class PolicyTests(unittest.TestCase):
    def test_select_bsr_periodicity_uses_allowed_values(self) -> None:
        self.assertEqual(select_bsr_periodicity(18.0), 16.0)
        self.assertEqual(select_bsr_periodicity(0.5), 1.0)
        self.assertEqual(select_bsr_periodicity(float('inf')), float('inf'))
        self.assertEqual(select_bsr_periodicity(18.0, guard_factor=0.5), 5.0)

    def test_invalid_guard_factor_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            select_bsr_periodicity(10.0, guard_factor=0.0)

    def test_adaptive_policy_bootstrap_then_predict(self) -> None:
        policy = AdaptivePeriodicityPolicy(predictor=MovingAveragePredictor(window=2), bootstrap_periodicity_ms=10.0)
        self.assertEqual(policy.next_periodicity([]), 10.0)
        self.assertEqual(policy.next_periodicity([12.0, 18.0]), 10.0)

    def test_fixed_policy_is_constant(self) -> None:
        policy = FixedPeriodicityPolicy(periodicity_ms=20.0)
        self.assertEqual(policy.next_periodicity([5.0, 10.0]), 20.0)

    def test_optional_backend_missing_dependency_raises_custom_error(self) -> None:
        model = OptionalBackendPredictor(
            name='missing_backend',
            dependency_module='definitely_missing_dependency_xyz',
            estimator_path='definitely_missing_dependency_xyz.Model',
        )
        with self.assertRaises(ModelDependencyError):
            model.fit([[1.0, 2.0, 3.0]], [4.0])


if __name__ == '__main__':
    unittest.main()
