import unittest

from ml_bsr.data_ingestion import PacketArrival
from ml_bsr.evaluation import compare_strategies
from ml_bsr.models import MovingAveragePredictor
from ml_bsr.policy import AdaptivePeriodicityPolicy, FixedPeriodicityPolicy
from ml_bsr.simulation import simulate_bsr_schedule


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.arrivals = [
            PacketArrival(time_ms=2.0),
            PacketArrival(time_ms=7.0),
            PacketArrival(time_ms=13.0),
            PacketArrival(time_ms=21.0),
        ]

    def test_fixed_policy_simulation_counts_packets(self) -> None:
        result = simulate_bsr_schedule(self.arrivals, FixedPeriodicityPolicy(periodicity_ms=5.0))
        self.assertEqual(result.total_packets, 4)
        self.assertGreaterEqual(result.total_bsr_reports, 1)
        self.assertEqual(len(result.packet_latencies_ms), 4)

    def test_compare_strategies_returns_metrics(self) -> None:
        strategies = {
            'fixed': FixedPeriodicityPolicy(periodicity_ms=10.0),
            'adaptive': AdaptivePeriodicityPolicy(predictor=MovingAveragePredictor(window=2), bootstrap_periodicity_ms=10.0),
        }
        comparison = compare_strategies(self.arrivals, strategies)
        self.assertIn('fixed', comparison)
        self.assertIn('adaptive', comparison)
        self.assertIn('metrics', comparison['fixed'])


if __name__ == '__main__':
    unittest.main()
