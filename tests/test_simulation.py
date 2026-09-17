import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_bsr.data_ingestion import PacketArrival
from ml_bsr.evaluation import build_adaptive_training_records, compare_strategies
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

    def test_simulation_preserves_global_time_order_across_ues(self) -> None:
        arrivals = [
            PacketArrival(time_ms=12.0, ue_id='ue-a'),
            PacketArrival(time_ms=13.0, ue_id='ue-a'),
            PacketArrival(time_ms=2.0, ue_id='ue-b'),
            PacketArrival(time_ms=7.0, ue_id='ue-b'),
        ]
        result = simulate_bsr_schedule(arrivals, FixedPeriodicityPolicy(periodicity_ms=5.0))
        self.assertEqual(result.total_packets, 4)
        self.assertEqual(result.packet_latencies_ms, [4.0, 4.0, 4.0, 3.0])

    def test_adaptive_training_records_can_target_single_ue(self) -> None:
        arrivals = [
            PacketArrival(time_ms=1.0, ue_id='ue-a'),
            PacketArrival(time_ms=2.0, ue_id='ue-a'),
            PacketArrival(time_ms=4.0, ue_id='ue-a'),
            PacketArrival(time_ms=3.0, ue_id='ue-b'),
            PacketArrival(time_ms=4.0, ue_id='ue-b'),
        ]
        records = build_adaptive_training_records(arrivals, window_size=1, ue_id='ue-a')
        self.assertEqual(records, [{'history_ms': [1.0], 'target_ms': 2.0}])

    def test_adaptive_training_records_combine_multi_ue_histories_by_default(self) -> None:
        arrivals = [
            PacketArrival(time_ms=1.0, ue_id='ue-a'),
            PacketArrival(time_ms=3.0, ue_id='ue-a'),
            PacketArrival(time_ms=6.0, ue_id='ue-a'),
            PacketArrival(time_ms=2.0, ue_id='ue-b'),
            PacketArrival(time_ms=5.0, ue_id='ue-b'),
            PacketArrival(time_ms=9.0, ue_id='ue-b'),
        ]
        records = build_adaptive_training_records(arrivals, window_size=1)
        self.assertEqual(records, [
            {'history_ms': [2.0], 'target_ms': 3.0},
            {'history_ms': [3.0], 'target_ms': 4.0},
        ])


if __name__ == '__main__':
    unittest.main()
