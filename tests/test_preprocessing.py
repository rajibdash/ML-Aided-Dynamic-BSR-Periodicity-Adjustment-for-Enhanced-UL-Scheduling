import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_bsr.data_ingestion import PacketArrival
from ml_bsr.preprocessing import build_supervised_records, extract_interarrival_times, normalize_arrivals, split_records


class PreprocessingTests(unittest.TestCase):
    def test_normalize_and_extract_interarrivals(self) -> None:
        arrivals = [
            PacketArrival(time_ms=20.0),
            PacketArrival(time_ms=5.0),
            PacketArrival(time_ms=15.0),
        ]
        normalized = normalize_arrivals(arrivals)
        self.assertEqual([item.time_ms for item in normalized], [5.0, 15.0, 20.0])
        self.assertEqual(extract_interarrival_times(arrivals), [10.0, 5.0])

    def test_build_and_split_records(self) -> None:
        records = build_supervised_records([5.0, 7.0, 9.0, 11.0, 13.0], window_size=2)
        self.assertEqual(len(records), 3)
        splits = split_records(records, ratios=(0.5, 0.25, 0.25))
        self.assertEqual(len(splits['train']), 1)
        self.assertEqual(len(splits['validation']), 0)
        self.assertEqual(len(splits['test']), 2)


if __name__ == '__main__':
    unittest.main()
