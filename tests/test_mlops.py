import json
import tempfile
import unittest
from pathlib import Path

from mlops import TrainingSample, evaluate_baseline, save_experiment_report, train_baseline


class MLOpsPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.samples = [
            TrainingSample(latency_ms=12, buffer_growth_rate=0.1, cell_load_ratio=0.45, label_periodicity_ms=40),
            TrainingSample(latency_ms=19, buffer_growth_rate=0.2, cell_load_ratio=0.50, label_periodicity_ms=35),
            TrainingSample(latency_ms=26, buffer_growth_rate=0.4, cell_load_ratio=0.70, label_periodicity_ms=25),
        ]

    def test_training_and_evaluation(self) -> None:
        model = train_baseline(self.samples)
        metrics = evaluate_baseline(model, self.samples)

        self.assertEqual(model["trained_samples"], 3)
        self.assertEqual(metrics["evaluated_samples"], 3)
        self.assertGreaterEqual(metrics["mae"], 0.0)

    def test_report_contains_research_goal(self) -> None:
        model = train_baseline(self.samples)
        metrics = evaluate_baseline(model, self.samples)
        goal = "Reduce uplink latency while keeping control overhead stable."

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = str(Path(temp_dir) / "reports" / "experiment.json")
            saved_path = save_experiment_report(model, metrics, goal, output_path)

            report = json.loads(Path(saved_path).read_text(encoding="utf-8"))
            self.assertEqual(report["research_goal"], goal)
            self.assertIn("generated_at", report)


if __name__ == "__main__":
    unittest.main()
