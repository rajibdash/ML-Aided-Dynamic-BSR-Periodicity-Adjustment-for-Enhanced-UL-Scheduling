from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_bsr.features import build_feature_matrix
from ml_bsr.models import build_model
from ml_bsr.utils import dump_json, load_json, project_path


def mean_absolute_error(actual: list[float], predicted: list[float]) -> float:
    if len(actual) != len(predicted):
        raise ValueError("actual and predicted must have the same length")
    if not actual:
        return 0.0
    return sum(abs(a - b) for a, b in zip(actual, predicted)) / len(actual)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline interarrival predictor.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    config = load_json(args.config)
    dataset = load_json(project_path(config["dataset_path"]))
    records = dataset.get("records", [])
    histories = [list(record["history_ms"]) for record in records]
    _, targets = build_feature_matrix(records)

    model_config = config.get("model", {})
    model = build_model(model_config.get("name", "moving_average"), **model_config.get("params", {}))
    model.fit(histories, targets)
    predictions = [model.predict(history) for history in histories]

    summary = {
        "model": model.describe(),
        "record_count": len(records),
        "metrics": {
            "mae_ms": mean_absolute_error(targets, predictions),
        },
    }

    output_path = project_path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(summary, output_path)
    print(f"Wrote model summary to {output_path}")


if __name__ == "__main__":
    main()
