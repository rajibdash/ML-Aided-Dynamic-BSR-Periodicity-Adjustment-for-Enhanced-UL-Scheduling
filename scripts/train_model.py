from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_bsr.features import build_feature_matrix
from ml_bsr.models import build_model
from ml_bsr.preprocessing import split_records
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
    if not records:
        raise ValueError("dataset must contain at least one record")

    splits = split_records(records, tuple(config.get("split_ratios", [0.7, 0.15, 0.15])))
    train_records = splits["train"] or records
    evaluation_records = splits["validation"] or splits["test"] or train_records
    histories = [list(record["history_ms"]) for record in train_records]
    _, targets = build_feature_matrix(train_records)

    model_config = config.get("model", {})
    model = build_model(model_config.get("name", "moving_average"), **model_config.get("params", {}))
    model.fit(histories, targets)
    eval_histories = [list(record["history_ms"]) for record in evaluation_records]
    _, evaluation_targets = build_feature_matrix(evaluation_records)
    predictions = [model.predict(history) for history in eval_histories]

    summary = {
        "model": model.describe(),
        "record_count": len(records),
        "training_record_count": len(train_records),
        "evaluation_record_count": len(evaluation_records),
        "metrics": {
            "mae_ms": mean_absolute_error(evaluation_targets, predictions),
        },
    }

    output_path = project_path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(summary, output_path)
    print(f"Wrote model summary to {output_path}")


if __name__ == "__main__":
    main()
