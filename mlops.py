import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TrainingSample:
    latency_ms: float
    buffer_growth_rate: float
    cell_load_ratio: float
    label_periodicity_ms: int


def train_baseline(samples: Iterable[TrainingSample]) -> dict:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("At least one sample is required for training.")

    labels = [sample.label_periodicity_ms for sample in sample_list]
    baseline = sum(labels) / len(labels)
    return {"baseline_periodicity_ms": baseline, "trained_samples": len(sample_list)}


def evaluate_baseline(model: dict, samples: Iterable[TrainingSample]) -> dict:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("At least one sample is required for evaluation.")

    baseline = float(model["baseline_periodicity_ms"])
    mae = sum(abs(sample.label_periodicity_ms - baseline) for sample in sample_list) / len(sample_list)
    return {"mae": mae, "evaluated_samples": len(sample_list)}


def save_experiment_report(model: dict, metrics: dict, research_goal: str, output_path: str) -> str:
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_goal": research_goal,
        "model": model,
        "metrics": metrics,
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(destination)


def serialize_samples(samples: Iterable[TrainingSample]) -> list[dict]:
    return [asdict(sample) for sample in samples]
