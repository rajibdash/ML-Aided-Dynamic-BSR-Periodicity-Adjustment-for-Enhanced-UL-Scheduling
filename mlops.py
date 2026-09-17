import json
from collections.abc import Mapping
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


@dataclass(frozen=True)
class BaselineModel:
    baseline_periodicity_ms: float
    trained_samples: int


def train_baseline(samples: Iterable[TrainingSample]) -> BaselineModel:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("At least one sample is required for training.")

    labels = [sample.label_periodicity_ms for sample in sample_list]
    baseline = sum(labels) / len(labels)
    return BaselineModel(baseline_periodicity_ms=baseline, trained_samples=len(sample_list))


def _extract_baseline_value(model: BaselineModel | Mapping[str, object]) -> float:
    if isinstance(model, BaselineModel):
        return model.baseline_periodicity_ms
    if not isinstance(model, Mapping) or "baseline_periodicity_ms" not in model:
        raise ValueError("Model must contain 'baseline_periodicity_ms'.")
    return float(model["baseline_periodicity_ms"])


def evaluate_baseline(model: BaselineModel | Mapping[str, object], samples: Iterable[TrainingSample]) -> dict:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("At least one sample is required for evaluation.")

    baseline = _extract_baseline_value(model)
    mae = sum(abs(sample.label_periodicity_ms - baseline) for sample in sample_list) / len(sample_list)
    return {"mae": mae, "evaluated_samples": len(sample_list)}


def save_experiment_report(
    model: BaselineModel | Mapping[str, object], metrics: Mapping[str, object], research_goal: str, output_path: str
) -> str:
    if isinstance(model, BaselineModel):
        model_payload = asdict(model)
    elif isinstance(model, Mapping):
        model_payload = dict(model)
    else:
        raise ValueError("Model must be a BaselineModel or mapping with JSON-serializable values.")

    if not isinstance(metrics, Mapping):
        raise ValueError("Metrics must be a mapping with JSON-serializable values.")

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_goal": research_goal,
        "model": model_payload,
        "metrics": dict(metrics),
    }
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        serialized_report = json.dumps(report, indent=2)
    except TypeError as exc:
        raise ValueError("Model and metrics must be JSON-serializable.") from exc
    destination.write_text(serialized_report, encoding="utf-8")
    return str(destination)


def serialize_samples(samples: Iterable[TrainingSample]) -> list[dict]:
    return [asdict(sample) for sample in samples]
