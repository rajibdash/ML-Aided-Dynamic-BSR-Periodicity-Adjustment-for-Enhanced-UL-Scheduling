from __future__ import annotations

from statistics import mean


def summarize_interarrivals(history_ms: list[float]) -> dict[str, float]:
    if not history_ms:
        raise ValueError("history_ms must not be empty")
    return {
        "latest_ms": history_ms[-1],
        "mean_ms": mean(history_ms),
        "min_ms": min(history_ms),
        "max_ms": max(history_ms),
        "range_ms": max(history_ms) - min(history_ms),
    }


def build_feature_vector(history_ms: list[float]) -> list[float]:
    summary = summarize_interarrivals(history_ms)
    return [
        summary["latest_ms"],
        summary["mean_ms"],
        summary["min_ms"],
        summary["max_ms"],
        summary["range_ms"],
        float(len(history_ms)),
    ]


def build_feature_matrix(records: list[dict[str, list[float] | float]]) -> tuple[list[list[float]], list[float]]:
    features: list[list[float]] = []
    targets: list[float] = []
    for record in records:
        history = list(record["history_ms"])
        features.append(build_feature_vector(history))
        targets.append(float(record["target_ms"]))
    return features, targets
