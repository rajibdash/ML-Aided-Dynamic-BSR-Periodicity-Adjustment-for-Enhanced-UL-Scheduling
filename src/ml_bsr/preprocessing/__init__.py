from __future__ import annotations

from math import isclose
from math import floor

from ml_bsr.data_ingestion import PacketArrival


def normalize_arrivals(arrivals: list[PacketArrival]) -> list[PacketArrival]:
    return sorted((arrival for arrival in arrivals if arrival.time_ms >= 0.0), key=lambda item: (item.time_ms, item.ue_id))


def extract_interarrival_times_by_ue(arrivals: list[PacketArrival]) -> dict[str, list[float]]:
    normalized = normalize_arrivals(arrivals)
    grouped: dict[str, list[PacketArrival]] = {}
    for arrival in normalized:
        grouped.setdefault(arrival.ue_id, []).append(arrival)

    interarrivals_by_ue: dict[str, list[float]] = {}
    for ue_id, ue_arrivals in grouped.items():
        interarrivals_by_ue[ue_id] = [
            round(current.time_ms - previous.time_ms, 6)
            for previous, current in zip(ue_arrivals, ue_arrivals[1:])
        ]
    return interarrivals_by_ue


def extract_interarrival_times(arrivals: list[PacketArrival]) -> list[float]:
    interarrivals_by_ue = extract_interarrival_times_by_ue(arrivals)
    if len(interarrivals_by_ue) > 1:
        raise ValueError("extract_interarrival_times requires arrivals from a single UE")
    return next(iter(interarrivals_by_ue.values()), [])


def build_supervised_records(interarrivals_ms: list[float], window_size: int) -> list[dict[str, list[float] | float]]:
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    records: list[dict[str, list[float] | float]] = []
    for index in range(window_size, len(interarrivals_ms)):
        history = interarrivals_ms[index - window_size:index]
        target = interarrivals_ms[index]
        records.append({"history_ms": history, "target_ms": target})
    return records


def split_records(
    records: list[dict[str, list[float] | float]],
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> dict[str, list[dict[str, list[float] | float]]]:
    if len(ratios) != 3:
        raise ValueError("split_records requires exactly three ratios")
    if any(ratio < 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError("Split ratios must be between 0 and 1")
    if not isclose(sum(ratios), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1.0")
    total = len(records)
    train_end = floor(total * ratios[0])
    validation_end = train_end + floor(total * ratios[1])
    return {
        "train": records[:train_end],
        "validation": records[train_end:validation_end],
        "test": records[validation_end:],
    }
