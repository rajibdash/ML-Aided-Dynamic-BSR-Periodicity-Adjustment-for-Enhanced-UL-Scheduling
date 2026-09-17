from __future__ import annotations

from ml_bsr.data_ingestion import PacketArrival
from ml_bsr.policy import AdaptivePeriodicityPolicy, FixedPeriodicityPolicy
from ml_bsr.preprocessing import build_supervised_records, extract_interarrival_times
from ml_bsr.simulation import SimulationResult, simulate_bsr_schedule

StrategyType = FixedPeriodicityPolicy | AdaptivePeriodicityPolicy


def summarize_result(result: SimulationResult) -> dict[str, object]:
    efficiency = 0.0
    if result.total_bsr_reports:
        efficiency = (result.total_bsr_reports - result.ineffective_bsr_reports) / result.total_bsr_reports
    return {
        **result.to_dict(),
        "effective_bsr_ratio": round(efficiency, 6),
    }


def compare_strategies(
    arrivals: list[PacketArrival],
    strategies: dict[str, StrategyType],
    grant_processing_ms: float = 1.0,
) -> dict[str, dict[str, object]]:
    comparison: dict[str, dict[str, object]] = {}
    for name, strategy in strategies.items():
        result = simulate_bsr_schedule(arrivals, strategy, grant_processing_ms=grant_processing_ms)
        comparison[name] = {
            "policy": strategy.describe(),
            "metrics": summarize_result(result),
        }
    return comparison


def build_adaptive_training_records(
    arrivals: list[PacketArrival],
    window_size: int,
    ue_id: str | None = None,
) -> list[dict[str, list[float] | float]]:
    filtered_arrivals = [arrival for arrival in arrivals if ue_id is None or arrival.ue_id == ue_id]
    interarrivals = extract_interarrival_times(filtered_arrivals)
    return build_supervised_records(interarrivals, window_size=window_size)
