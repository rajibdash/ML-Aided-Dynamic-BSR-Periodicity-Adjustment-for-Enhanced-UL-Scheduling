from dataclasses import dataclass


@dataclass(frozen=True)
class BSRState:
    queue_bytes: int
    avg_ul_latency_ms: float
    buffer_growth_rate: float
    cell_load_ratio: float


class DynamicBSRAgent:
    def __init__(
        self,
        min_periodicity_ms: int = 5,
        max_periodicity_ms: int = 80,
        target_latency_ms: float = 20.0,
    ) -> None:
        if min_periodicity_ms <= 0 or max_periodicity_ms <= 0:
            raise ValueError("Periodicity bounds must be positive integers.")
        if min_periodicity_ms > max_periodicity_ms:
            raise ValueError("min_periodicity_ms must be <= max_periodicity_ms.")
        if target_latency_ms <= 0:
            raise ValueError("target_latency_ms must be greater than 0.")
        self.min_periodicity_ms = min_periodicity_ms
        self.max_periodicity_ms = max_periodicity_ms
        self.target_latency_ms = target_latency_ms

    def recommend_periodicity(self, state: BSRState) -> int:
        latency_factor = max(0.0, state.avg_ul_latency_ms / self.target_latency_ms)
        growth_factor = max(0.0, state.buffer_growth_rate)
        queue_factor = 1.2 if state.queue_bytes > 50_000 else 1.0
        load_factor = 1.15 if state.cell_load_ratio > 0.85 else 1.0

        urgency = (0.6 * latency_factor + 0.4 * growth_factor) * queue_factor * load_factor
        recommendation = int(self.max_periodicity_ms / (1.0 + urgency))
        recommendation = max(self.min_periodicity_ms, recommendation)
        recommendation = min(self.max_periodicity_ms, recommendation)
        return int(recommendation)
