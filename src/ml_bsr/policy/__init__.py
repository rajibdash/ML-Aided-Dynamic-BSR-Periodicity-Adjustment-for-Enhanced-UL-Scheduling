from __future__ import annotations

from dataclasses import dataclass
from math import isinf

from ml_bsr.models import Predictor

VALID_BSR_PERIODICITIES_MS: tuple[float, ...] = (
    1.0, 5.0, 10.0, 16.0, 20.0, 32.0, 40.0, 64.0, 80.0, 128.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, float('inf')
)


def select_bsr_periodicity(predicted_interarrival_ms: float, guard_factor: float = 0.9) -> float:
    if predicted_interarrival_ms <= 0.0:
        return VALID_BSR_PERIODICITIES_MS[0]
    if isinf(predicted_interarrival_ms):
        return VALID_BSR_PERIODICITIES_MS[-1]
    target = predicted_interarrival_ms * guard_factor
    finite = [value for value in VALID_BSR_PERIODICITIES_MS if value != float('inf')]
    eligible = [value for value in finite if value <= target]
    return eligible[-1] if eligible else finite[0]


@dataclass
class FixedPeriodicityPolicy:
    periodicity_ms: float

    def next_periodicity(self, observed_interarrivals_ms: list[float]) -> float:
        return self.periodicity_ms

    def describe(self) -> dict[str, object]:
        return {"type": "fixed", "periodicity_ms": self.periodicity_ms}


@dataclass
class AdaptivePeriodicityPolicy:
    predictor: Predictor
    bootstrap_periodicity_ms: float = 10.0
    guard_factor: float = 0.9

    def next_periodicity(self, observed_interarrivals_ms: list[float]) -> float:
        if len(observed_interarrivals_ms) < 2:
            return self.bootstrap_periodicity_ms
        predicted = self.predictor.predict(observed_interarrivals_ms)
        return select_bsr_periodicity(predicted, guard_factor=self.guard_factor)

    def describe(self) -> dict[str, object]:
        return {
            "type": "adaptive",
            "bootstrap_periodicity_ms": self.bootstrap_periodicity_ms,
            "guard_factor": self.guard_factor,
            "predictor": self.predictor.describe(),
        }
