from __future__ import annotations

from dataclasses import dataclass
from math import inf

from ml_bsr.data_ingestion import PacketArrival
from ml_bsr.policy import AdaptivePeriodicityPolicy, FixedPeriodicityPolicy
from ml_bsr.preprocessing import normalize_arrivals

PolicyType = FixedPeriodicityPolicy | AdaptivePeriodicityPolicy


@dataclass
class SimulationResult:
    total_packets: int
    total_bsr_reports: int
    ineffective_bsr_reports: int
    average_latency_ms: float
    packet_latencies_ms: list[float]
    selected_periodicities_ms: list[float]

    def to_dict(self) -> dict[str, object]:
        return {
            "total_packets": self.total_packets,
            "total_bsr_reports": self.total_bsr_reports,
            "ineffective_bsr_reports": self.ineffective_bsr_reports,
            "average_latency_ms": self.average_latency_ms,
            "packet_latencies_ms": self.packet_latencies_ms,
            "selected_periodicities_ms": self.selected_periodicities_ms,
        }


def simulate_bsr_schedule(
    arrivals: list[PacketArrival],
    policy: PolicyType,
    grant_processing_ms: float = 1.0,
) -> SimulationResult:
    normalized = normalize_arrivals(arrivals)
    if not normalized:
        return SimulationResult(0, 0, 0, 0.0, [], [])

    arrival_times = [arrival.time_ms for arrival in normalized]
    observed_interarrivals: list[float] = []
    next_periodicity = policy.next_periodicity(observed_interarrivals)
    current_time = next_periodicity if next_periodicity != inf else inf
    next_arrival_index = 0
    pending_arrivals: list[float] = []
    latencies: list[float] = []
    total_bsr_reports = 0
    ineffective_bsr_reports = 0
    selected_periodicities = [next_periodicity]
    last_seen_arrival_by_ue: dict[str, float] = {}

    while next_arrival_index < len(arrival_times) or pending_arrivals:
        if current_time == inf:
            break
        next_arrival_time = arrival_times[next_arrival_index] if next_arrival_index < len(arrival_times) else inf

        if next_arrival_time <= current_time:
            arrival = normalized[next_arrival_index]
            pending_arrivals.append(next_arrival_time)
            if arrival.ue_id in last_seen_arrival_by_ue:
                observed_interarrivals.append(next_arrival_time - last_seen_arrival_by_ue[arrival.ue_id])
            last_seen_arrival_by_ue[arrival.ue_id] = next_arrival_time
            next_arrival_index += 1
            continue

        total_bsr_reports += 1
        if pending_arrivals:
            for pending_time in pending_arrivals:
                latencies.append(round((current_time - pending_time) + grant_processing_ms, 6))
            pending_arrivals.clear()
        else:
            ineffective_bsr_reports += 1

        next_periodicity = policy.next_periodicity(observed_interarrivals)
        selected_periodicities.append(next_periodicity)
        if next_periodicity == inf:
            current_time = inf
        else:
            current_time += next_periodicity

    average_latency = round(sum(latencies) / len(latencies), 6) if latencies else 0.0
    return SimulationResult(
        total_packets=len(arrival_times),
        total_bsr_reports=total_bsr_reports,
        ineffective_bsr_reports=ineffective_bsr_reports,
        average_latency_ms=average_latency,
        packet_latencies_ms=latencies,
        selected_periodicities_ms=selected_periodicities,
    )
