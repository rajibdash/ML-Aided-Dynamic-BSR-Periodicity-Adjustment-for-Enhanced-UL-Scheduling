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

    ue_ids = sorted({arrival.ue_id for arrival in normalized})
    remaining_arrivals_by_ue = {ue_id: 0 for ue_id in ue_ids}
    for arrival in normalized:
        remaining_arrivals_by_ue[arrival.ue_id] += 1

    ue_state = {
        ue_id: {
            "observed_interarrivals": [],
            "pending_arrivals": [],
            "last_seen_arrival": None,
            "trailing_empty_report_counted": False,
            "next_periodicity": policy.next_periodicity([]),
            "next_bsr_time": policy.next_periodicity([]),
        }
        for ue_id in ue_ids
    }
    next_arrival_index = 0
    latencies: list[float] = []
    total_bsr_reports = 0
    ineffective_bsr_reports = 0
    selected_periodicities = [state["next_periodicity"] for state in ue_state.values()]

    while True:
        next_bsr_ue = min(ue_ids, key=lambda ue_id: ue_state[ue_id]["next_bsr_time"])
        current_time = ue_state[next_bsr_ue]["next_bsr_time"]
        if current_time == inf:
            break
        if (
            next_arrival_index >= len(normalized)
            and all(not ue_state[ue_id]["pending_arrivals"] for ue_id in ue_ids)
            and all(ue_state[ue_id]["trailing_empty_report_counted"] or ue_state[ue_id]["next_bsr_time"] == inf for ue_id in ue_ids)
        ):
            break
        next_arrival = normalized[next_arrival_index] if next_arrival_index < len(normalized) else None
        next_arrival_time = next_arrival.time_ms if next_arrival is not None else inf

        if next_arrival_time <= current_time:
            arrival = next_arrival
            if arrival is None:
                break
            state = ue_state[arrival.ue_id]
            state["pending_arrivals"].append(next_arrival_time)
            if state["last_seen_arrival"] is not None:
                state["observed_interarrivals"].append(next_arrival_time - state["last_seen_arrival"])
            state["last_seen_arrival"] = next_arrival_time
            state["trailing_empty_report_counted"] = False
            remaining_arrivals_by_ue[arrival.ue_id] -= 1
            next_arrival_index += 1
            continue

        state = ue_state[next_bsr_ue]
        total_bsr_reports += 1
        pending_arrivals = state["pending_arrivals"]
        if pending_arrivals:
            for pending_time in pending_arrivals:
                latencies.append(round((current_time - pending_time) + grant_processing_ms, 6))
            pending_arrivals.clear()
            state["trailing_empty_report_counted"] = False
        else:
            ineffective_bsr_reports += 1
            if remaining_arrivals_by_ue[next_bsr_ue] == 0:
                state["trailing_empty_report_counted"] = True

        next_periodicity = policy.next_periodicity(list(state["observed_interarrivals"]))
        selected_periodicities.append(next_periodicity)
        state["next_periodicity"] = next_periodicity
        if next_periodicity == inf or (state["trailing_empty_report_counted"] and remaining_arrivals_by_ue[next_bsr_ue] == 0):
            state["next_bsr_time"] = inf
        else:
            state["next_bsr_time"] = current_time + next_periodicity

    average_latency = round(sum(latencies) / len(latencies), 6) if latencies else 0.0
    return SimulationResult(
        total_packets=len(normalized),
        total_bsr_reports=total_bsr_reports,
        ineffective_bsr_reports=ineffective_bsr_reports,
        average_latency_ms=average_latency,
        packet_latencies_ms=latencies,
        selected_periodicities_ms=selected_periodicities,
    )
