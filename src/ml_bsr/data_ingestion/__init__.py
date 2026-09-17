from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class PacketArrival:
    time_ms: float
    ue_id: str = "ue-0"
    size_bytes: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def read_packet_arrivals_csv(path: str | Path) -> list[PacketArrival]:
    arrivals: list[PacketArrival] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            arrivals.append(
                PacketArrival(
                    time_ms=float(row["time_ms"]),
                    ue_id=row.get("ue_id", "ue-0") or "ue-0",
                    size_bytes=int(float(row.get("size_bytes", 0) or 0)),
                )
            )
    return arrivals


def write_packet_arrivals_csv(arrivals: list[PacketArrival], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["time_ms", "ue_id", "size_bytes"])
        writer.writeheader()
        for arrival in arrivals:
            writer.writerow(arrival.to_dict())


def generate_synthetic_trace(
    count: int,
    base_interval_ms: float,
    jitter_ms: float = 0.0,
    seed: int | None = None,
    ue_id: str = "ue-0",
    size_bytes: int = 0,
) -> list[PacketArrival]:
    if count <= 0:
        return []
    rng = random.Random(seed)
    time_ms = 0.0
    arrivals: list[PacketArrival] = []
    for _ in range(count):
        time_ms += max(0.1, base_interval_ms + rng.uniform(-jitter_ms, jitter_ms))
        arrivals.append(PacketArrival(time_ms=round(time_ms, 3), ue_id=ue_id, size_bytes=size_bytes))
    return arrivals
