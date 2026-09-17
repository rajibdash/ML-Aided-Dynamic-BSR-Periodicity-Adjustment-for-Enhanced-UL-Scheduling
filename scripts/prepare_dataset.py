from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_bsr.data_ingestion import generate_synthetic_trace, read_packet_arrivals_csv, write_packet_arrivals_csv
from ml_bsr.preprocessing import build_supervised_records, extract_interarrival_times_by_ue, normalize_arrivals
from ml_bsr.utils import dump_json, load_json, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare packet-arrival datasets.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    config = load_json(args.config)
    mode = config.get("mode", "synthetic")
    if mode == "synthetic":
        arrivals = generate_synthetic_trace(
            count=int(config.get("count", 20)),
            base_interval_ms=float(config.get("base_interval_ms", 10.0)),
            jitter_ms=float(config.get("jitter_ms", 0.0)),
            seed=int(config.get("seed", 0)),
        )
        sample_csv = project_path('data/samples/sample_trace.csv')
        write_packet_arrivals_csv(arrivals, sample_csv)
    elif mode == "csv":
        arrivals = read_packet_arrivals_csv(project_path(config["input_path"]))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    arrivals = normalize_arrivals(arrivals)
    interarrivals_by_ue = extract_interarrival_times_by_ue(arrivals)
    window_size = int(config.get("window_size", 4))
    selected_ue_id = config.get("ue_id")
    if selected_ue_id is not None:
        records = build_supervised_records(interarrivals_by_ue.get(selected_ue_id, []), window_size=window_size)
    else:
        records = []
        for current_ue_id in sorted(interarrivals_by_ue):
            records.extend(build_supervised_records(interarrivals_by_ue[current_ue_id], window_size=window_size))

    output_path = project_path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(
        {
            "arrivals": [arrival.to_dict() for arrival in arrivals],
            "interarrivals_by_ue_ms": interarrivals_by_ue,
            "records": records,
        },
        output_path,
    )
    print(f"Wrote dataset to {output_path}")


if __name__ == "__main__":
    main()
