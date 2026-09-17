from __future__ import annotations

import argparse

from ml_bsr.data_ingestion import read_packet_arrivals_csv
from ml_bsr.evaluation import compare_strategies
from ml_bsr.models import build_model
from ml_bsr.policy import AdaptivePeriodicityPolicy, FixedPeriodicityPolicy
from ml_bsr.utils import dump_json, load_json, project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed and adaptive BSR evaluation.")
    parser.add_argument("--config", required=True, help="Path to a JSON config file.")
    args = parser.parse_args()

    config = load_json(args.config)
    arrivals = read_packet_arrivals_csv(project_path(config["input_path"]))

    strategies = {}
    for periodicity in config.get("fixed_periodicities_ms", []):
        strategies[f"fixed_{int(periodicity)}ms"] = FixedPeriodicityPolicy(periodicity_ms=float(periodicity))

    adaptive_config = config.get("adaptive", {})
    predictor_config = adaptive_config.get("predictor", {})
    strategies["adaptive"] = AdaptivePeriodicityPolicy(
        predictor=build_model(predictor_config.get("name", "moving_average"), **predictor_config.get("params", {})),
        bootstrap_periodicity_ms=float(adaptive_config.get("bootstrap_periodicity_ms", 10.0)),
        guard_factor=float(adaptive_config.get("guard_factor", 0.9)),
    )

    results = compare_strategies(
        arrivals=arrivals,
        strategies=strategies,
        grant_processing_ms=float(config.get("grant_processing_ms", 1.0)),
    )

    output_path = project_path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump_json(results, output_path)
    print(f"Wrote evaluation summary to {output_path}")


if __name__ == "__main__":
    main()
