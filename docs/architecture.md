# Architecture overview

This repository organizes the adaptive BSR workflow into loosely coupled stages so that packet preprocessing, prediction, policy selection, simulation, and evaluation can evolve independently.

## Module mapping

- `src/ml_bsr/data_ingestion`: packet trace acquisition and schema normalization.
- `src/ml_bsr/preprocessing`: converts packet arrivals into interarrival histories and dataset splits.
- `src/ml_bsr/features`: feature summaries over recent interarrival windows.
- `src/ml_bsr/models`: predictor abstraction plus baseline and optional external-model adapters.
- `src/ml_bsr/policy`: maps predictions to valid 3GPP BSR periodicity values.
- `src/ml_bsr/simulation`: simulates BSR events, grants, and packet delay.
- `src/ml_bsr/evaluation`: compares fixed and adaptive strategies using common metrics.
- `src/ml_bsr/utils`: configuration loading and logging.

## Data flow

1. A traffic trace is loaded from CSV or generated synthetically.
2. Packet arrivals are normalized and converted into interarrival-time sequences.
3. Sliding-window records are built for model training or evaluation.
4. A predictor estimates the next interarrival time.
5. A policy maps the prediction to the nearest valid BSR periodicity under 3GPP constraints.
6. The simulator replays arrivals to produce latency and signaling metrics.
7. The evaluator exports comparable summaries for fixed and adaptive baselines.

## Scope

The code intentionally provides a lightweight experimentation baseline. It focuses on reproducible structure and clear module boundaries rather than implementing a standards-complete radio access network.
