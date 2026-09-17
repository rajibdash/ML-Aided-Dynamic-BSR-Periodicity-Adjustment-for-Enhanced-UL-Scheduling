# ML-Aided Dynamic BSR Periodicity Adjustment for Enhanced UL Scheduling

This repository contains a Python-first research scaffold for exploring the idea described in the accompanying paper: using traffic-interarrival prediction to adapt Buffer Status Report (BSR) periodicity in cellular uplink scheduling.

## Repository structure

```text
/home/runner/work/ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling/ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling
├── configs/                 # Experiment, model, and evaluation configs
├── data/
│   ├── processed/           # Derived datasets and split outputs
│   ├── raw/                 # Raw traces
│   └── samples/             # Small sample inputs for local experiments
├── docs/
│   ├── figures/             # Architecture or result figures
│   └── ML-Aided_Dynamic_BSR_Periodicity_Adjustment_for_En.pdf
├── notebooks/               # Exploratory analysis and reproduction notebooks
├── results/                 # Generated metrics, plots, and summaries
├── scripts/                 # Repeatable CLI entry points
├── src/ml_bsr/
│   ├── data_ingestion/      # CSV loading and synthetic trace generation
│   ├── evaluation/          # Metric aggregation and strategy comparison
│   ├── features/            # Feature engineering over interarrival histories
│   ├── models/              # Predictor interface and baseline/optional models
│   ├── policy/              # 3GPP-constrained BSR periodicity mapping
│   ├── preprocessing/       # Normalization, interarrival extraction, dataset splits
│   ├── simulation/          # Simplified UE/gNB scheduling simulation
│   └── utils/               # JSON config and logging helpers
└── tests/                   # Unit tests for the core workflow
```

## Research pipeline

1. **Inputs**: packet arrival traces from CSV or synthetic generation.
2. **Preprocessing**: sort and normalize arrivals, extract interarrival times, and build supervised windows.
3. **Prediction**: estimate the next interarrival time with a common predictor interface.
4. **Policy**: map predictions to valid 3GPP BSR periodicities.
5. **Simulation**: replay traffic under fixed and adaptive BSR strategies.
6. **Evaluation**: compare latency, BSR count, and ineffective BSR reports.

## Implemented modules

- `data_ingestion`: packet arrival schema, CSV I/O, and synthetic trace generation.
- `preprocessing`: normalization, interarrival extraction, dataset windowing, and train/validation/test splits.
- `features`: summary-statistic feature builders for model inputs.
- `models`: a shared predictor API with working baseline predictors and optional wrappers for SVR, Random Forest, XGBoost, and LSTM-style backends when their dependencies are available.
- `policy`: valid BSR periodicity values and adaptive/fixed policy objects.
- `simulation`: event-driven approximation of periodic BSR scheduling.
- `evaluation`: experiment comparison helpers and metric serialization.
- `utils`: JSON config loading and logger creation.

## Quick start

Create a sample dataset:

```bash
python scripts/prepare_dataset.py       --config /home/runner/work/ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling/ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling/configs/dataset.sample.json
```

Train a baseline predictor:

```bash
python scripts/train_model.py       --config /home/runner/work/ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling/ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling/configs/model.sample.json
```

Run fixed-vs-adaptive evaluation:

```bash
python scripts/run_evaluation.py       --config /home/runner/work/ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling/ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling/configs/evaluation.sample.json
```

## Testing

Run repository tests with:

```bash
python -m unittest discover -s /home/runner/work/ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling/ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling/tests -p "test_*.py"
```

## Notes

- The current implementation is a research scaffold, not a production-grade 5G stack.
- The ML wrappers for Random Forest, SVR, XGBoost, and LSTM backends intentionally degrade with a clear error message when their optional dependencies are unavailable.
- The included sample configs use only standard-library-compatible baseline predictors so the repository works out of the box.
