# ML-Aided Dynamic BSR Periodicity Adjustment for Enhanced UL Scheduling

This repository now contains both:

- a lightweight root-level baseline for heuristic/agent experimentation (`agent.py`, `mlops.py`), and
- a Python-first research scaffold for broader ML-aided BSR periodicity experiments under `src/ml_bsr/`.

The accompanying paper and supporting reading notes live under `docs/`.

## Repository structure

```text
.
├── agent.py                    # Lightweight heuristic BSR agent
├── mlops.py                    # Minimal baseline training/evaluation helpers
├── configs/                    # Experiment, model, and evaluation configs
├── data/
│   ├── processed/              # Derived datasets and split outputs
│   ├── raw/                    # Raw traces
│   └── samples/                # Small sample inputs for local experiments
├── docs/
│   ├── figures/                # Architecture or result figures
│   ├── ML-Aided_Dynamic_BSR_Periodicity_Adjustment_for_En.pdf
│   └── architecture.md
├── notebooks/                  # Exploratory analysis and reproduction notebooks
├── results/                    # Generated metrics, plots, and summaries
├── scripts/                    # Repeatable CLI entry points
├── src/ml_bsr/
│   ├── data_ingestion/         # CSV loading and synthetic trace generation
│   ├── evaluation/             # Metric aggregation and strategy comparison
│   ├── features/               # Feature engineering over interarrival histories
│   ├── models/                 # Predictor interface and baseline/optional models
│   ├── policy/                 # 3GPP-constrained BSR periodicity mapping
│   ├── preprocessing/          # Normalization, interarrival extraction, dataset splits
│   ├── simulation/             # Simplified UE/gNB scheduling simulation
│   └── utils/                  # JSON config and logging helpers
└── tests/                      # Unit tests for root modules and scaffold workflow
```

## Root baseline modules

### `agent.py`

Provides a simple heuristic dynamic-BSR controller:

- `BSRState`: input state for periodicity decisions
- `DynamicBSRAgent`: bounded periodicity recommendation logic

The current heuristic becomes more aggressive when `cell_load_ratio > 0.85`.

### `mlops.py`

Provides a minimal baseline experiment pipeline:

- `TrainingSample`
- `train_baseline`
- `evaluate_baseline`
- `save_experiment_report`
- `serialize_samples`

These modules are useful for simple experiments without depending on the full scaffold.

## Research scaffold pipeline

1. **Inputs**: packet arrival traces from CSV or synthetic generation.
2. **Preprocessing**: sort and normalize arrivals, extract interarrival times, and build supervised windows.
3. **Prediction**: estimate the next interarrival time with a common predictor interface.
4. **Policy**: map predictions to valid 3GPP BSR periodicities.
5. **Simulation**: replay traffic under fixed and adaptive BSR strategies.
6. **Evaluation**: compare latency, BSR count, and ineffective BSR reports.

## Implemented scaffold modules

- `data_ingestion`: packet arrival schema, CSV I/O, and synthetic trace generation
- `preprocessing`: normalization, interarrival extraction, dataset windowing, and split handling
- `features`: summary-statistic feature builders for model inputs
- `models`: shared predictor API, baseline predictors, and optional wrappers for SVR, Random Forest, XGBoost, and LSTM backends
- `policy`: valid BSR periodicity values and adaptive/fixed policy objects
- `simulation`: simplified multi-UE BSR scheduling approximation
- `evaluation`: comparison helpers and metric serialization
- `utils`: JSON config loading and logger creation

## Setup

- Python 3.10+ recommended.
- No third-party dependencies are required for the built-in baseline workflow and repository tests.
- Optional ML backends require their own dependencies if you choose to use them.

## Quick start

Create a sample dataset:

```bash
python scripts/prepare_dataset.py --config configs/dataset.sample.json
```

Train a baseline predictor:

```bash
python scripts/train_model.py --config configs/model.sample.json
```

Run fixed-vs-adaptive evaluation:

```bash
python scripts/run_evaluation.py --config configs/evaluation.sample.json
```

## Testing

Run repository tests with:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Notes

- This repository is a research and experimentation codebase, not a production-grade 5G stack.
- The ML wrappers for Random Forest, SVR, XGBoost, and LSTM backends degrade with a clear error message when their optional dependencies are unavailable.
- `docs/PAPER_UNDERSTANDING_GUIDE.md` provides a concise reading companion for the paper.
