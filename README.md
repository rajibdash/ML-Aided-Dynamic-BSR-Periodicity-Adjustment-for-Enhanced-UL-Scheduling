# ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling

This repository now includes a minimal **Agent** module and a minimal **ML Ops (`mlops.py`)** module for ML-aided dynamic BSR periodicity experiments.

## Research Goal

Primary goal:

- Reduce uplink latency by dynamically adapting BSR periodicity based on queue, growth, and load conditions while maintaining stable control signaling overhead.

## Agent Module

File: `agent.py`

- `BSRState`: input state for periodicity decisions.
- `DynamicBSRAgent`: recommends BSR periodicity (ms) with bounded outputs, using more frequent BSR reporting under high-load/high-urgency conditions.

## ML Ops Module

File: `mlops.py`

- `TrainingSample`: data structure for training/evaluation examples.
- `train_baseline`: builds a baseline model from samples.
- `evaluate_baseline`: computes MAE for the baseline.
- `save_experiment_report`: stores model, metrics, and research goal as JSON.

Example:

```python
from mlops import TrainingSample, train_baseline, evaluate_baseline, save_experiment_report

samples = [TrainingSample(12, 0.1, 0.5, 35), TrainingSample(20, 0.3, 0.7, 25)]
model = train_baseline(samples)
metrics = evaluate_baseline(model, samples)
save_experiment_report(model, metrics, "Reduce UL latency while controlling overhead", "artifacts/experiment.json")
```

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```
