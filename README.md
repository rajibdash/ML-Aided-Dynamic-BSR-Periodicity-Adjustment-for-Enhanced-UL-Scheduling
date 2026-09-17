# ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling

This repository includes a minimal **Agent** module (`agent.py`) and a minimal **mlops** module (`mlops.py`) for ML-aided dynamic BSR periodicity experiments.

## Setup

- Python 3.10+ recommended.
- No third-party dependencies are required for the examples and tests in this repository.
- Run tests with:
  - `python -m unittest discover -s tests -p "test_*.py"`

## Research Goal

Primary goal:

- Reduce uplink latency by dynamically adapting BSR periodicity based on queue, growth, and load conditions while maintaining stable control signaling overhead.

## Agent Module

File: `agent.py`

- `BSRState`: input state for periodicity decisions.
- `DynamicBSRAgent`: recommends BSR periodicity (ms) with bounded outputs, using more frequent BSR reporting under high-load/high-urgency conditions.

Example:

```python
from agent import BSRState, DynamicBSRAgent

agent = DynamicBSRAgent()
state = BSRState(queue_bytes=60000, avg_ul_latency_ms=24.0, buffer_growth_rate=0.5, cell_load_ratio=0.9)
recommended_periodicity_ms = agent.recommend_periodicity(state)
print(recommended_periodicity_ms)
```

## mlops Module

File: `mlops.py`

- `TrainingSample`: data structure for training/evaluation examples.
- `train_baseline`: builds a baseline model from samples.
- `evaluate_baseline`: computes MAE for the baseline.
- `save_experiment_report`: stores model, metrics, and research goal as JSON.

Example:

```python
from mlops import TrainingSample, train_baseline, evaluate_baseline, save_experiment_report

samples = [
    TrainingSample(latency_ms=12, buffer_growth_rate=0.1, cell_load_ratio=0.5, label_periodicity_ms=35),
    TrainingSample(latency_ms=20, buffer_growth_rate=0.3, cell_load_ratio=0.7, label_periodicity_ms=25),
]
model = train_baseline(samples)
metrics = evaluate_baseline(model, samples)
save_experiment_report(model, metrics, "Reduce UL latency while controlling overhead", "artifacts/experiment.json")
```

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```
