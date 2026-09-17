# ML-Aided-Dynamic-BSR-Periodicity-Adjustment-for-Enhanced-UL-Scheduling

This repository now includes a minimal **Agent** module and a minimal **ML Ops** module for ML-aided dynamic BSR periodicity experiments.

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

## Run Tests

```bash
python -m unittest discover -s tests -p "test_*.py"
```
