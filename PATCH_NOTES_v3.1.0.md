# CALO-RPD Studio v3.1.0 Patch Notes

## Batched Throughput Engine

Version 3.1.0 keeps one runtime alive per verified CUDA/XPU/CPU device, batches compatible candidate evaluations from simultaneous independent runs, calibrates the FP64 evaluator microbatch on each device, and allocates complete jobs according to measured candidate-scenario throughput.

## Comparative evaluation

- Persistent per-device workers avoid repeated CUDA/XPU context and model initialization.
- Cross-run batching preserves run ordering and evaluation accounting.
- Device calibration is outside algorithm budgets and is stored in provenance.
- The scheduler records calibrated batch sizes, measured throughput, selected lane, and stage timing.
- Manual weighted and reference CPU modes remain available.

## Policy training

- CUDA/XPU actor interpreters and CPU rollout workers can remain persistent for all epochs.
- Calibration episodes are discarded before PPO collection.
- Fresh episode shares can be allocated by measured complete actor transitions per second.
- ORPD development rollouts can use accelerator-native FP64 evaluation and cross-episode batching.
- One policy snapshot is enforced for every PPO epoch; stale trajectories remain rejected.

## Scientific safeguards

The throughput layer does not modify optimizer equations, seeds, initial populations, constraint normalization, robust aggregation, discrete decoding, evaluation budgets, or final independent validation. CPU/accelerator parity remains mandatory when configured.
