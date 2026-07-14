# CALO-RPD Studio v2.0.1 Patch Notes

## Weighted heterogeneous scheduling

The Experiment Manager now provides a deterministic weighted split mode with default requested shares:

- NVIDIA CUDA: 50%
- Intel XPU: 30%
- CPU: 20%

The shares are applied to jobs that genuinely contain accelerator-compatible CALO policy inference. Conventional algorithms and the AC Newton-Raphson evaluator remain CPU implementations and are never mislabeled as GPU work.

The software now displays both the requested split and the attainable total-job allocation. For example, a 20-algorithm × 5-run comparison contains 100 jobs but only five CALO jobs; therefore at most those five jobs can be split across CUDA/XPU/CPU without a GPU-native implementation of the remaining algorithms and evaluator.

## Execution behavior

- Device lanes are assigned before execution and recorded in result metadata.
- CUDA/XPU lanes are admitted before CPU on every cycle.
- Concurrent worker slots follow the configured weighted shares.
- Utilization and memory thresholds remain safety gates.
- Running jobs are not migrated between devices.
- The run queue shows the planned lane and the actual device used.

## Scientific limitation

This release improves routing but does not convert PYPOWER/NumPy power flow or the 19 conventional optimizers into GPU-native code. Consequently CPU utilization can remain substantial even for a CALO job assigned to CUDA/XPU because its power-flow evaluations are still performed on CPU.
