# CALO-RPD Studio v3.3.0 Patch Notes

## CUDA-Resident Execution Engine

Version 3.3.0 replaces the previous host-orchestrated accelerator path with a substantially more device-resident scientific pipeline for both comparative evaluation and CALO policy-training ORPD rollouts.

### Default execution policy

- New recommended scheduler: **CUDA-resident priority — 80% CUDA / 10% Intel XPU / 10% CPU**.
- New **CUDA-only resident — 100% CUDA** mode when a verified NVIDIA CUDA backend is available.
- Custom weighted and measured-throughput modes remain available.
- CUDA-priority work stealing may redirect only unstarted XPU/CPU jobs to idle CUDA capacity. Running jobs are never migrated.

### Reduced host/device round trips

- Candidate tensors are passed directly from tensor-native optimizer kernels to the common ORPD evaluator.
- Mixed continuous/discrete decoding, scenario expansion, Y-bus assembly, FP64 Newton-Raphson solving, branch flows, active loss, voltage deviation, L-index, constraint components and robust aggregation remain on the assigned device.
- PV-to-PQ switching is grouped by candidate bus-type mask instead of reverting to one Python power-flow solve per candidate.
- Cross-run batching concatenates compatible tensor requests on-device.
- One packed population result is materialized on the host for the stable Evaluation/provenance interface; final independent validation remains CPU-reference based.

### All primary algorithms

- All twenty primary algorithms and CALO ablation variants are eligible for CUDA/XPU allocation under the PyTorch FP64 scientific backend.
- The legacy CPU-reference evaluator remains deliberately CPU-only.
- Canonical algorithm identities, seeds, evaluation budgets, constraints and mixed-variable definitions are unchanged.

### Policy training

- Heterogeneous rollout defaults are now **80/10/10**.
- Added one-click **80/10/10** and **100% CUDA** rollout presets.
- ORPD development-case rollouts use the same device-resident FP64 evaluator.
- The central PPO learner remains accelerator-first; synthetic curriculum environments remain lightweight host-side tasks.

### Scientific safeguards

- CPU/accelerator parity remains mandatory for final benchmark campaigns.
- A new v3.3 cryptographic freeze manifest includes the device-resident evaluator, throughput engine, scheduler, tensor algorithms and policy-training runtime.
- CPU remains responsible for mandatory GUI/process orchestration, sparse telemetry, persistence, checkpoints, portfolio generation and independent reference validation.
