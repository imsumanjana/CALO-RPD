# CALO-RPD Studio 3.3.0 — Release Validation Record

Version 3.3.0 introduces the CUDA-Resident Execution Engine for comparative evaluation and CALO policy-training ORPD rollouts. The release changes device execution and throughput plumbing while preserving common FP64 scientific formulations, equal evaluation accounting, seed management, and independent CPU-reference validation.

## Implemented release gates

- Default comparison allocation: CUDA 80%, Intel XPU 10%, CPU 10%.
- Optional 100% CUDA job allocation when CUDA is available.
- All twenty primary optimizers and CALO ablation variants are accelerator-eligible under the PyTorch FP64 backend.
- Device-resident mixed-variable decoding, scenario expansion, batched Newton-Raphson power flow, branch flows, objective/constraint aggregation, L-index and robust aggregation.
- Grouped tensor PV-to-PQ switching without candidate-by-candidate Python power-flow fallback.
- Persistent CUDA/XPU/CPU services, cross-run tensor batching, automatic microbatch calibration, and CUDA-priority work stealing for unstarted jobs.
- Policy-training rollout defaults changed to 80/10/10 with a 100% CUDA preset; ORPD development rollouts use the same device-resident evaluator.
- CPU/accelerator parity gate and independent CPU-reference final validation retained.

## Automated verification

- Pytest: **104 passed, 25 skipped**.
- Skipped tests: 22 PyQt6 GUI tests because PyQt6 was unavailable in the packaging environment; 3 IEEE/PYPOWER scientific tests because PYPOWER was unavailable.
- Python source compilation: passed.
- Device-resident ORPD parity on the available deterministic toy case: passed.
- All-primary 400-job allocation regression:
  - 80/10/10 -> 320 CUDA, 40 XPU, 40 CPU.
  - 100/0/0 -> 400 CUDA, 0 XPU, 0 CPU.
- Tensor-native baseline smoke tests: passed.
- Wheel build: passed.
- Source distribution build: passed.
- Clean wheel import: reported version 3.3.0.
- Frozen v3.3 verification: passed across **50 files**.

## Environment limitations

The packaging environment had a CPU-only PyTorch runtime and no physical NVIDIA CUDA or Intel XPU device. Physical accelerator throughput, utilization, device-memory behavior, and CUDA/XPU parity must therefore be verified by the prerequisite wizard and parity audit on the target workstation. Ruff was not installed in the packaging environment and is not claimed as executed.

## Scientific boundary

The numerical hot path is substantially more device-resident, but CPU use is not eliminated. Mandatory host responsibilities remain: GUI and process orchestration, sparse telemetry, one packed population result materialisation for the stable public result/provenance contract, SQLite/file persistence, checkpointing, portfolio generation, and final independent CPU-reference validation. The 80/10/10 or 100% CUDA setting refers to optimizer-job numerical assignment, not a guarantee of an identical Task Manager utilization percentage.
