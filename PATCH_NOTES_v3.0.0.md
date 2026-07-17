# CALO-RPD Studio v3.0.0 Patch Notes

## Accelerator-native common scientific backend

- Added double-precision PyTorch mixed-variable decoding.
- Added batched AC Newton–Raphson power flow for CPU, NVIDIA CUDA, and supported Intel XPU backends.
- Added accelerator-native branch-flow, active-loss, voltage-deviation, Kessel–Glavitsch L-index, generator-limit, voltage-limit, branch-limit, and robust aggregation calculations.
- Added candidate-specific Q-limit/PV-to-PQ handling and isolated fallback when batch bus sets diverge.
- Added CPU-reference reconstruction of final publication states.

## Twenty accelerator-compatible optimizers

- Added canonical FP64 tensor kernels for all nineteen non-CALO primary baselines.
- CALO retains its dedicated cognitive/AI method and now shares the same accelerator ORPD evaluator.
- All methods retain common Deb feasibility-first comparison, mixed-variable decoding, objective-evaluation accounting, scenario sets, seeds, and boundary handling.
- Baseline equations were not silently modified to manufacture stronger results.

## Scheduling and fairness

- Weighted CUDA/XPU/CPU shares now apply to the complete v3 optimizer plan.
- A 100% CUDA share can assign all jobs to CUDA when the PyTorch FP64 backend and a verified CUDA runtime are available.
- Added CPU/accelerator parity audit and an optional mandatory fairness gate.
- Added backend, dtype, device, batch size, and parity metadata to experiment provenance.

## Freeze and reproducibility

- Added `calo_v3_freeze.json` covering the accelerator evaluator, tensor decoder, baseline torch suite, CALO implementation, policy checkpoint, training snapshot, and shared constraint rules.
- Final TEST campaigns use the v3 freeze by default.

## Compatibility

- The trusted legacy CPU reference backend remains available.
- The v2.0.3 patch preserves existing databases and experiment records.
- Old experiments remain readable but do not retroactively become accelerator runs.
