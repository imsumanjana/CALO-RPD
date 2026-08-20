# compute

**Purpose:** Resource admission, topology/device binding, execution contracts, persistent workers and accelerator kernels.

**Important state:** Leases, memory budgets, worker/device identity, telemetry and execution provenance.

**Major flow:** request -> Safe-80 admission -> CUDA or CPU lane -> worker/kernel -> attested result.

**Constraints/invariants:** CUDA-preferred/CPU-only; Intel XPU non-executable; at most 80% of free VRAM/available RAM.

**Common failure points:** OOM/resource races, identity mismatch, silent fallback, stale workers and CPU/CUDA parity drift.

## Primary files
- `calo_rpd_studio/compute/resource_scheduler.py`
- `calo_rpd_studio/accelerated/throughput_engine.py`
- `calo_rpd_studio/accelerated/device_resident_orpd.py`
- `calo_rpd_studio/accelerated/torch_power_flow.py`
- `calo_rpd_studio/compute/topology.py`
- `calo_rpd_studio/accelerated/torch_orpd.py`
- `calo_rpd_studio/accelerated/vram_residency.py`
- `calo_rpd_studio/compute/execution_contract.py`
- `calo_rpd_studio/compute/persistent_accelerator_worker.py`
- `calo_rpd_studio/compute/source_identity.py`
- `calo_rpd_studio/compute/governor.py`
- `calo_rpd_studio/compute/persistent_training_actor.py`

## Important public/entry symbols
- `CudaWindowTiming` — `calo_rpd_studio/accelerated/cuda_timing.py:18-32`
- `DeviceContext` — `calo_rpd_studio/accelerated/device.py:15-21`
- `DeviceResidentBatch` — `calo_rpd_studio/accelerated/device_resident_orpd.py:86-392`
- `DeviceResidentORPDEvaluator` — `calo_rpd_studio/accelerated/device_resident_orpd.py:395-1227`
- `ScratchPool` — `calo_rpd_studio/accelerated/scratch_pool.py:12-33`
- `StageTiming` — `calo_rpd_studio/accelerated/throughput_engine.py:37-44`
- `PerformanceLedger` — `calo_rpd_studio/accelerated/throughput_engine.py:47-76`
- `timed_stage` — `calo_rpd_studio/accelerated/throughput_engine.py:82-97`
- `DeviceCalibration` — `calo_rpd_studio/accelerated/throughput_engine.py:101-110`
- `ThroughputProfile` — `calo_rpd_studio/accelerated/throughput_engine.py:114-153`
- `_BatchRequest` — `calo_rpd_studio/accelerated/throughput_engine.py:157-162`
- `CrossRunBatchBroker` — `calo_rpd_studio/accelerated/throughput_engine.py:165-411`
- `TorchVariableDecoder` — `calo_rpd_studio/accelerated/torch_decoder.py:10-91`
- `ParityReport` — `calo_rpd_studio/accelerated/torch_orpd.py:60-75`
- `AcceleratedORPDProblem` — `calo_rpd_studio/accelerated/torch_orpd.py:78-780`
- `TorchPowerFlowOptions` — `calo_rpd_studio/accelerated/torch_power_flow.py:56-62`
- `TorchBranchResult` — `calo_rpd_studio/accelerated/torch_power_flow.py:66-70`
- `TorchPowerFlowResult` — `calo_rpd_studio/accelerated/torch_power_flow.py:74-92`

## Dependencies
- `calo-policy`, `core`, `experiments`, `persistence`, `power-system`

## Dependents
- `bootstrap`, `calo-policy`, `desktop`, `experiments`, `optimization`, `tests`, `validation-release`

## Associated tests
- `tests/gui/test_phase6_ribbon_workspace.py`
- `tests/scientific/test_v34_scientific_integrity.py`
- `tests/unit/test_accelerator_evidence.py`
- `tests/unit/test_cuda_residency_contract.py`
- `tests/unit/test_phase6_command_and_native_contracts.py`
- `tests/unit/test_prerequisites_and_resources.py`
- `tests/unit/test_source_identity.py`
- `tests/unit/test_tsh_calo_training_environment.py`
- `tests/unit/test_tsh_calo_training_resources.py`
- `tests/unit/test_v120_phase2_contracts.py`
- `tests/unit/test_v120_phase4_development_freeze.py`
- `tests/unit/test_v120_phase4_policy_retirement.py`
- `tests/unit/test_v120_phase5_release_preparation.py`
- `tests/unit/test_v31_batched_throughput.py`
- `tests/unit/test_v33_cuda_resident.py`

## Retrieval note
Use the machine indexes for exact locations and confidence-labelled relationships; authored invariants live in `../ARCHITECTURE.md` and `../DECISIONS.md`.
