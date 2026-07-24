# CALO-RPD v6.7.0 — Implementation Report

**Release name:** Hardware Runtime Binding & Telemetry Integrity  
**Base:** CALO-RPD v6.6.0

## Scope

v6.7.0 closes the runtime/hardware-integration defects exposed by target-machine diagnostics on an NVIDIA GeForce RTX 4060 Laptop GPU and by the subsequent CPU/CUDA/XPU mapping audit. The scientific formulation, v6.5 must-resolve closure, and v6.6 remaining-audit closure are preserved.

## Implemented fixes

| ID | Resolution |
|---|---|
| V67-R01 | Added `nvidia-ml-py>=13,<14` to project/bootstrap dependencies so `torch.cuda.utilization()` has its declared NVML Python dependency on fresh installs. |
| V67-R02 | CUDA compute discovery is now independent from optional utilization/memory telemetry. NVML import/telemetry failure cannot clear an otherwise valid CUDA device snapshot. Optional telemetry warnings are rate-limited. |
| V67-R03 | `nvidia-smi` telemetry is associated with CUDA runtime devices by UUID/PCI identity when available; name and single-device fallbacks are conservative. CUDA runtime index is no longer assumed to equal `nvidia-smi`/Windows enumeration order. |
| V67-R04 | Added one canonical device-binding function used by the primary experiment manager, persistent accelerator worker, persistent XPU sidecar, and one-shot XPU worker. `runtime_compute_device`, optimizer execution device, torch backend, and CALO inference device are bound consistently. |
| V67-R05 | XPU sidecar telemetry now propagates total memory and hardware identity metadata and performs an explicit FP64 tensor/matmul smoke before ORPD evaluator capability is accepted. |
| V67-R06 | Every completed run now records planned-vs-actual device attestation: runtime probe, evaluator device/name, optimizer/control-plane device, CALO policy device, and a binding-consistency flag. |
| V67-R07 | Windows adapter labels no longer imply that CIM enumeration order is the same as Task Manager GPU numbering; stable PNP/runtime identities remain authoritative. |

## Hardware mapping semantics

- NVIDIA runtime identifiers are PyTorch CUDA identifiers such as `cuda:0`.
- Intel GPU runtime identifiers are PyTorch XPU identifiers such as `xpu:0`, including a separate sidecar interpreter when CUDA and XPU require different PyTorch builds.
- CPU runtime identifier is `cpu`.
- Windows “GPU 0/GPU 1” labels are presentation labels and are not used as the scheduling identity.

## Validation performed in the build environment

- Python compileall: PASS.
- v6.7 focused hardware/runtime tests: PASS.
- v6.5 must-resolve + v6.6 remaining-audit regression selection: PASS.
- Resource/prerequisite + Safe-80 compute tests: PASS.

The build environment does not expose a physical CUDA/XPU device, PyQt6, or PYPOWER. Therefore physical accelerator throughput/thermal/GUI/full external power-flow validation is not claimed by this build. User-provided target diagnostics separately confirmed CUDA 13.0 / PyTorch CUDA availability and successful NVML installation for an RTX 4060 Laptop GPU; those diagnostics are not substituted for the automated release gate.
