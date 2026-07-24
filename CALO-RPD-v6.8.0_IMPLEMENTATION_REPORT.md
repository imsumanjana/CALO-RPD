# CALO-RPD v6.8.0 Implementation Report

## Release

**Version:** 6.8.0  
**Release name:** Independent CALO Intelligence & XPU Recovery

## Purpose

v6.8.0 resolves two runtime/workflow defects observed on the target Windows laptop after v6.7:

1. CALO Intelligence policy training was incorrectly inheriting Comparison/Portfolio execution constraints. A one-run policy-development `ExperimentConfig` could therefore be rejected by the portfolio minimum-run rule (for example, `runs=1` versus a publication portfolio minimum of 30).
2. The Intel XPU could disappear from System Readiness when CUDA was healthy but the isolated XPU runtime was missing, stale in bootstrap state, or hidden from `Win32_VideoController` on a hybrid-graphics laptop.

## Implemented corrections

### 1. CALO Intelligence is an independent workflow

- Added `ExperimentConfig.validate_policy_development()`.
- Policy-development validation now checks only the scientific formulation used by CALO Intelligence: objective, controls/variables, power-flow options, constraint tolerances, robust objective and scenarios.
- It deliberately does **not** inherit Comparison/Portfolio run minima, benchmark repetition requirements, campaign budget rules, execution-lane shares, or other tab-specific execution constraints.
- All real-ORPD policy-development paths now use this independent scientific validation.
- Removed automatic `config_changed -> load_from_config()` cross-tab rehydration from CALO Intelligence.
- Policy qualification now uses CALO Intelligence's own scientific template and its own training seed control, not mutable Experiment Manager/Comparison Study state.
- Experiment binding remains explicit through the existing Apply Policy action.

### 2. Mixed NVIDIA + Intel systems are repaired per accelerator

- Bootstrap now checks CUDA and XPU readiness independently.
- A healthy NVIDIA CUDA backend can no longer hide a missing Intel XPU runtime behind a single aggregate `gpu_ready=True` flag.
- On a mixed system, missing/unverified XPU now triggers prerequisite repair even when CUDA is already ready.

### 3. XPU sidecar discovery self-heals stale bootstrap state

- `configured_xpu_interpreter()` no longer trusts only serialized `environment_state.json` readiness flags.
- It checks an explicit `CALO_XPU_PYTHON` override, the recorded interpreter, and the canonical `~/.calo_rpd_studio/xpu_runtime` interpreter.
- Every candidate is live-probed before use.
- `ResourceMonitor` can rediscover a repaired XPU sidecar without requiring the application process to be restarted.

### 4. Intel GPU hardware identity detection is stronger

- Windows detection still uses `Win32_VideoController` first.
- It now falls back to the PnP Display class and stable Intel PCI vendor tag `VEN_8086` when hybrid graphics hides the Intel adapter from `Win32_VideoController`.
- System topology also merges PnP display adapters into the physical adapter map.

### 5. System Readiness no longer silently hides Intel hardware

When a physical Intel GPU is present but no verified XPU runtime exists, the dashboard now keeps a non-schedulable row visible with status similar to:

`XPU hardware detected — runtime unavailable`

This row is diagnostic only and is never admitted as a compute device until `xpu:0` passes the scientific runtime probe.

## Expected target-machine behavior

After running v6.8 through `bootstrap.py`, a mixed RTX 4060 + Intel integrated graphics system should resolve to:

- NVIDIA GeForce RTX 4060 Laptop GPU -> `cuda:0`
- Intel GPU -> isolated verified `xpu:0` sidecar when the Intel PyTorch XPU runtime is supported and passes the FP64 probe
- CPU -> `cpu`

If Intel hardware is detected but XPU provisioning fails, System Readiness should show the Intel adapter explicitly as unavailable rather than omitting it.

## Validation performed in the build environment

- Python `compileall`: **PASS**
- v6.8 focused CALO-independence/XPU tests: **5 passed**
- Combined focused + v6.7 hardware + prerequisites/resources + heterogeneous training + configuration selection: **36 passed**

The build environment does not provide the user's physical Intel XPU, therefore actual `xpu:0` execution remains a target-machine verification step. The user's separately supplied diagnostics already confirm the NVIDIA RTX 4060 CUDA/NVML stack, but that external evidence is not represented as an independently executed v6.8 build test.
