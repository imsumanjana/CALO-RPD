# CALO-RPD Studio v6.0.0a4 — Alpha 1–4 Implementation Report

**Release name:** Policy-First Workflow and Safe-80 Compute Foundation  
**Date:** 2026-07-23  
**Baseline:** CALO-RPD v5.9.0 scientific-closure repository

## Purpose

v6.0.0a4 is an architecture release. It deliberately retains the scientifically validated native v5.9 CALO/policy semantics while changing how the application identifies workspaces, discovers compute hardware, establishes a protected resource envelope, and gates the RPD workflow behind an explicitly qualified active CALO governing policy.

No new optimizer-superiority claim is made by this release.

## v6.0-alpha1 — key-based workspace/navigation migration

Implemented a stable workspace registry with keys including `dashboard`, `calo_intelligence`, `power_system`, `orpd`, `algorithms`, `portfolio`, `scenarios`, `experiment`, and the downstream result/publication workspaces.

The new canonical order begins:

1. Dashboard
2. CALO Intelligence
3. Power System
4. ORPD Formulation
5. Algorithms
6. Portfolio Manager
7. Robust Scenarios
8. Experiment Manager

Scientific workflow logic, signal routing, restoration, navigation requests, and persisted UI state now use workspace keys as the authoritative identity. Numerical stack indexes remain only a presentation/compatibility detail.

A versioned legacy v5.9 index map preserves old positional workspace restoration. In particular, old v5.9 index 5 maps to CALO Intelligence and old index 1 maps to Power System rather than being reinterpreted under the reordered v6 stack.

The workflow snapshot schema now stores the governing-policy SHA. If the active governing policy changes, unfinished downstream setup is invalidated for reconfirmation rather than silently continuing under a different scientific controller.

## v6.0-alpha2 — Dashboard ComputeTopology and hardware mapping

Added a central compute-topology service used by application state and the Dashboard readiness scan.

The scan records:

- CPU name, physical cores, logical threads, system RAM and RAM pressure;
- CUDA devices and runtime IDs such as `cuda:0`;
- direct XPU devices such as `xpu:0` when available;
- configured XPU sidecar devices exposed by the existing resource monitor;
- best-effort Windows physical/OS adapter identity matched to runtime devices;
- device capabilities for PPO learning, policy actors, ORPD evaluation and full competitive-branch admission.

The Dashboard explicitly distinguishes physical/OS adapter identity from PyTorch runtime numbering. It therefore does not assume that Windows “GPU 1” must be CUDA device 1.

For safety, an XPU sidecar is visible as a mapped compute resource but is **not** counted as a validated full independent competitive-training branch in this alpha. That certification is reserved for later parity/capability work.

## v6.0-alpha3 — Safe-80 resource-budget engine

Added the default **Safe 80%** static protection profile.

The profile means CALO-RPD intentionally retains approximately 20% operating reserve; it does **not** cap instantaneous GPU compute utilization at 80%.

The engine calculates:

- one global protected CPU worker budget;
- an 80% system-RAM ceiling and remaining protected RAM branch capacity;
- per-device accelerator-memory admission where total device memory is measurable;
- one independent branch maximum per validated accelerator in alpha;
- a hard maximum safe simultaneous competitive-branch count.

The global CPU budget is shared across admitted branches. It is not multiplied independently for every branch.

Competitive training now receives and records:

- `safe_parallel_branches`;
- `safe_global_cpu_workers`;
- `compute_profile_fingerprint`.

The per-branch worker budget is derived from the single global budget.

Automatic competitive scheduling no longer silently spills excess accelerator demand onto heavy CPU branches. If validated accelerator hardware exists but has insufficient protected admission capacity, launch fails closed instead of silently moving work to CPU. Explicit CPU training remains possible when the user deliberately selects CPU and the Dashboard profile permits the requested concurrency.

At alpha scope, **total requested branches and simultaneous branches are still equivalent**. Requests above the calculated Safe-80 limit are blocked. Queuing total scientific branches separately from simultaneous concurrency is intentionally a v6 beta feature.

## v6.0-alpha4 — policy-first governing intelligence

CALO Intelligence is now directly after Dashboard and before Power System.

Power System is locked until the governing-policy readiness check proves that an active CALO policy is:

- present and usable;
- non-archived;
- runtime-ABI compatible;
- formally qualified (including accepted legacy-qualified status where explicitly supported);
- explicitly active;
- integrity-verified against its registered SHA-256.

No-AI CALO does not satisfy the normal governing-intelligence gate.

CALO Intelligence panel gating is tiered:

- **no policy record:** Training & Provisioning only;
- **any policy record exists:** policy inspection/qualification/intelligence blocks become available;
- **qualified active compatible integrity-verified policy:** CALO governing intelligence becomes READY and Power System unlocks.

The readiness decision is live and fail-closed. It is not a remembered workflow checkbox.

## Dashboard changes

The Dashboard now presents:

- System Protection status;
- Safe parallel-branch ceiling;
- CALO Governing Intelligence readiness;
- CPU topology and global worker budget;
- RAM and Safe-80 ceiling;
- accelerator branch slots;
- explicit physical/OS adapter ↔ runtime mapping table;
- validated device roles.

The existing power-system/experiment scientific context remains visible below the readiness layer.

## Native policy/algorithm semantics

This alpha does not alter the native v5.9 policy ABI:

- runtime architecture: `calo-v5.9`
- state schema: `calo-state-v5.9-32`
- action schema: `calo-action-v5.9-raw-global-4r-6o-6p`
- training environment: `calo-training-v5.9-exact-controller`

The v5.9 scientific-closure fixes remain the computational baseline.

## Verification performed in the build environment

- `compileall`: PASS for `calo_rpd_studio`, `calo_bootstrap`, and `tests`.
- v6 alpha architecture tests: 8 passed.
- CALO core v2 tests: 9 passed.
- competitive-training regression tests: 9 passed.
- v5.9 scientific-closure regression tests: 14 passed.
- heterogeneous policy-training + cache/broker hardening tests: 7 passed.
- Combined targeted regression selection: **47 passed, 0 failed**.
- Historical v5.9 evidence-retention test: 1 passed.
- v6 current freeze generated and independently verified: **127 files, 0 missing, 0 changed** before packaging.

Environment limitations:

- PyQt6 unavailable, so the complete GUI test suite could not be executed here.
- PYPOWER unavailable, so the complete external AC-PF scientific validation suite could not be executed here.
- Physical CUDA/XPU hardware unavailable; the build runtime contains CPU-only PyTorch.
- Ruff unavailable.

## Explicitly deferred beyond alpha4

The following are intentionally **not claimed complete** in this repository:

- global training-exclusive UI/application lock (beta1);
- total scientific branches separated from queued simultaneous concurrency (beta2);
- full first-class XPU-sidecar independent branch certification/scheduling (beta); 
- continuous thermal/power Green-Amber-Red governor;
- staged branch startup and dynamic admission;
- application-wide compute governor for later experiments/robust campaigns;
- target-laptop long-duration thermal soak and fault-injection qualification.

These boundaries are encoded in the current software freeze rather than hidden.
