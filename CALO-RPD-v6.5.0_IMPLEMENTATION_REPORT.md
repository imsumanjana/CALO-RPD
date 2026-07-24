# CALO-RPD v6.5.0 Implementation Report

## Release identity

- Version: **6.5.0**
- Release name: **Must-Resolve Audit Closure**
- Date: **24 July 2026**
- Baseline: CALO-RPD v6.4.0 Stage-B device-resident policy-training release

## Purpose

v6.5 closes every item classified as **Must Resolve** in the post-v6.4 priority resolution list. The release is intentionally focused on scientific parity, numerical edge cases, checkpoint integrity, deterministic shutdown, stale-GUI configuration prevention, result-browser robustness, and holdout protection. Performance/maintainability items classified only as **Better to Resolve** are not relabelled as completed.

## 1. Scientific and numerical correctness

### C04 — Torch/CUDA Newton–Raphson damping/backtracking parity

The Torch single-case and batched Newton–Raphson solvers now use CPU-reference-style damping/backtracking instead of accepting every full Newton step. Trial damping begins at `1.0`, reduces geometrically, accepts the first improving finite trial, and retains the best finite trial when strict improvement is unavailable. `minimum_damping` is propagated through accelerated ORPD configuration.

This closes the stressed-case backend divergence mechanism identified in v6.4 while preserving FP64 accelerator execution.

### C10 — bounded stepped discrete values

`stepped_values()` now validates finite bounds and a positive finite step, rejects inverted bounds, constructs the lattice using a floor-based count with floating-point tolerance, and filters any numerical endpoint above the declared upper bound.

### H01 — stable voltage-limit normalization

Constraint normalization no longer divides by an effectively zero voltage span. A zero/near-zero declared span uses a stable engineering-scale denominator rather than an artificial `1e-12` divisor that could inflate tiny violations by orders of magnitude.

### H05 — one zero-impedance rule across Torch paths

Single-case, batched, and device-resident Torch admittance construction now share the same active-branch zero/near-zero impedance threshold. Inactive padded branches use safe division without weakening the active-network validity rule.

### H15 — stable near-zero policy qualification

Relative policy-comparison evidence now uses a symmetric stable scale `max(|candidate|, |comparator|, 1.0)`. Near-zero comparator objectives therefore cannot create artificial enormous percentage differences or qualification evidence.

## 2. Checkpoint concurrency and integrity

### H21 / H22 — transactional delete and qualification updates

Policy-checkpoint delete eligibility checks, metadata reads/merges, and mutations now occur inside `BEGIN IMMEDIATE` database transactions. This removes the read-before-lock TOCTOU window that could otherwise permit stale decisions or lost metadata updates.

### M30 — monotonic latest-checkpoint lineage

Checkpoint registration now chooses `latest` using cumulative epoch and phase ordering inside the same transaction. Late registration of an older checkpoint cannot displace a newer authoritative latest checkpoint.

### M32 — atomic authenticated exact-resume publication

Exact-resume checkpoints use a **self-authenticating v2 envelope** containing both the serialized checkpoint payload and signed trust metadata in one atomically replaced file. Resume verification validates the internal digest/HMAC before pickle-capable loading. The external legacy trust sidecar is retained only for compatibility and is no longer the trust-critical atomicity boundary.

### M45 — streaming checkpoint hashing

Checkpoint SHA-256 calculation now streams fixed-size chunks rather than reading entire checkpoint files into RAM.

## 3. Runtime lifecycle and GUI correctness

### M18 / V64-N01 — deterministic broker shutdown

Both the policy inference broker and the Stage-B synthetic cross-episode broker now maintain explicit lifecycle state and pending-request registries. `close()` atomically prevents new submissions and fails all pending/in-flight waiters deterministically, avoiding waits until timeout or indefinite blocking during shutdown races.

### H28 — apply current Comparison Study GUI values before execution

`start_comparison()` now applies and validates the current GUI configuration before fairness gating and plan construction. A comparison can no longer silently start from stale previously-applied settings when the visible form has changed.

### H30 — robust Results Explorer parsing

Results Explorer now uses guarded JSON-object parsing, safe missing-field access, explicit unavailable-seed handling, and `row.get("run_id") or row["id"]` fallback semantics. Corrupt/incomplete rows no longer fail refresh merely because JSON or a nullable run ID is malformed.

## 4. Stage-B evidence and holdout protection

### V64-N02 — equal-length parity is mandatory

Stage-B reference-vs-accelerated parity verification now rejects unequal result lengths before element-wise comparison. A truncated accelerator result can no longer pass by matching only a reference prefix.

### V64-N03 — canonical holdout identity protection

Training and normal GUI development-suite selection no longer rely only on filenames such as `case118` or `case300`. A canonical case-identity layer compares standard-case scientific checksums and, when canonical dependencies are unavailable, conservatively protects equivalent 118-/300-bus cases. Renaming a protected case file does not make it admissible for development training.

## 5. Release freeze additions

The v6.5 freeze includes the new canonical case-identity module and explicitly records closure properties for:

- Torch Newton backtracking parity;
- bounded discrete stepped values;
- stable zero-span constraint normalization;
- consistent zero-impedance thresholds;
- stable near-zero qualification evidence;
- transactional checkpoint mutations;
- monotonic latest lineage;
- self-authenticating atomic exact-resume envelopes;
- streaming checkpoint hashing;
- deterministic broker shutdown failure propagation;
- current-GUI comparison configuration application;
- corrupt-JSON-safe Results Explorer behavior;
- equal-length Stage-B parity gating;
- canonical holdout identity protection.

## 6. Validation executed in the build runtime

- Focused v6.5 must-resolve test suite: **16 passed**.
- Combined must-resolve + Stage-B + broker + continuation + accelerated power-flow regressions: **57 passed**.
- Python `compileall`: **PASS** after implementation.
- PyTorch runtime: CPU-only (`2.10.0+cpu`); physical CUDA/XPU unavailable.
- PyQt6: unavailable in the build runtime.
- PYPOWER: unavailable in the build runtime.

Physical CUDA/XPU throughput, target-Windows GUI interaction, and PYPOWER-backed final validation remain target-environment gates and are not fabricated as build-runtime evidence.

## Final implementation classification

All findings classified **Must Resolve** in the supplied v6.4 priority list: **IMPLEMENTED / CLOSED IN v6.5 SOURCE AND TARGETED REGRESSION COVERAGE**.

Items classified **Better to Resolve**: **DEFERRED / NOT CLAIMED AS CLOSED BY v6.5**.
