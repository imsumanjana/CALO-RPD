# CALO-RPD Studio 5.8.0

**CALO-RPD Studio 5.8.0** is a Python/PyQt6 research platform for deterministic and robust optimal reactive power dispatch (ORPD), reproducible comparison of twenty optimizers, Cognitive Adaptive Learning Optimizer (CALO) research, policy training and qualification, independent validation, statistics, and publication evidence.

## v5.8 — Transactional Competitive Training and Scientific Closure

Version 5.8 focuses on the unresolved competitive-training findings identified in the v5.6/v5.7 deep audits and on the new/reopened v5.7 scientific-integrity issues. It preserves the central design rule that parallel PPO branches are **independent trajectories**: neural weights, optimizer states, RNG states, and curriculum states are never arithmetically averaged.

### Competitive training and exact resume

- **Two-phase branch-generation persistence:** exact-resume branches train only against session-staged checkpoints. A complete immutable branch generation is prepared first; the root branch manifest is the authoritative commit point. A failed later branch cannot overwrite only part of the previously coherent official generation.
- **Cumulative and Infinite modes:** session duration is separated from curriculum science. Infinite training no longer derives curriculum progression from the hidden/disabled cumulative-epoch value; explicit curriculum milestones are part of exact scientific state.
- **Bounded infinite state:** resume history, coordinator messages, and champion-decision history are bounded. Full epoch telemetry is written separately as append-only JSONL rather than growing the exact-resume payload indefinitely.
- **Safe Stop:** the exact session-start state is materialized before expensive validation. Rolling common safe checkpoints use a fixed 10-epoch cadence. Safe Stop is reported as `SAFE_STOPPED`/`SAFE_STOPPED_DEGRADED`, not as normal completion.
- **Stuck-worker protection:** Safe Stop has a grace deadline; nonresponsive branch processes can be terminated after the deadline while the coordinator retains the latest coherent common safe generation.
- **Crash recovery index:** interrupted competitive sessions retain a durable recovery record and staged exact states for explicit Recover/Discard handling. Recovery restores a coherent branch generation and does not promote an unfinalized branch champion.
- **Branch-aware accelerator admission:** CUDA/XPU slots are assigned per branch with configurable per-accelerator concurrency; excess branches fall back to CPU rather than silently oversubscribing the same accelerator.
- **Final queue accounting:** the coordinator performs a deterministic final queue drain and tracks terminal branch messages before result construction.

### Champion and Base Model science

- Branch candidates must pass validity/feasibility eligibility before promotion.
- Hardware-dependent inference latency is diagnostic only and is excluded from scientific policy-quality ranking.
- Champion evidence carries a comparator schema and validation-bundle fingerprint.
- Final Base selection re-evaluates the previous Base and all final branch candidates under one common validation bundle.
- Global selection uses a deterministic, order-independent feasibility-first scientific ranking with stable tie resolution; it is not a sequential majority-vote tournament.
- If no candidate is eligible, no candidate is falsely promoted as a Base. A terminal model may be retained only as an explicitly provisional artifact.

### Qualification, publication, continuation, and power flow

- **Formal superiority qualification is fail-closed:** saved-Base promotion requires the predeclared paired evidence size, favorable direction, win/effect gates, and Holm-adjusted significance. Non-inferiority is a distinct protocol with an explicit margin.
- **Publication evidence is complete-or-blocked:** when independent validation is required, publication-grade portfolio export requires the complete expected paired evidence set to be independently verified; partial verified subsets are not silently promoted to publication-ready evidence.
- **Scenario compatibility hashing is stronger:** `functools.partial`, bound methods, callable objects, closures/defaults, and scientific instance state are canonicalized so different physical transforms do not silently share an exact-resume fingerprint.
- **Legacy exact-resume migration:** a deliberate local-trust migration utility can convert verified legacy pre-HMAC exact-resume checkpoints into the authenticated v5.8 trusted-local format. Migration is never automatic.
- **Sparse Newton reference path:** when SciPy is available, the AC Newton Jacobian is constructed from sparse complex-voltage derivatives without densifying Ybus or allocating full dense angle/trigonometric matrices.

## Policy training modes

- **Cumulative:** run a fixed number of additional epochs for the current session.
- **Infinite:** run without a terminal epoch until Safe Stop; scientific curriculum milestones remain independent of the duration field.
- **Exact Resume:** restore model, optimizer, RNG, curriculum and branch state exactly from the last committed coherent branch generation.
- **Base-Guided Fork:** initialize fresh optimizer/RNG trajectories from deployable Base knowledge. This is scientifically distinct from Exact Resume.

## Scientific rules

CALO never receives hidden extra objective evaluations. Equal-FE comparisons use the same evaluator, formulation, constraints, scenarios, seeds and declared FE budget semantics. Robust feasibility defaults to all-scenario maximum constraint violation unless a different formulation is explicitly selected. Per-generator P/Q capability is enforced at individual online units, while voltage-deviation and L-index partitions are fixed by the declared formulation rather than changing candidate-by-candidate after Q-limit switching.

CALO-RPD 5.8.0 intentionally ships with **no automatically active/default neural policy**. Policy-assisted CALO is fail-closed: the user must train or import a compatible policy, qualify it as required, explicitly activate it, and bind its immutable SHA-256 to the experiment. No random, untrained, missing-policy, or legacy fallback is permitted.

## Run

```bash
python bootstrap.py
```

Windows and Linux launch helpers are included. The bootstrapper checks the local environment and installs/verifies supported prerequisites according to the packaged requirements.

## Legacy exact-resume migration

A pre-v5.7 trusted-local exact-resume checkpoint is **not automatically trusted**. After manually verifying its provenance, migration requires an explicit trust assertion:

```bash
python -m calo_rpd_studio.scripts.migrate_legacy_resume legacy.resume.pt --i-trust-this-local-file
```

The original file is retained; the migrated copy receives the current machine-local authenticated trust sidecar and migration provenance.

## Important remaining limitations

- CALO cognitive/control execution is not yet fully Torch/CUDA/XPU resident. The numerical ORPD evaluator is accelerator-capable, but portions of seeded CALO control remain host-side to preserve the frozen trajectory until a separately parity-qualified device-native implementation exists.
- Exact optimizer-state horizon continuation remains CALO-specific; other algorithms use paired recomputation from original seeds when extending horizons.
- Full target-environment release qualification still requires the complete PYPOWER, PyQt6 GUI, static-analysis, physical CUDA/XPU parity/fault-injection, and long-duration soak suites on the intended deployment hardware.
- Recovery of an interrupted first-ever competitive session can restore coherent exact branch state, but no unqualified/unfinalized branch model is promoted as a deployable Base.
- Large orchestration modules still contain maintainability debt. v5.8 strengthens service contracts and transactional behavior without claiming a complete structural rewrite.

See `RELEASE_VALIDATION.md`, `CALO-RPD-v5.8.0_IMPLEMENTATION_REPORT.md`, and `FINDINGS_CLOSURE_v5.8.0.csv` for release evidence and residual scope.
