# CALO-RPD Studio v5.8.0 — Release Validation

**Release:** 5.8.0 — *Transactional Competitive Training and Scientific Closure*  
**Validation date:** 22 July 2026

## Scope

v5.8 addresses the unresolved competitive multi-branch training findings from the v5.6 deep re-audit and the new/reopened v5.7 findings. The release focuses on transactional exact resume, bounded Infinite training, Safe Stop/recovery semantics, champion/Base scientific selection, branch resource admission, formal qualification evidence, verified publication completeness, callable scenario fingerprints, trusted legacy-resume migration, sparse Newton construction, and silent broad-exception suppression.

No optimizer or neural-policy superiority claim is made by this release validation.

## Release identity

- Application version: **5.8.0**
- Release name: **Transactional Competitive Training and Scientific Closure**
- Current software freeze: `calo_rpd_studio/data/frozen/calo_v580_freeze.json`
- Freeze scope: **110 scientific/runtime files**
- Freeze canonical manifest SHA-256: `0cfac9946fc73d7c329f31f8db313f42d6436b668a4cde770c77f7df4e0a68f1`
- Freeze file SHA-256: `bd2ce992d341c1a1ff409122bf791e6d7afc35b53e273669635fa74cd1165b59`
- Default/bundled neural policy: **none**
- Untrained/missing-policy fallback: **forbidden / fail-closed**

## Validation executed in this build runtime

### Syntax / import compilation

`python -m compileall` was run against:

- `calo_bootstrap`
- `calo_rpd_studio`
- `tests`

Result: **PASS**.

### Focused v5.8 and carried-forward regression suites

The following targeted suites passed in the audit/build runtime:

- Competitive multi-branch / exact-resume / Base-Guided Fork / immediate Safe Stop regressions: **9 passed**
- v5.8 new/reopened audit-closure + v5.7 closure integration/previous-audit regressions: **30 passed**
- Release-integrity / policy-gating / freeze checks: **4 passed**
- Policy cache/broker + native policy-system + numerical robustness regressions: **12 passed**
- Robustness + continuation regressions: **16 passed**

**Focused total: 71 passed, 0 failed.**

Key new regression coverage includes:

- prior exact branch generation remains immutable across continuation,
- immediate Safe Stop commits the exact session-start common state,
- Safe Stop status is distinct from normal completion,
- hardware latency cannot overturn scientifically better champion quality,
- global Base ranking is deterministic and candidate-order independent,
- validation-bundle fingerprints change with scientific evidence settings,
- Infinite-mode curriculum is independent of hidden session epoch duration,
- exact-resume history payload is bounded,
- `functools.partial` scientific parameters change compatibility identity,
- formal superiority fails closed without paired favorable/effect/Holm evidence,
- partial verified publication subsets are blocked,
- primary Newton Jacobian remains sparse when SciPy sparse is available,
- legacy trusted-resume migration requires explicit trust and authenticates the migrated copy,
- no broad `Exception`/`BaseException`/bare handler may silently consist only of `pass`, `continue`, or `return`.

## Freeze validation

`calo_v580_freeze.json` was regenerated after the v5.8 scientific/runtime changes and verified against the current tree:

- checked: **110 files**
- missing: **0**
- changed: **0**
- status: **PASS**

The freeze records the v5.8 contracts for transactional competitive branch generations, bounded Infinite resume state, typed Safe Stop, recovery index, order-independent common-bundle Base selection, validation fingerprints, branch-aware accelerator admission, complete verified publication evidence, formal superiority statistics, callable scientific fingerprints, and sparse Newton construction.

## Important target-environment validation not claimed here

The build/audit runtime did not provide the complete intended deployment environment. Therefore this release document does **not** claim execution of:

- the complete PyQt6 GUI interaction suite,
- the complete PYPOWER-dependent scientific suite in the final packaged environment,
- physical NVIDIA CUDA / Intel XPU throughput and fault-injection validation,
- long-duration multi-day Infinite-training soak tests on the target hardware,
- the complete static-analysis/Ruff release gate in the target CI environment.

Those gates remain required before a final publication campaign or unattended production-scale training run on the user's deployment hardware.

## Residual declared limitations

- CALO cognitive/control logic remains partly host-side; a fully device-resident rewrite is deferred until seeded scientific parity can be independently qualified.
- Large legacy orchestration/UI modules still carry structural maintainability debt despite stronger v5.8 service/transaction contracts.
- UI-only transient telemetry after the last committed scientific checkpoint remains non-authoritative and may be lost on a hard crash.
- Physical accelerator admission/utilization behavior must be benchmarked on the actual CUDA/XPU hardware before throughput claims are made.

## Release assessment

v5.8 is the intended corrected successor to v5.7 for the audited competitive-training and reopened scientific-integrity scope. It is materially safer for cumulative/exact-resume/Infinite branch training than v5.7, but final publication claims still require the complete target-environment validation gates and frozen paired independently validated campaign evidence.
