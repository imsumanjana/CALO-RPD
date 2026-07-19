# CALO-RPD Studio 4.0.0 — Release Validation Record

## Scope

Version 4.0.0 implements the tensor-native constraint-cognitive CALO upgrade defined in `docs/CALO_vNext_Tensor_Native_Scientific_Upgrade_Plan.pdf`. The release changes CALO search/memory architecture while preserving the common ORPD formulation, mixed-variable decoder, constraints, requested function-evaluation accounting, independent validation, and publication-evidence gates.

The scientific target is stronger feasible convergence without granting CALO an unfair budget or formulation advantage. The release does **not** claim universal superiority; final claims require frozen, paired, independently validated multi-run benchmark evidence.

## CALO v4 scientific corrections and additions

- Persistent learner/personal-best memory is retained through environmental selection and lineage branching.
- HPEM stores one canonical `[7,D]` feasible quality-diversity elite set and derives Best-1/3/5/7 `[4,D]` summaries without duplicate elite storage.
- HPEM duplicate/diversity logic follows mixed-variable decoder semantics for discrete tap/shunt lattices.
- Operator and memory-level credit are contextual 3D tensors and are updated once per batch using each learner's effective regime/context, eliminating per-candidate update-order bias.
- Success-direction memory uses bounded `[operator, history, variable]` storage with finite NaN/Inf-safe reward normalization.
- Generator-voltage, transformer-tap, and shunt groups receive separate regime-aware productivity learning.
- `recovery_fraction` now controls bounded weak-learner recovery proposals while useful knowledge is retained.
- Epsilon handling is behavior-driven by feasibility/progress/stagnation signals.
- Discovery and learned search operate as dual lanes within one independent run and one common requested FE budget; runtime memory is reset between repeated runs.
- Cognitive precision proposals are counted inside the common FE budget and use no hidden local optimizer or uncounted solver evaluations.
- Exact decoded duplicate controls may reuse a physical solve, while every requested candidate still consumes one FE budget unit.
- CALO uses hybrid 1D/2D/3D state and temporary 3D broadcast/scratch calculations; no persistent 4D CALO architecture was introduced.
- Strict benchmark mode rejects historical priors/warm starts. Historical transfer learning remains an explicit separate study mode.

## Test evidence

Final repository collection: **200 tests**.

Validation was executed in isolated partitions to avoid cumulative GUI/training resource interference:

- Unit tests: **146 passed, 0 failed**.
- Scientific tests: **20 passed, 0 failed**.
- GUI + integration + regression tests: **34 passed, 0 failed** in Qt offscreen mode.
- Total: **200 passed, 0 failed**.
- Ruff: **passed with zero findings**.
- `compileall`: **passed**.
- CALO v4 frozen manifest: **85 files verified**.

A queued Matplotlib/Qt deleted-canvas teardown race observed during testing was corrected with a safe canvas draw guard; the final GUI/integration/regression partition completed cleanly.

## IEEE base-case and independent PYPOWER validation

All four supported benchmark cases loaded, converged, and independently cross-validated:

| Case | Internal PF | Q-limit rounds | Max |ΔV| p.u. | Max |Δangle| deg | |Δloss| MW | Bus-type mismatches | Aggregate-Q mismatches |
|---|---|---:|---:|---:|---:|---:|---:|
| IEEE 30 | PASS | 0 | 6.66e-16 | 5.68e-14 | 5.91e-14 | 0 | 0 |
| IEEE 57 | PASS | 0 | 2.11e-15 | 8.53e-14 | 2.38e-13 | 0 | 0 |
| IEEE 118 | PASS | 1 | 8.88e-16 | 1.71e-13 | 8.36e-12 | 0 | 0 |
| IEEE 300 | PASS | 1 | 3.09e-14 | 3.13e-12 | 1.67e-11 | 0 | 0 |

These values are numerical agreement checks, not optimization-performance claims.

## Budget and independence checks

Targeted v4 tests confirm:

- CALO never exceeds the configured requested FE budget in the single-budget self-learning run.
- Repeated runs with the same seed reproduce the same result in the tested deterministic path.
- HPEM, pbest, success memory, contextual credit, and other runtime learning state start fresh for each independent optimizer instance.
- Strict benchmark mode blocks historical parameter priors/warm starts.
- Exact cache hits reduce physical duplicate work only; requested FE accounting is unchanged.

## Lightweight development diagnostics — not publication evidence

A 12-D synthetic Sphere diagnostic was used only to test whether the new mechanisms produce a measurable optimization signal before expensive ORPD campaigns.

Twelve paired seeds, population 16, 160 FE, AI disabled:

| Variant | Median final objective | Mean | Best |
|---|---:|---:|---:|
| Complete CALO v4 | 0.20385 | 0.20752 | 0.09091 |
| No HPEM | 0.23095 | 0.24138 | 0.13465 |
| No contextual credit | 0.21069 | 0.21603 | 0.06519 |
| No dual lane | 0.20955 | 0.22790 | 0.14584 |
| No precision | 0.20248 | 0.21825 | 0.09318 |
| Foundational-only | 0.27618 | 0.28914 | 0.16364 |

Interpretation: the combined architecture materially improved over the foundational-only variant, with useful signal from HPEM/context/dual-lane mechanisms. Precision did not improve the median in this small synthetic diagnostic, so precision superiority is **not** claimed and must be tuned/ablated only on declared development cases.

A separate 20-seed, 320-FE synthetic diagnostic gave CALO v4 a median of ~0.10338 versus TLBO ~0.04144. CALO beat several other tested baselines but did not beat TLBO on that problem. This explicitly prevents a scientifically unsupported “CALO is already best” claim.

## Packaging and hardware notes

- Wheel and source distribution build successfully as version 4.0.0.
- Installed-wheel import smoke test passed using the final package artifact and host scientific dependencies.
- Physical NVIDIA CUDA and Intel XPU hardware were not available in the build environment. The common FP64 numerical backend remains CUDA-first, then XPU, then CPU fallback, but physical utilization/throughput must be measured on the target workstation.
- CALO control/memory state is compact contiguous host tensor/array state in v4; common ORPD population evaluation remains accelerator-capable/device-resident where supported. Full CALO-control device migration is intentionally gated behind future CPU/accelerator parity evidence rather than forced at the expense of reproducibility.

## Release gate conclusion

**PASS for CALO v4.0.0 architecture/research release.**

This means the implementation, regression suite, scientific base-case validation, fairness guards, packaging, and freeze integrity passed. It does **not** mean CALO is publication-proven superior to all competitors. That claim remains gated by development-only tuning on IEEE 30/57, cryptographic freeze, blind IEEE 118/300 holdout evaluation, and 30–50 paired independently validated final runs.
