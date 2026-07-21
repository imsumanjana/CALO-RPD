# CALO-RPD Studio v5.0.0 — Release Validation

## Release scope

CALO-RPD Studio v5.0.0 is the **Continuable Policy and Experiment Evolution Release**. It extends the validated v4.1 scientific/reproducibility foundation with resumable long-lived policy training, policy lineages/checkpoint provenance, experiment revisions, additional paired-run extension, FE-horizon evolution, exact CALO run-state continuation, paired recompute-from-seed extension for all algorithms, and horizon-aware evidence preservation/statistics/export.

This report validates the software release. It is **not** a claim that CALO, any bundled policy, or any optimizer is universally superior.

## Automated validation

The final source tree was tested in isolated partitions so GUI/Qt process lifetime does not contaminate non-GUI scientific tests.

| Partition | Result |
|---|---:|
| Unit + integration + regression | **177 passed** |
| Scientific integrity/solver tests | **20 passed** |
| GUI tests | **25 passed** |
| **Total** | **222 passed, 0 failed** |

Additional static checks:

- `ruff check .` — **PASS, zero findings**
- `compileall` for application/bootstrap/tests — **PASS**
- CALO v5 software freeze manifest — **PASS, 101 frozen files verified**
- Python wheel + source distribution build — **PASS**
- Isolated wheel import/version/module smoke test — **PASS**

## Continuation-specific scientific checks

Targeted v5 tests cover, among other cases:

- cumulative/additional/indefinite policy-training modes;
- exact policy-training resume with preserved optimizer/RNG state and deterministic equivalence to uninterrupted continuation on the controlled CPU test;
- immutable deployable checkpoints and lineage `latest` versus `best-qualified` roles;
- exact CALO run checkpoint restore before any fresh population evaluation;
- interrupted CALO same-planned-horizon resume reproducing the uninterrupted trajectory in the controlled deterministic test while consuming only the remaining requested evaluations;
- scientific problem-fingerprint binding for run checkpoints;
- revision-scoped checkpoint paths and explicit source-horizon selection;
- run-count extension preserving prior paired run indices/seeds;
- publication-eligible revision branching independently of later exploratory branches;
- all-paired/predeclared/manual-exploratory extension semantics;
- horizon-specific validation/evidence selection and deletion cleanup.

### Important trajectory distinction

An exact segmented continuation such as `5,000 FE -> 10,000 FE` is deliberately **not represented as equivalent** to a run configured for `10,000 FE` from FE=0. Adaptive schedules during the first segment saw the original horizon. For publication-safe from-start higher-horizon comparison across algorithms, v5 supports paired **recompute from the original seed** under the new horizon.

Exact optimizer-state continuation is currently implemented for **CALO**. Baseline algorithms use paired recompute-from-seed for scientifically fair larger-horizon comparisons and are never mislabeled as exact-resumed.

## Independent IEEE / PYPOWER cross-validation

The four supported IEEE cases were revalidated with the internal solver against the independent PYPOWER path.

| Case | Internal PF | Q-limit rounds | Max |ΔV| (p.u.) | Max |Δangle| (deg) | |Δloss| (MW) | Bus-type mismatch | Aggregate-Q mismatch |
|---|---|---:|---:|---:|---:|---:|---:|
| IEEE 30 | PASS | 0 | 6.661e-16 | 5.684e-14 | 5.906e-14 | 0 | 0 |
| IEEE 57 | PASS | 0 | 2.109e-15 | 8.527e-14 | 2.380e-13 | 0 | 0 |
| IEEE 118 | PASS | 1 | 8.882e-16 | 1.705e-13 | 8.356e-12 | 0 | 0 |
| IEEE 300 | PASS | 1 | 3.086e-14 | 3.126e-12 | 1.666e-11 | 0 | 0 |

Base-case losses observed in this validation were approximately 2.44380313, 27.86375151, 132.4807493, and 408.3256523 MW respectively. These are solver-agreement checks, **not optimization benchmark results**.

## Freeze verification

`calo_rpd_studio/data/frozen/calo_v500_freeze.json` was generated after the implementation stabilized and verified successfully:

- frozen files checked: **101**
- missing frozen files: **0**
- changed frozen files: **0**
- manifest integrity: **PASS**

The manifest explicitly records unresolved/deferred capabilities as `false`, including full device-resident CALO control, exact optimizer-state continuation for all baselines, and automatic periodic policy qualification.

## Scientifically important unresolved/deferred items

v5.0 intentionally does **not** claim the following are solved:

1. A single bit-identical transition implementation shared by the PPO training environment and complete runtime CALO.
2. Demonstrated ORPD superiority of the bundled/legacy policy; every candidate still requires Candidate-vs-Reference-vs-No-AI qualification.
3. Universally optimal AI/rule/online-credit authority weights.
4. Full Torch/CUDA/XPU-resident CALO cognitive/control execution; host-side compact NumPy/Python control paths remain in parts of CALO.
5. Exact optimizer-state continuation for every baseline optimizer.
6. Automatic asynchronous qualification/promotion at every configured policy-training interval; the interval is advisory in v5.0.

These items are tracked explicitly in `calo_rpd_studio/algorithms/calo/v5_disputes.py` and `FINDINGS_CLOSURE_v5.0.0.csv`.

## Hardware validation boundary

No claim is made here about real NVIDIA CUDA or Intel XPU utilization/throughput because this release-validation environment did not provide the user's physical accelerator hardware. Device-specific performance must be profiled on the target workstation before publication or performance claims.

## Release conclusion

The planned v5 continuation/provenance architecture is implemented and the tested software release passes the partitioned automated, static, freeze-integrity, and IEEE/PYPOWER scientific checks listed above. The release preserves explicit scientific boundaries between exact continuation, segmented continuation, paired recomputation, exploratory extensions, and publication-eligible evidence.
