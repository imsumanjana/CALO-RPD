# CALO v4 Implementation Report

## Scope

CALO-RPD Studio 4.0.0 implements the architecture in `CALO_vNext_Tensor_Native_Scientific_Upgrade_Plan.pdf` while preserving the common ORPD evaluator, mixed-variable decoder, constraint formulation, paired-seed protocol, independent validation, and function-evaluation accounting used by competing algorithms. CALO v4 does **not** claim universal superiority; that requires frozen 30–50 run paired feasible evidence on the declared benchmark suite.

## Implemented mechanisms

- **Persistent personal/lineage memory:** personal bests survive environmental selection; child/parent branching preserves unambiguous learner lineage.
- **HPEM:** one canonical feasible elite store `E[7,D]` derives four Best-1/3/5/7 summaries `H[4,D]`. Mixed-variable distance maps discrete tap/shunt controls to decoded lattice indices before duplicate/diversity decisions.
- **Contextual batch credit:** operator credit `[R,O,C]` and memory-depth credit `[R,4,C]` are updated once per batch and attributed to each learner's effective regime/context, avoiding per-candidate decay ordering bias.
- **Contextual success memory:** bounded successful directions use compact `[O,H,D]` storage with finite objective/feasibility reward normalization and context-aware sampling.
- **Variable-group intelligence:** `[R,G,K]` statistics separately learn generator-voltage, transformer-tap, and shunt productivity.
- **Behavior-driven epsilon:** feasibility ratio, violation progress, stagnation, and budget progress modulate epsilon rather than relying only on a fixed iteration schedule.
- **Single-budget dual-lane learning:** discovery and learned candidates coexist inside one independent run and one configured FE budget. Memory readiness, diversity, stagnation, progress, and HPEM consensus control lane allocation.
- **Cognitive precision:** counted candidates use Best-1/3/5 information, successful directions, group focus, adaptive radius, and legal mixed-variable moves. No hidden local optimizer or uncounted objective evaluations are used.
- **Recovery without forgetting:** `recovery_fraction` now controls a bounded subset of weak learners that receive recovery proposals while elite, personal, and success knowledge is retained.
- **Exact evaluation reuse:** exact decoded duplicate controls may reuse a physical solve, but every optimizer-requested candidate still consumes one FE budget unit.
- **Hybrid-dimensional state:** 1D/2D/3D structures are used according to semantics; no persistent 4D CALO state is introduced. Temporary `[P,4,D]` memory relationships use reusable scratch buffers.
- **Strict benchmark guard:** runtime historical priors/warm starts are rejected in strict benchmark mode. Explicit historical transfer-learning mode disables that guard visibly and remains separate from locked TEST campaigns.

## Key tensor shapes

| State | Shape | Role |
|---|---:|---|
| Objective / violation / flags | `[P]` | Candidate scalar state |
| Population / personal best | `[P,D]` | Core learner state |
| Canonical HPEM | `[7,D]` | Single-source elite memory |
| HPEM summaries | `[4,D]` | Best-1/3/5/7 knowledge |
| Success directions | `[O,H,D]` | Bounded contextual success memory |
| Contextual operator credit | `[R,O,C]` | Regime/operator/context learning |
| Variable-group statistics | `[R,G,K]` | Regime/group productivity |
| Temporary memory directions | `[P,4,D]` | Reused batch scratch only |

## Validation performed

- Unit tests: 144 passed before the final two v4-specific tests were added; the complete final count is recorded in `RELEASE_VALIDATION.md` after final validation.
- Scientific tests: 20 passed, including IEEE 30/57/118/300 internal and independent validation coverage.
- GUI/integration/regression partition: 34 passed in offscreen Qt mode.
- Ruff and `compileall`: passed after final source cleanup.
- Torch-FP64 CALO smoke on case30 completed with the exact requested FE budget and correct v4 metadata/tensor shapes.

## Lightweight development diagnostics

These are engineering diagnostics, **not publication evidence**.

On a 12-D synthetic Sphere problem, 12 paired seeds, population 16, 160 FE, AI disabled:

- complete CALO v4 median: 0.20385;
- no HPEM median: 0.23095;
- no contextual credit median: 0.21069;
- no dual lane median: 0.20955;
- no precision median: 0.20248;
- foundational-only variant median: 0.27618.

The combined architecture materially improved over the foundational-only variant, and HPEM/context/dual-lane each showed useful signal. Precision improved some distribution measures but did not lower the median in this small diagnostic, so its superiority is **not** claimed and it remains subject to ORPD development ablation.

In a separate 20-seed, 320-FE synthetic diagnostic, CALO v4 outperformed MTLA-DE, QODE, PSO, and CLPSO by median objective but did not outperform TLBO. This is explicit evidence that v4 architecture alone does not justify a claim that CALO is already universally best.

## Scientific limitations and next gates

1. Final CALO hyperparameters and policy authority require development-only ablation on IEEE 30/57, followed by a cryptographic freeze before IEEE 118/300 holdout evaluation.
2. Publication superiority requires 30–50 paired independent runs, feasible-only statistics, equal requested FE budgets, independent validation, effect sizes, Wilcoxon/Holm, and multi-case ranking evidence.
3. Physical NVIDIA CUDA and Intel XPU throughput was not available in the build environment. The common FP64 accelerator evaluator remains CUDA/XPU capable, but physical utilization must be measured on target hardware.
4. CALO control/memory tensors are compact contiguous NumPy state in v4; common ORPD evaluation can remain device-resident. Full device migration of CALO control logic should occur only behind CPU/accelerator parity tests, not by sacrificing reproducibility.
5. Adaptive active-population reduction is intentionally not enabled by default in v4 because it needs a separate ablation to prove benefit without changing effective search pressure unfairly.
