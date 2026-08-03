# CALO-RPD modernization implementation gates

This file controls implementation of the remediation plan in
`docs/COMPLETE_REPOSITORY_CONTAINERIZATION_SCIENTIFIC_AUDIT_2026-08-03.md`.
The live requirement-to-evidence audit is
[`REQUIREMENT_TRACEABILITY.md`](REQUIREMENT_TRACEABILITY.md); it distinguishes local tests from
physical/external and scientific-campaign proof.

## Baseline

- Starting commit: `307402df5c7a44a6bb852770347b1b1ef995548d`.
- Starting branch: `main`; implementation branch: `codex/calo-complete-modernization`.
- The starting checkout already contained uncommitted v6.9 work. Those changes are treated as
  pre-existing work and must not be discarded or silently attributed to this modernization.
- The v6.9 freeze and `MANIFEST.sha256` describe a historical candidate snapshot. They must fail
  after a frozen file changes; they are regenerated only for a new verified release artifact.
- The requested audit Markdown is an analysis artifact, not proof that any implementation gate has
  passed.

## Change classes

### Class A — behavior-preserving engineering

May proceed without algorithm approval:

- tests and release-harness scoping;
- schema/code contract alignment where runtime behavior is already strict;
- container/build/CI infrastructure;
- XPU compatibility readers and removal of executable XPU paths;
- GUI terminology and presentation changes;
- configuration decomposition that preserves scientific fingerprints and values;
- memory admission, telemetry, resource leases, staging and explicitly governed backend fallback;
- policy lifecycle orchestration and immutable experiment binding;
- diagnostics and documentation that describe existing behavior truthfully.

Every Class A change still requires focused tests and compatibility evidence.

### Class B — scientific behavior or promotion semantics

Requires explicit user approval before implementation:

- operator allocation or operator equations;
- policy state/action space, architecture, reward or training transition;
- contextual-credit authority;
- HPEM, personal/contextual memory or variable-group behavior;
- environmental selection, adaptive epsilon, lane, precision or recovery behavior;
- qualification/effect-size/non-inferiority/AUC changes that can alter policy promotion;
- any change that makes an old CALO run scientifically non-equivalent under the same algorithm ID.

Diagnostic tests may expose a Class B defect before approval, but production behavior remains
unchanged until the proposal is approved.

## Gates

| Gate | Scope | Required evidence |
|---|---|---|
| G0 | Baseline and contracts | dirty-tree inventory, matching implementation branch, change classification |
| G1 | Foundational correctness | configuration contract tests, correctly scoped historical/current release tests, clean relevant lint/tests |
| G2 | Memory and fallback | shared GPU leases, 80% of free-at-admission tests, host available-RAM tests, OOM state-machine tests, parity/provenance |
| G3 | XPU removal | versioned migration fixtures, zero executable XPU paths, historical records viewable, CUDA/CPU regressions |
| G4 | Containers | hashed CPU/CUDA locks, non-root images, volumes, health checks, SBOM, CPU and physical-CUDA qualification |
| G5 | Experiment protocol | lossless migrations, one immutable global protocol, transactional application, fingerprint stability |
| G6 | Scientist GUI | no normal-view engineering/venue language, evidence wizard validation, headless interaction tests |
| G7 | Policy lifecycle | independent training state, central resource leases, every experiment bound to governing policy SHA |
| G8 | Algorithm proposal | alternatives, rationale, risks, versioning, ablations, falsification tests; explicit user approval |
| G9 | Approved algorithm | new algorithm version, parity boundaries, ablations, development/validation qualification |
| G10 | Scientific evidence | expanded cases/baselines/holdouts, power-aware runs, raw artifacts, independent validation |
| G11 | Release | complete requirement audit, staged artifact manifests, reproducible images, documentation and handoff |

## Current implementation status — 2026-08-03

This is an implementation ledger, not a release declaration. “Implemented” means the source and
focused tests exist; hardware-dependent gates remain open until they are executed on the target
machine/container. The historical v6.9 freeze and root manifest are intentionally stale during this
development branch and must not be regenerated before the final staged release audit.

| Gate | Status | Current evidence / remaining boundary |
|---|---|---|
| G0 | Complete | Baseline commit and dirty-tree ownership recorded; work isolated on `codex/calo-complete-modernization`. |
| G1 | Implemented, final release check pending | Current configuration contains no utilization, memory-percentage, lane-share, device-job-count or work-stealing knobs. Historical fields are accepted only at the strict loader migration boundary and discarded. SQLite now uses schema version 1, creates and integrity-checks an online backup before populated v0 migration, records its SHA-256 receipt, migrates transactionally, preserves representative legacy rows, reopens idempotently and rejects future schemas without mutation. Historical release artifacts remain immutable. |
| G2 | Implemented, physical CUDA proof pending | Shared device lease, 80%-of-free VRAM/RAM admission, CUDA-resident/staged-host/CPU-fallback states and focused regression tests exist. Physical RTX 4060 pressure/parity/thermal qualification remains. |
| G3 | Implemented | New schemas expose only `cuda_preferred` and `cpu_only`; old CUDA modes migrate to `cuda_preferred`; historical XPU modes remain readable but validation rejects them as view-only. No executable XPU source/import remains. |
| G4 | CI/static/reproducibility inputs implemented; runtime proof pending | CPU/CUDA profiles, local-only noVNC, unprivileged UID/GID 10001, dropped capabilities, no-new-privileges, read-only root filesystem, bounded `/tmp`, persistent `/data`, one explicitly selected NVIDIA device, a 24 GiB default host-RAM ceiling, health check and runbook exist. The linux/amd64 Python base is digest-pinned, Debian APT uses a dated snapshot, and separate Python 3.11 CPU/CUDA 12.8 dependency graphs are exact and SHA-256 enforced. A third hash-complete CI lock drives the release-critical source/scientific, offscreen-GUI, staged-artifact and container lanes; every external action is pinned to a full commit SHA. The cross-version Linux/Windows compatibility matrix intentionally resolves within bounded project ranges because wheel graphs differ by Python and platform, then runs `pip check` and uploads the exact `pip freeze --all` result for each job rather than pretending the Python 3.11 Linux lock applies everywhere. CI verifies approved indexes, required exact pins and complete SHA-256 coverage without re-resolving the release graph against mutable index metadata, then pip enforces every release-lock hash during installation. The container smoke gate checks non-root/read-only execution, CUDA visibility contract, writable `/data`, the exact database schema version, config/SQLite round-trip, and an image-filesystem manifest derived from `/opt/calo`. Static contract tests pass. Actual workflow runs, generated image attestations, Docker runtime proof and WSL2 GPU passthrough remain open. This host has a Docker client but no daemon or Compose/Buildx plugin, so runtime proof cannot be produced here. |
| G5 | Implemented | Study-strength protocols are validated on a deep copy, display a scientist-readable before/after diff, then atomically replace shared configuration and propagate through state signals. Run counts use a persisted paired-effect/power/Holm planning approximation, preserve governing-policy binding, and cannot be reduced by a fixed legacy evidence profile. Final run snapshots remain immutable in the experiment database. |
| G6 | Implemented; Linux packaged-lane execution pending | Normal experiment UI exposes two compute choices and no device percentages/batches/schema controls; policy UI hides No-AI/unqualified and routing internals; Dashboard readiness exposes available memory, admission status and recoverable queue progress instead of utilization/worker engineering. A rendered-widget contract checks Dashboard, Experiment Manager, Portfolio Manager, CALO Intelligence and Benchmark & Evidence for venue/development/backend/schema/XPU/Safe-80/utilization language. The complete Windows/offscreen GUI suite passes locally (33 tests) and produced a validated 1440x900 dashboard PNG. CI persists the corresponding Linux rendering plus accessibility evidence; its first packaged execution remains pending. |
| G7 | Implemented | Policy training remains independently configured; qualified active-policy binding is synchronized into every new experiment while stored experiment snapshots remain immutable. |
| G8 | Complete | On 2026-08-03 the scientific lead stated exactly: “Approve TSH-CALO A–E, with F experimental and evidence-gated.” The current CALO is frozen as the baseline. TSH-CALO will use new algorithm/state/action/policy ABI versions, cannot auto-activate, and cannot inherit old superiority evidence. The required nine-part runtime/training confirmation was presented before upgraded implementation. |
| G9 | In progress — production-path optimizer and qualification-receipt mechanics locally verified; scientific qualification pending | A–F evidence remains green and F remains experimental/off by default. `TSHCALOOptimizer` requires a separately qualified, explicitly activated, immutable ensemble before any power-system evaluation; consumes counted topology contexts, shielded per-learner actions and the canonical transition; preserves exact accounting/resume; and blocks or explicitly relabels fallback. TSH activation now also requires an integrity-checked development-only qualification receipt binding the exact policy, protocol, seeds, evidence-artifact reference and OOD calibration; protected holdouts and any receipt/calibration mutation are rejected, and inference revalidates the full receipt. Five added receipt-path cases, a 27-test qualification/lifecycle/inference/optimizer focus and the 545-pass active tree pass. These checks validate receipt integrity, not the referenced evidence or acceptance decision. A real candidate, formal qualification execution, target-CUDA parity/pressure traces, counted Jacobian availability for E, ablation benefit and Change-F promotion remain absent. |
| G10 | Partially complete | Statistical corrections, honest claim boundaries, power-aware planning and a frozen-design preregistration protocol exist in `SCIENTIFIC_VALIDATION_PROTOCOL.md`. The runtime-enumerated 22-method campaign defaults to 98 initiated paired runs for 21 CALO-versus-comparator tests at effect 0.50, 95% power, Holm family control and 10% failures; pilot/simulation designs require an evidence SHA. Immutable campaign design is hashed while runtime status remains updateable. case30/57 are validation replays, case118/300 are protected tests, and a confirmatory plan cannot omit all protected tests or relabel one. Source-traceable L-SHADE 1.0.1 supplies corrected success-history DE mechanics on CPU/tensor paths. The official pinned pycma 4.4.4 engine supplies active CMA-ES with a disclosed feasibility-first dense-rank adapter, latent mixed-variable encoding, CPU control residency and CUDA-capable common evaluation. Both have deterministic snapshots and focused formulation tests. External campaign execution, mathematical-solver comparisons, approved-architecture ablations and physical qualification remain. |
| G11 | Harness partially implemented | A fresh dedicated `artifacts/python-dist` stage is required to be absent before each build, preventing obsolete distributions from entering the new wheel/sdist manifest. Generated policy checkpoints, lineages and training metadata are explicitly excluded from packages. The CPU smoke container generates its filesystem manifest from the built `/opt/calo` tree. CI uploads those staged records with the GUI rendering and CycloneDX SBOM. Final release freeze, actual image digests/attestations, clean-machine reproduction and requirement-by-requirement closure still follow G9/G10. |

Latest verification evidence:

- continuation baseline at `7ec5b840193a4fe347c42e2d9ea1796fcac929e6`, before upgraded CALO
  implementation: **453 passed, 63 skipped**; Ruff lint/format pass across 363 checked Python files;
  generated experiment schema current;
- Change-A canonical transition refactor: **461 passed, 63 skipped** complete active-tree regression;
  **45 passed** focused CALO/parity/continuation; **22 passed** frozen optimizer seeded
  snapshot/exact-budget gate; **8 passed** dedicated canonical-kernel invariants. No B–F behavior
  was enabled by this evidence.
- Change-B topology-state/encoder contract: **5 passed** focused invariants and **466 passed, 63
  skipped** complete active-tree regression. This is implementation correctness only, not evidence
  that graph context improves CALO.
- Change-C hierarchical policy/action contract: **8 passed** focused invariants and **474 passed,
  63 skipped** complete active-tree regression. The focused set includes a bounded CPU/CUDA forward
  comparison on the available device; it is not physical CUDA qualification or optimization
  performance evidence.
- Change-D uncertainty/bandit/safety/fallback contract: **6 passed** focused invariants, **28
  passed** cumulative A–D focus and **480 passed, 63 skipped** complete active-tree regression.
  These tests establish mechanics and deterministic replay, not calibrated uncertainty quality or
  optimization benefit.
- Change-E optional physics proposal/accounting contract: **10 passed** focused invariants, **38
  passed** cumulative A–E focus and **490 passed, 63 skipped** complete active-tree regression. The
  operator remains masked by default and has no incremental-benefit evidence.
- Change-F experimental population-schedule mechanics: **9 passed** focused invariants and **47
  passed** cumulative A–F mechanics. The active development tree excluding the deliberately stale
  v6.9 release-integrity file is **499 passed, 63 skipped**. The complete tree is **502 passed, 63
  skipped, 2 failed**, where both failures are the expected stale freeze/root-manifest gates. The
  schedule remains disabled by default and has no promotion, benefit or acceptable-cost evidence.
- TSH-CALO immutable candidate lifecycle: **7 passed** dedicated artifact/registry tests and **33
  passed** with existing policy compatibility, binding, independence, topology and hierarchical-
  action regressions. The active development tree is **506 passed, 63 skipped**; Ruff lint/format
  pass across 378 Python files and the generated experiment schema is current. This proves lifecycle
  mechanics only; no candidate has been trained, qualified or activated.
- Independent TSH-CALO PPO core: **7 passed** dedicated design-hash/leakage/masked-update/exact-
  resume/export/separation tests and **22 passed** with lifecycle and hierarchical-action regressions.
  The active development tree is **513 passed, 63 skipped**; Ruff lint/format pass across 380 Python
  files and the generated schema is current. Rollout state production and target-CUDA admission/
  execution remain to be integrated and physically qualified.
- Immutable ensemble and shielded inference core: single members cannot activate; ensemble assembly
  preserves member hashes/provenance. **6 passed** dedicated admission/identity/replay/shield/fallback/
  separation tests and **35 passed** across inference, lifecycle, uncertainty shield, trainer and
  hierarchical actions. The active tree is **520 passed, 63 skipped**; Ruff lint/format pass across
  382 Python files and the schema is current. This is not end-to-end optimizer execution or physical
  CUDA qualification.
- Counted ORPD solver context: **3 passed** dedicated no-extra-call/equivalence/fail-closed tests and
  **37 passed** with topology, repair, transition and frozen-CALO runtime guards. The active tree is
  **523 passed, 63 skipped**; Ruff lint/format pass across 383 Python files and the schema is current.
  The context is ephemeral and not publication evidence by itself.
- TSH-CALO runtime context and versioned candidate-transition mechanics: **9 passed** dedicated
  measured-context/group-action/physics-failure/ABI/precision-channel invariants and **68 passed**
  with counted physics, canonical-kernel, seeded optimizer snapshots and deployed/native parity.
  The active tree excluding only the deliberately stale v6.9 release-integrity file is **532 passed,
  63 skipped**; repository Ruff lint/format passes across 387 Python files and the generated schema
  is current. This does not yet constitute an end-to-end optimizer, CUDA qualification or benefit
  evidence.
- TSH-CALO policy-gated optimizer mechanics: **8 passed** dedicated execution/preflight/fallback/
  accounting/registry/F-gate/independence/exact-resume cases and **66 passed** across optimizer,
  registry, frozen campaign and benchmark snapshots. The active tree excluding only the stale v6.9
  release-integrity file is **540 passed, 63 skipped**; repository Ruff lint/format passes across
  389 Python files and the generated schema is current. These are synthetic CPU mechanics, not a
  trained or scientifically qualified policy, target-CUDA evidence, or performance evidence.
- Immutable TSH-CALO qualification/calibration receipt: **5 added cases** cover exact receipt
  round-trip, protected-case rejection, generic-row activation refusal, mutation refusal and runtime
  revalidation; **27 passed** across qualification, lifecycle, inference and optimizer. The active
  tree is **545 passed, 63 skipped**; Ruff lint/format passes across 391 Python files and the schema
  is current. A receipt authenticates declared inputs only and does not prove that qualification
  evidence exists, is sufficient, or passed its preregistered criteria.

- automatic CUDA-first scheduling/config/GUI regressions: **54 passed**; versioned database migration,
  history, learning, resume and continuation regressions: **29 passed**;
- focused execution/schema/VRAM/GUI/policy suite: **39 passed**, followed by **27 passed** after
  current-schema serialization was tightened;
- full unit suite checkpoint: **358 passed, 62 skipped, 4 failed**; two failures were obsolete GUI
  contract assertions and have since been corrected (**12 passed** on rerun); the two remaining
  failures are the deliberately stale v6.9 freeze and package manifest gates;
- current complete development-tree suite, including offscreen GUI, integration, regression and
  scientific tests and excluding only that historical v6.9 release-integrity file:
  **453 passed, 63 skipped**, with the latest measured CI-style coverage gate passing at **66%**
  (threshold 60%);
- complete offscreen GUI/scientist contract: **33 passed** with a validated 1440x900 PNG artifact;
- repository-wide Ruff lint and format: **pass** across 359 Python files; the initial formatter pass
  mechanically normalized 115 files without intentional behavior changes;
- pinned mypy 1.20.2 bounded safety target with untyped-body checking: **pass, 9 source files**; artifact/container/lock/L-SHADE/
  study-planning focused regression: **24 passed, 1 platform skip**;
- latest power-planning/transactional-study/scientist-GUI checks: **18 passed**; schema and
  fixed-memory contract checks: **24 passed**;
- scientist-facing Dashboard readiness/queue contract plus study checks: **19 passed**;
- power-aware campaign/design-hash/case-role and rendered-interface checks: **48 passed**;
- focused L-SHADE mechanics, CPU/tensor execution, campaign integration and deterministic release
  regression checks: **39 passed**, followed by **36 passed** after source-exact rounding and repair
  telemetry corrections;
- no executable XPU match remains outside the explicit historical view-only compatibility reader;
- physical CUDA, Docker image, thermals, energy and WSL2/GUI evidence: **not yet executed**.

## Invariants

1. No old result, experiment, policy, or fingerprint is silently rewritten.
2. A changed scientific method receives a new algorithm/schema version and new evidence.
3. Training never auto-activates a policy.
4. A new experiment snapshots its governing policy and scientific protocol immutably.
5. Available-memory percentages are admission ceilings, never forced utilization targets.
6. CPU/GPU timing comparisons never mix fallback modes without explicit stratification.
7. Generated checkout files are not treated as packaged release contents.
8. A release freeze is generated from staged artifacts only after all gates pass.
