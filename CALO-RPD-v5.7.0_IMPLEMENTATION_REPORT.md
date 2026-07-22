# CALO-RPD Studio v5.7.0 — Scientific Audit Closure Implementation Report

**Release:** 5.7.0 — *Scientific Audit Closure and Evidence Integrity*  
**Date:** 2026-07-22  
**Scope:** remediation of Section 2, “Previous v5.4 findings that still need resolution,” from the v5.6 deep re-audit.

## Scope discipline

This release deliberately addresses the **carried-forward v5.4 findings** first. It does **not** claim that the separate new-v5.6 competitive-training findings from Section 3 of the re-audit are all solved. In particular, transactional multi-branch exact-resume commit, early Safe Stop snapshot coverage, infinite-training memory bounds, stuck-child cancellation deadlines, comparator order-independence/common-bundle re-evaluation, and branch-aware accelerator admission remain a separate remediation phase.

The v5.6 competitive multi-branch architecture is retained: independent PPO branches do not average neural parameters, exact branch working state is separate from Branch Champion/Base state, and Base-Guided Fork remains distinct from Exact Resume.

## 1. Release and evidence integrity

- Application identity is consistently **5.7.0**.
- Release-facing benchmark defaults use **v5.7/v570** naming rather than stale v500 defaults.
- Generated `publication_export` evidence is excluded from the source release.
- Current release evidence is regenerated for the v5.7 tree and does not reuse v5.6 validation claims.
- `calo_v570_freeze.json` cryptographically freezes **110** scientific/runtime files, including newly corrected objective, Newton, statistics, fairness, comparison, robust/scenario, qualification, restoration and trust-boundary modules.
- No default neural policy is bundled or implied.
- `MANIFEST.sha256` covers **372 release-controlled files** and is self-excluded by design to avoid recursive hashing.

## 2. Policy qualification

Formal saved-Base qualification was hardened:

- independent power-flow validation is a mandatory grading gate;
- default promotion evidence is 30 paired runs, with a 30-run minimum for promotion-grade decisions;
- objective evidence is aggregated case-wise rather than pooling raw objectives from different IEEE cases;
- feasibility-first convergence AUC penalizes the pre-feasible interval;
- infeasible raw objective values are not substituted when no best-feasible trajectory exists;
- finite-comparator edge cases are guarded;
- strongest-grade decisions require favorable paired direction/effect evidence plus Holm multiplicity correction.

Periodic in-training formal qualification is intentionally **retired by design**. The compatibility field remains fixed at `0`; formal qualification applies to saved Base artifacts under a separately budgeted campaign.

The `calo-v4.1` policy/state/action identifiers remain ABI labels, not application-version labels.

## 3. Robust ORPD and scenarios

- Robust feasibility defaults to **all-scenario maximum constraint violation**.
- Expected-weighted violation is retained only as an explicit alternative formulation.
- Scenario weights are validated as finite, nonnegative and positive-sum at the low-level scientific boundary.
- `Scenario` objects are immutable and validate name/weight/transform invariants.
- Transformed cases undergo full case validation plus topology/baseMVA/bus-number invariants.
- Exact-resume callable fingerprints include code identity, defaults, keyword defaults and closure-cell values, preventing physically different closure-captured scenario transforms from hashing identically.

## 4. Power-flow constraints and objective definitions

- Active and reactive generator limits are accounted **per individual online generator**, not hidden inside bus-aggregate capability checks.
- Voltage-deviation uses a fixed pre-solve formulation PQ/load-bus partition.
- Kessel–Glavitsch L-index uses a fixed declared formulation load/generator partition.
- Dynamic PV→PQ switching remains a numerical feasibility mechanism and no longer redefines these objective/metric partitions candidate-by-candidate.
- CPU Newton-Raphson now uses vectorized Jacobian construction, sparse solve where available, deterministic dense fallback, and monotone backtracking damping.
- Boundary clipping remains a declared common repair rule and is now instrumented with repair counts/rates instead of being silent.

## 5. FE fairness and runtime semantics

Strict equal-FE studies now require the requested FE budget to be divisible by the common population size. This prevents CALO from stopping before a partial tail while another optimizer consumes it. Nondivisible strict configurations fail validation before a comparison campaign starts.

The following performance work is deliberately **not falsely closed**:

- the full CALO cognitive/control plane is not wholly device resident;
- per-learner stochastic candidate/cognition logic remains partly Python/NumPy;
- an attempted deeper vectorization was rejected because it changed the frozen seeded CALO trajectory;
- further vectorization therefore requires a separately parity-qualified algorithm-version protocol.

Exact-cache persistent storage adapts after 64 requests rather than carrying a long low-yield warm period. Policy inference defaults to CPU for the current host/NumPy CALO control architecture, avoiding pointless accelerator-to-host policy-action round trips. Cross-run policy batching is bypassed for single-slot execution.

## 6. Publication evidence and statistics

Publication generation is fail-closed:

- when independent validation is required, zero verified rows blocks the publication artifact;
- publication plots/tables/statistics use verified publication rows only;
- no arbitrary row is labeled “best” when no feasible finite solution exists;
- scenario feasibility requires PF convergence **and** finite total constraint violation within tolerance;
- scenario loss uses the stored `total_loss_mw` field;
- `publication_ready` requires complete expected paired verified evidence and feasible evidence for every compared algorithm;
- generic comparison ranking filters to verified, feasible, finite objective evidence;
- composite feasibility-first merit statistics are explicitly labeled as composite merit rather than raw-objective effect sizes;
- statistical helpers guard NaN/Inf/empty/all-identical/insufficient-pair cases.

## 7. Continuation and workspace restoration

- Exact-resume scenario fingerprints include closure/code identity.
- Workspace restoration selects an explicit scientific revision, not whichever row was updated most recently.
- Base PF restoration uses the exact saved `PowerFlowOptions`.
- PF restoration failure blocks downstream restoration/unlock.
- Legacy/no-workspace restoration infers completion conservatively.
- Exact CALO continuation and paired recompute-from-original-seed are stored as distinct trajectory semantics.
- JSON/checkpoint durability helpers use flush/fsync + atomic replace where supported; the release avoids claiming absolute crash-proof behavior.
- Live GUI-only telemetry is explicitly ephemeral/non-publication evidence. Authoritative reconstruction uses committed histories/checkpoints/revision records/results.

## 8. GUI workflow/state

The scientific workflow order is now:

`Power System / formulation -> ORPD -> Algorithms -> CALO policy only if CALO is selected -> Scenarios / Portfolio -> Experiment`.

Workspace persistence failures are logged/reported. Policy-binding restoration preserves useful diagnostic context. Panel restoration uses named identities rather than hard-coded page indexes.

## 9. Policy registry and trust boundary

- Policy suppression is scoped to the project/results database and persisted transactionally by SQLite.
- Delete/suppression semantics distinguish actual file removal and prevent silent rediscovery.
- Malformed/incompatible policy discovery is diagnostic rather than silent.
- Portable deployable policy loading uses safe weights-only semantics.
- Pickle-capable exact-resume state is **trusted-local only** and requires machine-local HMAC-SHA256 authentication plus SHA integrity before `weights_only=False` loading.
- Portable/deployable checkpoint APIs are explicitly separated from trusted-local exact-resume APIs.

## 10. Broker/device diagnostics and exception handling

- Forced broker evaluator failures propagate under bounded waits.
- Evaluator/device calibration is explicitly labeled evaluator-only.
- CALO runtime reports end-to-end requested-FE throughput including control/evaluation/learning wall time.
- A regression test forbids any broad `Exception`/`BaseException` handler whose sole body is `pass`.
- Critical checkpoint/provenance/device failures are logged or propagated rather than silently swallowed.

## 11. Carried-forward findings that remain partial/accepted rather than falsely closed

The detailed matrix is `FINDINGS_CLOSURE_v5.7.0.csv`. The main residual classifications are:

- **H-002 — PARTIAL/DEFERRED FOR PARITY:** full per-learner/device-native CALO control optimization remains performance work; frozen seeded behavior is preserved.
- **H-003 — MITIGATED:** adaptive cache overhead is reduced and observable; exact within-request keying remains intentionally active.
- **J-008 — ACCEPTED NON-SCIENTIFIC TELEMETRY SCOPE:** hard crash can lose UI-only transient points; publication evidence does not depend on them.
- **N-001/N-003 — TARGET ENVIRONMENT LIMITATION:** PyQt6, PYPOWER, Ruff and physical CUDA/XPU validation are unavailable in this build runtime and are not falsely reported as passed.
- **N-004 — PARTIAL:** carried-forward scientific defects now have regression tests; tests for the separate new-v5.6 Section-3 findings belong to the next remediation phase.
- **O-001 — PARTIAL STRUCTURAL DEBT:** architecture boundaries are documented, but several orchestration modules remain large.

## 12. Validation performed

- `python -m compileall -q calo_rpd_studio calo_bootstrap tests` — PASS.
- Dependency-light unit partitions — **203 passed, 1 skipped**. The skip is the PYPOWER-dependent formulation test because PYPOWER is not installed.
- Integration + regression — **9 passed**.
- v5.7 carried-forward audit-closure tests — **19 passed**.
- v5.7 release-integrity tests — **4 passed**.
- Frozen seeded optimizer regression + accelerator parity/device tests — **22 passed**.
- `calo_v570_freeze.json` — **PASS, 110/110 files**.

Unavailable in this build environment and therefore **not claimed passed**: complete PyQt6 GUI suite, full PYPOWER scientific suite, Ruff, and physical CUDA/XPU parity/throughput testing.

## 13. Scientific release position

v5.7 closes or explicitly resolves-by-design the carried-forward v5.4 scientific/evidence defects that can be corrected without silently changing CALO into a new stochastic algorithm. It deliberately does not hide residual performance/structural work, and it does not claim CALO or any policy is superior without frozen paired feasible independently validated evidence.
