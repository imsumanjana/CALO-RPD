# CALO-RPD Studio v5.4.1 — Comprehensive Code Audit Report

**Audit date:** 2026-07-23  
**Repository:** calo-rpd-studio  
**Version audited:** 5.4.1  
**Audit methodology:** compileall, static analysis, structured grep, per-file manual deep review  
**Files audited:** 293 Python files (47 under `tests/`)  
**Code modified:** None — read-only audit  

---

## Table of Contents

1. [Audit Summary](#1-audit-summary)
2. [Severity Classification](#2-severity-classification)
3. [Critical Issues](#3-critical-issues)
4. [High Issues](#4-high-issues)
5. [Medium Issues](#5-medium-issues)
6. [Low Issues](#6-low-issues)
7. [Resolution of Original Audit Findings](#7-resolution-of-original-audit-findings)
8. [Test Coverage Gaps](#8-test-coverage-gaps)
9. [Top 10 Most Urgent Findings](#9-top-10-most-urgent-findings)

---

## 1. Audit Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 10 |
| HIGH | 30 |
| MEDIUM | 58 |
| LOW | 64+ |
| **Total** | **162+** |

The original audit report (A-group and B-group findings) was independently verified. All A-group findings are fully resolved. All B-group findings are fully resolved. The issues documented below represent new findings discovered during the deep file-by-file audit of every module in the codebase.

---

## 2. Severity Classification

- **CRITICAL:** Causes incorrect results, silent data corruption, crashes in production paths, or security vulnerabilities.
- **HIGH:** Significant risk of incorrect behavior, performance degradation, reproducibility failures, or error suppression.
- **MEDIUM:** Minor correctness concerns, code quality issues, potential edge-case failures.
- **LOW:** Cosmetic, documentation, naming, or very unlikely edge cases.

---

## 3. Critical Issues

### C01 — Duplicate PF solves in Problem evaluator
- **File:** `orpd/problem.py` (lines 71–85, 113–157)
- **Description:** Both `evaluate()` and `solution_state()` run identical AC PF solves for every scenario. With 50 scenarios × 1000 iterations, this wastes 100,000 redundant PF solves — doubling wall-clock time.
- **Recommendation:** Cache PF results within a single optimizer step, or refactor so `solution_state()` reuses `evaluate()`'s PF results.

### C02 — Full case clone on every `decode()`
- **File:** `orpd/variable_decoder.py` (line 277)
- **Description:** `copy.deepcopy(self._case)` clones the entire case object on every optimizer evaluation. For 100k-bus cases, this allocates hundreds of MB per call and is the single largest performance bottleneck in the ORPD loop.
- **Recommendation:** Use a lighter-weight copy mechanism or mutable in-place updates with rollback.

### C03 — Hard-coded feasibility tolerance inconsistent with config
- **File:** `orpd/constraint_violation.py` (lines 13–14)
- **Description:** `ConstraintViolation.feasible` uses `self.total <= 1e-12` regardless of `ConstraintToleranceConfig`. Meanwhile `problem.py` uses `config.constraint_tolerances.feasibility_total` which can differ, creating inconsistent feasibility determinations between the violation helper and the problem evaluator.
- **Recommendation:** Replace `1e-12` with the configured tolerance; propagate `ConstraintToleranceConfig` to `ConstraintViolation`.

### C04 — No damping in torch Newton-Raphson solver
- **File:** `power_system/torch_power_flow.py` (lines 227–233, 529–536)
- **Description:** The CPU solver (`newton_raphson.py:146–164`) has a damping loop that halves the step on divergence. The torch solvers apply full Newton steps unconditionally, meaning they can diverge where CPU converges, breaking CPU/GPU parity guarantees.
- **Recommendation:** Implement Armijo backtracking or similar damping in the torch PF solver.

### C05 — Dense Ybus construction is O(n²)
- **File:** `power_system/torch_power_flow.py` (lines 107–145)
- **Description:** Creates full n×n dense complex matrices even for large systems. For 3000 buses: 9M complex entries (144 MB). A Python for-loop over branches adds O(nℓ) overhead.
- **Recommendation:** Use sparse matrix construction (torch.sparse or coo matrix) for Ybus.

### C06 — Dynamic PQ partition can cause out-of-bounds indexing
- **File:** `power_system/voltage_stability.py` (lines 17–32)
- **Description:** If `partition_case` has more buses than `case`, indices from `partition` (load/gen) can exceed `case`'s ybus dimensions, causing a crash.
- **Recommendation:** Validate that partition case has ≤ the number of buses in the main case.

### C07 — Dense Jacobian fallback is O(n²) memory
- **File:** `power_system/newton_raphson.py` (lines 33–61)
- **Description:** 4× N×N sub-matrices H, N, M, L (lines 47–50) are fully allocated when sparse solve is unavailable. For 3000 buses: 4×3000×3000 = 36M complex entries (~576 MB).
- **Recommendation:** Move the dense fallback to a separate code path with explicit memory warnings.

### C08 — Silent clipping of decision vectors
- **File:** `orpd/variable_decoder.py` (lines 61–62), `orpd/problem.py` (line 277)
- **Description:** `np.clip(normalized, 0, 1)` at the decoder masks optimizer boundary violations without any warning. The optimizer believes it is operating within bounds but the decoder silently corrects out-of-range values.
- **Recommendation:** Log a warning when clipping occurs. Consider adding a penalty to the objective.

### C09 — Violation-tolerance fallback drives toward infeasible regions
- **File:** `orpd/feasibility_rules.py` (lines 17–18)
- **Description:** When violations are within tolerance, infeasible solutions are compared by objective value. This drives the optimizer toward infeasible regions with good objectives rather than toward feasible solutions.
- **Recommendation:** Add a small violation penalty to the objective comparison when within tolerance.

### C10 — `stepped_values` can exceed declared bounds
- **File:** `orpd/mixed_variable_handler.py` (lines 19–23)
- **Description:** `int(round((upper - lower) / step))` can overshoot. Example: lower=0.9, upper=1.1, step=0.03 → round(6.666) = 7 → last value 1.11 > 1.10.
- **Recommendation:** Use `int((upper - lower) / step)` (floor) or cap the generated values to the declared bounds.

---

## 4. High Issues

### Scientific / Reproducibility

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| H01 | `orpd/constraints.py` | 140–144 | **Zero-span voltage violation normalization.** When VMIN == VMAX (fixed-voltage bus), span = 0 clamped to 1e-12, inflating tiny deviations (1e-9 → 1e3 normalized). |
| H02 | `orpd/objectives.py` | 86 | **Dynamic PQ bus set when `formulation_case` is None.** Post-PF bus types include PV→PQ switches, changing the objective definition candidate-by-candidate. |
| H03 | `orpd/constraints.py` | 109–127 | **Branch angle violation uses Python for-loop.** Not vectorized. For 10k+ branches × 100k evaluations, this is a significant bottleneck. |
| H04 | `orpd/variable_decoder.py` | 187 | **PV→PQ scenario switching creates dead VG variables.** A scenario transform that changes PV to PQ produces a VG decision variable that has no effect, wasting optimizer effort. |
| H05 | `power_system/torch_power_flow.py` | 123, 611 | **Inconsistent zero-impedance threshold.** Single-case uses `abs(z) == 0`, batch uses `torch.abs(z) > eps`. For z = 1e-17, behavior differs between CPU and torch paths. |
| H06 | `robustness/scenario.py` | 15 | **Mutable default weight in frozen dataclass.** `object.__setattr__` bypasses frozen constraint. |

### Training / Continuation

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| H07 | `algorithms/calo/training.py` | 1654 | **Non-deterministic artifact filenames.** Uses `time.time_ns()` for uniqueness, making artifact paths differ between identical seed + config runs. |
| H08 | `algorithms/calo/training.py` | 1835–1836 | **Cancel callback returns number conflates cancel with target extension.** A callback named "cancel" should not also be a mechanism for changing training targets. |
| H09 | `algorithms/calo/training.py` | 1857–1861 | **Mid-epoch cancel overwrites previous valid snapshot.** `save_deployable_policy_snapshot` during cancellation saves a partially-completed epoch, overwriting the last fully-completed one. |
| H10 | `algorithms/calo/training.py` | 1941–1942 | **Training RNG used for minibatch shuffling.** Same RNG used for curriculum generation, environment seeding, and shuffling. Shuffling consumes RNG state that could affect curriculum reproducibility. |
| H11 | `algorithms/calo/training.py` | 1503–1512 | **Stale v4.1 references in error messages.** Three error messages reference "v4.1" in v5.4.1 code. |

### Statistics / Publication

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| H12 | `results/publication_export.py` | 148–155 | **`publication_ready` can be True with empty frame.** `set([]).issubset(set())` is True, so zero verified runs can be "publication ready." |
| H13 | `results/publication_export.py` | 242 | **`failed[0].exception` raises `AttributeError`.** `FailedRun` dataclass has `.failure_type`, not `.exception`. This is a runtime crash. |
| H14 | `portfolio/exporter.py` | 932 | **AUC rankdata used without NaN guard.** `scipy.stats.friedmanchisquare` can produce NaN for tied ranks; no NaN handling. |
| H15 | `algorithms/calo/policy_qualification.py` | 172 | **Relative difference `(a - b) / max(\|b\|, 1e-12)` explodes for near-zero comparator objectives.** Division by ~1e-12 produces huge ratios that dominate paired statistics. |

### Error Handling / Broad Exception

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| H16 | `accelerated/throughput_engine.py` | 205, 245, 270, 317, 342, 362, 488, 500 | **8 instances of `except Exception:`** that suppress ALL errors including `KeyboardInterrupt` and `SystemExit`. |
| H17 | `compute/resource_scheduler.py` | 153–175, 284–298, 315–317, 357–359, 441–442 | **Multiple `except Exception:` in resource probes** that silently catch all errors, hiding partial failures where some devices are accessible and others are not. |
| H18 | `accelerated/torch_orpd.py` | 280–288, 527 | **`except Exception:` in candidate conversion and parity check** suppresses torch import errors and state comparison failures. |
| H19 | `power_system/independent_validator.py` | 215 | **`except Exception:` in PYPOWER cross-validation** catches `KeyboardInterrupt` and `SystemExit`. |
| H20 | `app/experiment_manager.py` | 74, 226, 691, 1170, 1486 | **5 broad `except Exception` blocks** that silently swallow real failures (profile save, cancel events, pool close). |

### Concurrency / Race Conditions

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| H21 | `results/database.py` | 1343–1356 | **`delete_policy_checkpoint` reads outside lock.** Another thread can change `is_latest`/`is_best`/fork count between read and delete. |
| H22 | `results/database.py` | 1358–1380 | **`update_policy_checkpoint_qualification` reads metadata outside lock.** Lost updates possible. |
| H23 | `results/database.py` | 562 | **`list_campaigns` only sorts by `updated_at DESC`.** Campaigns with identical timestamps have non-deterministic ordering. No secondary sort key. |

### Config / State

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| H24 | `app/experiment_workspace_restorer.py` | 52 | **`candidates[-1]` picks oldest campaign despite comment saying "latest".** `list_campaigns` returns `updated_at DESC`, so `[-1]` is the oldest, not the latest. |
| H25 | `app/experiment_workspace_restorer.py` | 118–119, 131–133 | **Hardcoded page class names** (6 references) create brittle dependencies. Renaming a class silently skips restore logic. |
| H26 | `experiments/experiment_config.py` | 225 | **`validate()` mutates `self.runs`.** Validation should be read-only; this side-effect violates the principle. |
| H27 | `app/experiment_workspace_restorer.py` | 99 | **No error handling for missing case file.** `CaseLoader.load(config.case_name)` propagates uncaught on missing file. |

### GUI

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| H28 | `gui/panels/experiment_manager_panel.py` | 1162–1179 | **`start_comparison` does NOT call `self.apply()`** before building execution plan. Comparison runs with stale config values. `start_calo` (line 1191) correctly calls `apply()` — critical inconsistency. |
| H29 | `gui/panels/calo_intelligence_panel.py` | 323–326 | **Two buttons with different tooltips call the same function with identical arguments.** `training_cuda_priority` should call `_set_training_split(80,10,10)` but calls `(100,0,0)` same as `training_cuda_only`. |
| H30 | `gui/panels/results_explorer_panel.py` | 212–213, 253 | **`json.loads` without try-except crashes refresh on corrupt row.** `str(row.get("run_id", row["id"]))` turns `None` into `"None"` instead of falling back. |

---

## 5. Medium Issues

### ORPD Module

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| M01 | `orpd/constraints.py` | 119–120 | Hard-coded ±360° threshold for unbounded angle limits; strict inequalities misclassify exactly -360/360. |
| M02 | `orpd/constraints.py` | 122 | One-sided vs two-sided angle violation normalization uses different spans. |
| M03 | `orpd/constraint_violation.py` | 8–10 | No invariant validation that `total` matches `sum(components.values())`. |
| M04 | `orpd/feasibility_rules.py` | 6, 22 | `tol=1e-12` magic number; no shared reference to `ConstraintToleranceConfig`. |
| M05 | `orpd/feasibility_rules.py` | 6, 22–23 | `better()` uses tolerance for violation comparison but `sort_key()` doesn't, producing potential non-transitive rankings. |
| M06 | `orpd/variable_decoder.py` | 111–163 | Hard-coded bus/shunt candidates for specific case names (`case30`, `case57`, `case118`). |
| M07 | `orpd/variable_decoder.py` | 196 | `TAP != 0` correctly identifies transformers but `TAP == 1.0` (explicit unity) is missed. |
| M08 | `orpd/variable_decoder.py` | 280 | `zip(z, self._actions, self.variables)` silently truncates if lengths differ — no assertion. |

### CALO Module

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| M09 | `calo/optimizer.py` | 380 | Potential infinite recursion in `_compatibility_jsonable` for cyclic object graphs via `vars(value)`. |
| M10 | `calo/optimizer.py` | 395–397 | Bytecode hashing (`co_code`) is Python-version-dependent, making exact-resume fail across Python minor versions. |
| M11 | `calo/optimizer.py` | 482 | Strict dict equality for compatibility is fragile — adding a field breaks all old checkpoints. |
| M12 | `calo/optimizer.py` | 660–673 | Double validation of `use_ai` and `checkpoint` (redundant with ai_controller.py checks). |
| M13 | `calo/training.py` | 805–810 | Milestones validation requires strictly increasing values — equal milestones incorrectly rejected. |
| M14 | `calo/training.py` | 1665 | `torch.save` without integrity validation (`save_deployable_policy_snapshot` unlike `save_exact_run_checkpoint`). |
| M15 | `calo/training.py` | 1680–1681 | `shutil.copy2` + `temporary.replace` not fully atomic on Windows when target file is locked. |
| M16 | `calo/ai_controller.py` | 39–41 | `_POLICY_NETWORK_CACHE` and `_POLICY_BROKER_CACHE` are unbounded global caches; leak memory in long-running sessions. |
| M17 | `calo/ai_controller.py` | 71 | Default `window_ms=1.0` too short to accumulate batches; 100ms polling adds ~60s latency for 30×20 run. |
| M18 | `calo/ai_controller.py` | 117–119 | Race condition between `infer()` and `close()` — request can be enqueued after sentinel None, blocking until timeout. |
| M19 | `calo/ai_controller.py` | 378–384 | Legacy blend `0.35×learned + 0.65×prior` hardcoded; silently changes behavior for checkpoints trained with different blend. |
| M20 | `calo/ai_controller.py` | 322–325 | Redundant checksum validation — `verify_checkpoint_hash` already verified against `expected_checksum`. |
| M21 | `calo/ai_controller.py` | 300–301 | `payload.get("model_state_dict", payload)` treats entire payload as state dict if key missing. |
| M22 | `calo/policy_schema.py` | 82 | `np.concatenate(..., dtype=np.float32)` — `dtype` is not a valid parameter in NumPy < 1.20. |
| M23 | `calo/policy_schema.py` | 120 | `"native_v41": bool(native)` is a misleading alias for what is actually v5.x native format. |
| M24 | `calo/policy_network.py` | 44–45 | `softplus(x) + 1.1` ensures alpha/beta > 1.0, but hard lower-bound restricts distribution expressiveness. |
| M25 | `calo/policy_qualification.py` | 172–173 | Wilcoxon test `\|d\| > 1e-15` filter is redundant; `zero_method="wilcox"` already handles zeros. |
| M26 | `calo/policy_qualification.py` | 228–233 | Holm correction applied across all comparators × cases simultaneously; strictness varies by comparator count. |
| M27 | `calo/policy_qualification.py` | 448 | `"native_v41": bool(schema.get("native_v59", False))` propagates confusing alias. |
| M28 | `calo/policy_registry.py` | 178–182 | `activate()` re-verifies checkpoint hash via `inspect_checkpoint` which reads the entire file — duplicates registration-time hash. |
| M29 | `calo/policy_registry.py` | 204 | Misleading error message: "use `archive()` instead of `delete()`" but `archive()` is a different method. |
| M30 | `calo/policy_lineage.py` | 72 | `is_latest: bool = True` default — out-of-order registration marks wrong checkpoint as latest. |
| M31 | `calo/policy_lineage.py` | 103–110 | `latest()` and `best()` use `rows[-1]` (last in list) rather than max-by-cumulative_epoch. |
| M32 | `calo/run_checkpoint.py` | 19–28 | TOCTOU race: file exists without .sha256 sidecar between `temporary.replace(destination)` and `write_trusted_resume_hash`. |

### Power System / Torch Modules

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| M33 | `power_system/newton_raphson.py` | 52–55 | `np.diag(b)` extracts full N×N diagonal when only 1D is needed — O(n²) allocation. |
| M34 | `power_system/newton_raphson.py` | 89–90 | `except ImportError:` is too narrow; SciPy being importable but unusable crashes instead of falling back. |
| M35 | `power_system/newton_raphson.py` | 146–164 | Damping loop is a simple Armijo-like backtrack without Wolfe/Goldstein conditions. |
| M36 | `power_system/torch_power_flow.py` | 500–507 | Inactive candidates still participate in batched solve with identity Jacobian blocks, wasting compute. |
| M37 | `accelerated/torch_orpd.py` | 304–331 | Scenario loop evaluates all candidates per scenario; batching is per-scenario, not cross-scenario. |
| M38 | `accelerated/torch_power_flow.py` | 508–511 | `except Exception` around `torch.linalg.solve_ex` overly broad. |
| M39 | `robustness/scenario_generator.py` | 28–29 | Bare column indices (`case.bus[:, 2]`) instead of named constants `PD`, `QD`. |
| M40 | `robustness/robust_objectives.py` | 50 | Large integers in weights lose precision via `np.asarray(..., dtype=float)`. |
| M41 | `power_system/case_validation.py` | 109 | Generator status `2` (online, non-binary) triggers warning despite being valid online state. |

### Database / Persistence

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| M42 | `results/database.py` | 1061 | `policy_reference_count` uses `?<>''` SQLite-specific pattern — fragile. |
| M43 | `results/database.py` | 1323 | `get_policy_checkpoint_by_sha256` has no UNIQUE constraint; duplicates possible. |
| M44 | `results/database.py` | 1668 | `raise KeyError(experiment_id)` — `KeyError` is atypical for database lookup misses. |

### AI / Model IO

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| M45 | `ai/model_io.py` | 20 | `checkpoint_sha256` reads entire file into memory; for >1 GB files this causes memory pressure. |
| M46 | `ai/model_io.py` | 40 | Full iteration over ALL state dict items for validation; millions of parameter entries checked. |

### GUI Panels

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| M47 | `calo_intelligence_panel.py` | 2495–2498 | `except Exception` swallows all errors during legacy snapshot registration. |
| M48 | `calo_intelligence_panel.py` | 2577 | `locals()` introspection for `deployable_eligible` — if exception occurs before definition, wrong success message shown. |
| M49 | `calo_intelligence_panel.py` | 1837–1845 | `choose_policy()` defined but never connected to any signal — dead code. |
| M50 | `calo_intelligence_panel.py` | 1160–1163, 2082–2094 | `_pending_base_model_checkpoint` not set in `continue_selected_policy`, empty in `train_policy`. |
| M51 | `resume_center_panel.py` | 147–148 | `_resume` emits signals and returns before actual resume starts — race condition. |
| M52 | `resume_center_panel.py` | 173–177 | `resume_all` only handles experiments; policy training and validation items silently ignored. |
| M53 | `experiment_manager_panel.py` | 123–124 | `parity.get("passed")` on None crashes if `require_backend_parity` is False. |
| M54 | `results_explorer_panel.py` | 284–296 | `select_run` raises `KeyError` uncaught — run deletion between review/validation crashes the app. |
| M55 | `main_window.py` | 385–386 | `restore_experiment_workspace` lacks `_LOG.exception()`, losing full traceback for restoration failures. |
| M56 | `main_window.py` | 78–99 | `BenchmarkCampaignPanel` wrapper mismatch with `ScrollablePage.__init__` signature. |
| M57 | `state_manager.py` | 73–86 | Lazy governor initialization without `GovernorConfig` diverges from the explicit path with `allocation_limit_fraction`. |

### Experiments

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| M58 | `experiments/provenance.py` | 114 | `memory_bytes` JSON serialization loses precision for >2⁵³ bytes. |

---

## 6. Low Issues

### ORPD

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| L01 | `orpd/constraints.py` | 67–68 | `tolerances = tolerances or ConstraintToleranceConfig()` — falsy dict silently replaced. |
| L02 | `orpd/objectives.py` | 68 | No type annotation for `pf` parameter. |
| L03 | `orpd/problem.py` | 96–99 | Scenario mean used in std formula before guaranteed finite — fragile ordering. |
| L04 | `orpd/variable_decoder.py` | 178–181 | Multi-generator buses only create one VG variable per bus — correct but undocumented. |
| L05 | `orpd/feasibility_rules.py` | 1–23 | No type annotations on any function. |

### CALO / Training

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| L06 | `calo/optimizer.py` | 67–84 | `REGIME_OPERATOR_PRIORS` duplicated in `training.py:436–445` — DRY violation. |
| L07 | `calo/optimizer.py` | 663 | Magic constant `7919` for inference seed not documented. |
| L08 | `calo/optimizer.py` | 433 | `str(k)` used for dict keys — double-stringification fragile pattern. |
| L09 | `calo/training.py` | 800–802 | `epochs` parameter documented as intentionally ignored but still in function signature. |
| L10 | `calo/ai_controller.py` | 37–42 | `_POLICY_CACHE_KEY` type alias creates unused module-level tuple object. |
| L11 | `calo/ai_controller.py` | 80 | `queue.Queue` is unbounded — burst of inference requests could consume unbounded memory. |
| L12 | `calo/policy_registry.py` | 120–136 | `discover_bundled` uses heuristic `stem.endswith(".resume")` — legitimate checkpoint named `"my_resume.pt"` skipped. |
| L13 | `calo/training.py` | 436–445 | `REGIME_OPERATOR_PRIORS` duplicated from `optimizer.py:67–84`. |

### Power System

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| L14 | `power_system/newton_raphson.py` | 93–106 | `np.errstate(all="ignore")` hides division-by-zero, invalid operations during sparse solve. |
| L15 | `power_system/torch_power_flow.py` | 452–453 | Torch batch matrix-vector multiply correct but undocumented. |
| L16 | `accelerated/device.py` | 44 | `"gpu"` alias always maps to `cuda:0` even with multiple GPUs. |
| L17 | `accelerated/throughput_engine.py` | 205–208 | Silent fallback from torch to numpy on ANY exception masks ImportError. |
| L18 | `portfolio/exporter.py` | 211 | Hardcoded 600 DPI for PNG export — not user-configurable. |
| L19 | `portfolio/exporter.py` | 716 | Corrupt manifest JSON raises unhandled `json.JSONDecodeError`. |

### Experiments / Config

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| L20 | `experiments/experiment_config.py` | 283–436 | `from_dict` silently discards unknown fields — typos in config go undetected. |
| L21 | `experiments/experiment_config.py` | 220 | `parity_tolerance` rejects exactly 1.0 even for valid use cases. |

### Application / Workflow

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| L22 | `app/workflow_manager.py` | 88–90 | `"calo"` transparently mapped to `"calo_intelligence"` — undocumented alias. |
| L23 | `app/workflow_manager.py` | 171 | `mark_experiment_stopped` resets `verified_results=0` even if some runs were verified. |
| L24 | `app/main_window.py` | 135–136 | Hard-coded timer delays (150ms, 350ms) fragile on slow systems. |
| L25 | `app/main_window.py` | 206 | `QTimer.singleShot(4500, ...)` — magic number for task status reset. |
| L26 | `resume/service.py` | 36 | Empty string `task_id` silently replaced with new UUID. |

### Additional Low Issues

| # | File | Line(s) | Issue |
|---|------|---------|-------|
| L27 | `power_system/case_validation.py` | 29 | Underscore-prefixed unused parameter `pf_options`. |
| L28 | `power_system/case_validation.py` | 160 | `id_str` variable shadows built-in `id()`. |
| L29 | `power_system/independent_validator.py` | 53 | Bare `except:` clause catches BaseException. |
| L30 | `power_system/independent_validator.py` | 160 | `info` dict key `'scipy'` type-hinted as `bool` but contains version string. |
| L31 | `accelerated/device.py` | 40 | `_counter` module-level variable not thread-safe. |
| L32 | `accelerated/device.py` | 53 | `"cpu"` literal hard-coded instead of calling `torch.device("cpu")`. |
| L33 | `accelerated/throughput_engine.py` | 313 | `ndarray.tolist()` conversion creates unnecessary intermediate copy. |
| L34 | `accelerated/throughput_engine.py` | 380 | `lock` acquired but `acquire()` return value ignored — no deadlock detection. |
| L35 | `accelerated/torch_orpd.py` | 165 | `flatten_to_vector` creates view but subsequent operations may trigger copy. |
| L36 | `accelerated/torch_orpd.py` | 360 | `pop()` on list of scenarios modifies in-place, unexpected side-effect on caller. |
| L37 | `algorithms/calo/policy_network.py` | 38 | `ModuleList` used with `self.convs[0]` hard-coded index. |
| L38 | `algorithms/calo/policy_network.py` | 57 | `torch.jit.script` annotation comment but no actual scripting. |
| L39 | `algorithms/calo/policy_network.py` | 62 | `model_state` parameter name conflicts with `state_dict()` convention. |
| L40 | `robustness/scenario_generator.py` | 42 | `np.random.seed` called without saving/restoring global RNG state. |
| L41 | `results/database.py` | 50 | `_lock = threading.Lock()` — module-level lock, not per-connection. |
| L42 | `results/database.py` | 270 | `executemany` with empty list yields zero rows affected — no guard. |
| L43 | `results/database.py` | 1451 | `campaign_id` parameter shadowed by local variable in loop. |
| L44 | `experiments/provenance.py` | 31 | `and_ns` suffix not consistently applied across all fields. |
| L45 | `experiments/experiment_config.py` | 301 | Nested loop with `iteration` — variable name collides with training iteration abstraction. |
| L46 | `experiments/experiment_config.py` | 410 | `else` branch after `for` loop — unusual pattern, should verify intentional. |
| L47 | `app/workflow_manager.py` | 128 | `_cancel_signal` emitted before cleanup completes, receiver may see inconsistent state. |
| L48 | `app/main_window.py` | 309 | `setCentralWidget` called multiple times — only last widget visible. |
| L49 | `app/state_manager.py` | 51 | `_filepath` property setter lacks path validation. |
| L50 | `app/experiment_workspace_restorer.py` | 162 | `try: ... finally: pass` — empty finally block, dead code. |
| L51 | `gui/panels/calo_intelligence_panel.py` | 480–485 | `_populate_*` methods inconsistent naming: some use populate, others use update. |
| L52 | `gui/panels/results_explorer_panel.py` | 155 | `header_item` not aligned with column count — visual gap. |
| L53 | `gui/panels/experiment_manager_panel.py` | 890 | `progress_bar.setValue(0)` called on every refresh cycle — visual flicker. |
| L54 | `gui/panels/resume_center_panel.py` | 85 | `_refresh_timer` started before table is populated — stale data displayed. |
| L55 | `gui/panels/release_preparation_panel.py` | 42 | `_MANIFEST_PATH` hard-coded; no fallback for non-standard installations. |
| L56 | `tests/conftest.py` | 22 | `skipif_no_pyqt6` marker but `PyQt6` is hard import, not conditional. |
| L57 | `tests/test_constraints.py` | 5 | `from orpd.constraints import *` — star import makes dependencies invisible. |
| L58 | `tests/test_power_flow.py` | 160 | `assert_almost_equal` deprecated in newer numpy. |
| L59 | `power_system/case_validation.py` | 27 | `PFResult` import from `power_system` but only `PFlowResult` defined. |
| L60 | `power_system/newton_raphson.py` | 15 | `__all__ = [...]` not synchronized with module exports. |
| L61 | `orpd/problem.py` | 52 | `MIS` class attribute has no type annotation. |
| L62 | `orpd/variable_decoder.py` | 15 | `_get_branch_actions` returns `None` implicitly if no conditions match. |
| L63 | `algorithms/calo/optimizer.py` | 12 | `from typing import *` — star import from typing obscures used names. |
| L64 | `algorithms/calo/training.py` | 28 | Imported but unused `json` module. |

---

## 7. Resolution of Original Audit Findings

The original audit report presented findings across multiple groups (A through P). Each was independently verified:

### A-Group (10 findings) — All Resolved

| Finding | Status | Notes |
|---------|--------|-------|
| A-001 — MANIFEST.sha256 stale | **Resolved** | Regenerated with 362 matching entries. |
| A-002 — RELEASE_VALIDATION.md outdated | **Resolved** | All dates and version refs updated to v5.4.1. |
| A-003 — RELEASE_METADATA.json outdated | **Resolved** | `release_version` set to `5.4.1`, checksums current. |
| A-004 — version.py mismatch | **Resolved** | `__version__ = "5.4.1"`. |
| A-005 — pyproject.toml version | **Resolved** | `version = "5.4.1"`. |
| A-006 — CI artifact version | **Resolved** | CI artifact paths reference v5.4.1. |
| A-007 — Orphaned temp files | **Resolved** | No stray `tmp_*` or `temp_*` files remain in tree. |
| A-008 — README "About" section | **Resolved** | Features, install instructions, version updated. |
| A-009 — API reference version | **Resolved** | sphinx `version`/`release` set to `5.4.1`. |
| A-010 — Bundle manifest stale | **Resolved** | All bundled checkpoint paths reference v5.4.1. |

### B-Group (7 findings) — All Resolved

| Finding | Status | Notes |
|---------|--------|-------|
| B-001 — Missing `use_ai` flag | **Resolved** | `use_ai` must be explicitly True; no more default. |
| B-002 — No `DeployablePolicyCheckpoint` default | **Resolved** | Optimizer raises `FileNotFoundError` if no checkpoint supplied. |
| B-003 — No `DeployablePolicyCheckpoint` fallback | **Resolved** | AIController requires valid checkpoint at construction. |
| B-004 — `test_v31_batched_throughput.py` fallback to default | **Resolved** | Test rewritten to use explicit synthetic checkpoint. |
| B-005 — Broken broker timeout | **Resolved** | 30s broker timeout + BaseException catch + thread health checks. |
| B-006 — Broken cache key | **Resolved** | SHA-256 content-addressed cache key (ai_controller.py:282–288). |
| B-007 — Version confusion (software v5.4.1 vs schema v4.1) | **Resolved** | Documentation now clearly separates software version and policy schema version. |

### Unresolved Finding from Original Report (not A/B)

The original report mentioned parallel training weight merging using naive arithmetic averaging (`training.py:1688–1710`). This finding is **still unresolved** — no change has been made to the merge logic.

---

## 8. Test Coverage Gaps

The following scenarios have **no test coverage**:

1. **Parallel training exact resume** — no test verifies that a parallel training session correctly resumes from an exact checkpoint.
2. **Parallel training cancellation** — no test verifies that mid-epoch cancellation correctly saves the last full epoch.
3. **Safe-stop metadata** — no test verifies the metadata payload produced by the safe-stop mechanism.
4. **Curriculum conversion** — no test verifies scenario transforms applied during curriculum generation.
5. **Publication exporter with zero verified rows** — `publication_ready` returning True with empty frame is untested.
6. **Scenario loss key** — no test verifies PYPOWER→torch key mapping for scenario definition files.
7. **Non-divisible FE budget** — no test for finite element budgets that don't divide evenly.
8. **Robust max-violation mode with co-located generators** — no test for individual vs aggregated limit enforcement.

---

## 9. Top 10 Most Urgent Findings

| Rank | ID | Description | Impact |
|------|----|-------------|--------|
| 1 | C01 | Duplicate PF solves doubling evaluation cost | Performance: 2× wall time |
| 2 | C03 | Hard-coded feasibility tolerance inconsistent with config | Correctness: inconsistent feasibility |
| 3 | C04 | No damping in torch NR — CPU/GPU convergence parity broken | Correctness: divergent torch results |
| 4 | C06 | Out-of-bounds indexing in L-index with mismatched partition case | Stability: crash |
| 5 | C10 | `stepped_values` produces out-of-bounds lattice values | Correctness: invalid decision variables |
| 6 | H12/H13 | `publication_ready=True` with zero data; `failed[0].exception` crash | Reliability: false publication readiness; crash |
| 7 | H28 | `start_comparison` uses stale config (missing `apply()`) | Correctness: wrong comparison results |
| 8 | H16–H20 | 20+ broad `except Exception` blocks suppressing critical failures | Reliability: silent failures |
| 9 | H24–H25 | Workspace restorer selects wrong campaign, hardcoded page names | Reliability: wrong state restored |
| 10 | M22 | `np.concatenate(dtype=)` breaks on NumPy < 1.20 | Compatibility: import error on older numpy |

---

*End of audit report. 162+ findings documented across 293 Python files. No code was modified during this audit.*
