# CALO-RPD Studio complete repository, containerization, and scientific audit

**Audit date:** 2026-08-03
**Repository:** CALO-RPD Studio 6.9.0
**Reviewed revision:** `307402df5c7a44a6bb852770347b1b1ef995548d` plus the pre-existing uncommitted v6.9 working tree
**Requested target:** NVIDIA CUDA VRAM first, CPU RAM only when VRAM cannot contain the admitted working set, no Intel XPU, an ordinary-scientist GUI, globally applied experiment-design guidance, and an independent CALO policy-training lifecycle
**Implementation status:** **analysis only—no product code, configuration, container, policy, GUI, or scientific behavior was changed by this audit**

## 1. Executive conclusion

The repository contains a serious and unusually broad scientific-software foundation: a common ORPD evaluator, explicit function-evaluation accounting, exact CALO resume state, immutable policy checksums, paired seeds, independent power-flow validation, robust-scenario support, policy qualification, and extensive provenance. Those are worth preserving.

It is **not ready for the requested deployment or for a universal superiority claim**. The decisive gaps are:

1. The new 80% CUDA limit is calculated from **total physical VRAM**, not currently available VRAM. PyTorch's process fraction API also defines its allowance from total memory. This directly contradicts the requested rule.
2. The VRAM governor is process-local. Several persistent workers can each receive an 80%-of-total allowance, so aggregate allocations can exceed the device.
3. CUDA OOM handling only reduces the microbatch and then fails. It does not implement a VRAM-first, CPU-RAM-offload state machine.
4. CPU “Safe-80” is also total-capacity based (`0.8 × total - used`), rather than `0.8 × currently available`.
5. XPU is an architectural lane, bootstrap requirement, configuration schema, scheduler path, GUI concept, provenance field, and test surface. Removing it is a controlled migration, not a two-file deletion.
6. There is no Dockerfile, Compose definition, `.dockerignore`, container health check, CPU/GPU image profile, or container CI gate.
7. The scientist-facing GUI exposes large amounts of implementation and publication-development terminology. The source contains 453 GUI/application references to terms such as backend, schema, provenance, microbatch, checkpoint, development, and debug; 516 repository references concern journals, Transactions, Q-level/publication language.
8. Policy training is scientifically configured independently, which is correct, but it is operationally enforced through a global exclusive lock. Conversely, policy binding is persisted only when CALO is selected even though the workflow treats the active policy as governing all downstream power-system work.
9. The documented CALO statement that every learner independently samples an operator is false for the native v5.9 neural-policy path: one controller decision is made per generation and its operator is applied to all ordinary learners, apart from precision and forced-recovery interventions.
10. The native neural path explicitly makes contextual operator credit diagnostic rather than decision-making. The deployed method is therefore not the fully contextual per-learner policy described by the methodology.
11. Policy-qualification statistics contain material definition problems: a sign imbalance is labeled rank-biserial correlation; a general Wilcoxon helper silently truncates unequal paired arrays; the “relative” difference becomes absolute below a one-unit scale; and convergence AUC uses a run-dependent pre-feasibility penalty.
12. The current working tree does not pass its release gates: Ruff reports 104 errors; the non-GUI suite reports 377 passed, 15 failed, 42 skipped; and the GUI suite cannot run because `pytest-qt` is absent from the active environment.

The correct near-term goal is not “stronger than every algorithm.” That is neither provable nor scientifically defensible across all optimization problems. The defensible target is:

> **CALO is a preregistered, domain-specialized ORPD optimizer that demonstrates superior or non-inferior feasibility, solution quality, robustness, and cost on a frozen, independently validated benchmark distribution under equal declared resource budgets.**

That claim can become very strong. It requires fixing the methodological inconsistencies, broadening the benchmark suite, adding deterministic optimization references and lower bounds, performing power-based repeated experiments, and holding back truly unseen systems and operating conditions.

## 2. Scope, method, and limitations

### 2.1 What was inspected

The audit mapped 487 non-ignored repository files, including 347 Python files and approximately 57,632 Python lines. It inspected the application/workflow layer, GUI panels, CALO controller and training code, ORPD and accelerated evaluators, compute topology and schedulers, policy qualification, statistics, schemas, packaging, bootstrap, tests, CI, release manifests, prior audit documents, and the 15-page tensor-native scientific upgrade plan.

The repository was already dirty when the audit began. The existing v6.9 modifications and untracked release artifacts were treated as user work and left untouched. Findings therefore describe the exact working tree, not only the checked-in commit.

### 2.2 Executable checks

| Check | Result | Interpretation |
|---|---:|---|
| Ruff | **104 errors** | 61 `E402`, 22 `E702`, 19 `F401`, 2 `F841`; current CI runs `ruff check .`, so CI would fail |
| Non-GUI pytest | **377 passed, 15 failed, 42 skipped** | One relevant functional fallback failure plus stale/historical release and manifest failures |
| Full pytest | **collection/setup errors** | GUI tests require `qtbot`; `pytest-qt` is declared but not installed in the active `.venv` |
| Container definitions | **none found** | No Dockerfile/Containerfile/Compose/`.dockerignore` |
| XPU references | **522** | XPU removal crosses code, tests, docs, schema, bootstrap, and provenance |
| Broad exception handlers | **150** | Not automatically bugs, but a large masking/observability surface |

The functional non-GUI failure in `tests/unit/test_heterogeneous_policy_training.py` expects two CPU episodes in CPU fallback but records zero. The release-integrity failures also reveal harness problems: old v6.5–v6.8 tests assert their historical version is still the current version, and root-manifest tests recursively include `.venv`, local reports, and other development files as if they were packaged release contents. Those failures must be separated into **product failures** and **invalid test assumptions** before release gating.

This was a source and available-runtime audit, not a physical CUDA benchmark. Exact throughput, power draw, thermal throttling, VRAM peaks, CPU/GPU overlap, and FP64 parity must be measured on the target RTX 4060 Laptop GPU in the final WSL2/container environment.

## 3. Severity-ranked findings

### 3.1 Critical findings

| ID | Finding | Evidence | Required outcome |
|---|---|---|---|
| C-01 | 80% VRAM rule uses total, not available, memory | `accelerated/vram_residency.py:118,127` | Budget a frozen admission allowance from current global free VRAM |
| C-02 | Each process can claim its own allowance | process-local lock/governor and persistent multiprocess architecture | One device-level admission/budget authority across every worker |
| C-03 | No requested CUDA→host-memory fallback | README explicitly says “never silently falls back”; OOM code only halves batches | Explicit, tested fallback/offload states with provenance |
| C-04 | XPU is a first-class architecture | 522 references; scheduler, sidecars, bootstrap, GUI, schema, tests | Versioned schema/data/provenance migration and XPU-free runtime |
| C-05 | No container deployment exists | no container files or CI jobs | Reproducible Linux CPU and CUDA profiles, locked dependencies, health checks |
| C-06 | Native-policy operator semantics contradict methodology | `optimizer.py:823-826,961,1016-1020`; `docs/calo_methodology.md:96+` | Correct docs/claims or implement and qualify a true per-learner policy |
| C-07 | “Stronger than any algorithm” is not a valid universal claim | No-Free-Lunch boundary and current open disputes | Narrow, preregistered ORPD-domain claim with falsifiable gates |
| C-08 | Governing-policy persistence is conditional on CALO selection | `app/experiment_manager.py:1882-1943` | Bind the governing policy/version to every power-system experiment |
| C-09 | Current release gates are red or un-runnable | Ruff and pytest results above | Clean, correctly scoped, CPU+GUI+CUDA+container release matrix |

### 3.2 High findings

| ID | Finding | Consequence |
|---|---|---|
| H-01 | Host RAM budget is total-based | Does not implement the requested “80% of available RAM” rule |
| H-02 | “CPU/GPU utilization should be removed” is physically ambiguous | Memory cannot execute algorithms; a CPU or GPU must perform the work |
| H-03 | Native contextual credit does not control neural operator selection | The claimed contextual learning loop is open for the deployed neural action |
| H-04 | PPO action chooses a global operator while memory/group choices remain heuristics | The learned action is only a partial controller, not joint cognition |
| H-05 | Qualification effect size is mislabeled | Promotion thresholds can be interpreted incorrectly |
| H-06 | Paired Wilcoxon helper truncates unequal vectors | Missing/misaligned repetitions can silently become a different experiment |
| H-07 | AUC pre-feasible penalty depends on each run's outcomes | Cross-algorithm AUC values can have different scales |
| H-08 | Qualification defaults cover case30/case57 only | Development qualification is not evidence of broad release generalization |
| H-09 | Fixed “Journal 30 / Transactions 50” runs are not power-based | False assurance; neither a venue nor evidence strength follows from a fixed count |
| H-10 | GUI mixes scientist workflow with engineering console concepts | High cognitive load and easy misconfiguration for normal scientific users |
| H-11 | Policy training is globally exclusive | Scientifically independent training is still operationally coupled to all GUI stages |
| H-12 | `No-AI CALO` is an ordinary GUI choice | Conflicts with the governing-policy model and confuses normal users |
| H-13 | Benchmark coverage is too narrow | IEEE 30/57 development and 118/300 holdout do not span modern ORPD conditions |
| H-14 | Baseline set needs stronger mathematical references | A large metaheuristic zoo cannot establish optimality or domain leadership |

### 3.3 Medium findings

1. `ExperimentConfig.validate()` allows population size 1, while the JSON schema requires at least 2.
2. The JSON schema says `additionalProperties: true`; the Python loader rejects unknown keys.
3. The schema defaults `throughput_profile_v31.json`; code defaults `throughput_profile_v34.json`.
4. Configuration retains utilization targets, XPU shares, XPU job counts, multiple ambiguous execution modes, and obsolete release-era terminology.
5. `.github/workflows/ci.yml` is still named “v6.2.1 CI,” has no CUDA/container job, and uses only a 60% coverage threshold.
6. `requirements-lock-cpu.txt` has exact versions but no hashes, identifies itself as a v3.4 audit lock, includes development tools, and has no matching CUDA lock.
7. The root-manifest test scans the mutable checkout instead of a staged wheel/sdist/image filesystem.
8. Environmental-selection diversity subtracts raw Euclidean distance from a normalized rank. Its influence changes with dimension and scaling; the comment that quality “still dominates” is not guaranteed.
9. `friedman_test` and related statistical paths need explicit block/shape contracts rather than relying on downstream library behavior.
10. 150 broad exception handlers warrant a separate failure-observability review, especially around accelerators, scheduling, persistence, and checkpoint import.

## 4. Feasibility correction: compute cannot be replaced by memory

VRAM and CPU RAM are storage. They do not execute Newton–Raphson, tensor operations, PPO updates, or CALO control. The requested operational intent should therefore be specified as follows:

- remove manual CPU/GPU **utilization target percentages** from the GUI and scientific configuration;
- expose only **CUDA preferred** and **CPU only** scientist-facing modes;
- in CUDA-preferred mode, execute eligible numerical kernels on CUDA and use host RAM only as staging/offload/fallback storage;
- retain a small CPU control plane for PyQt, orchestration, data loading, SQLite, logging, checksums, policy decisions that have not been ported, and final reference validation;
- never claim “zero CPU use” or “memory-only execution.”

This is consistent with the repository's own v6.9 boundary: GUI, logging, SQLite, checkpoints, and validation remain host responsibilities. It is also consistent with NVIDIA's programming model: ordinary CPU allocations and device allocations reside in different memory spaces and movement has a cost. Even unified memory performs best when migration is minimized ([CUDA Programming Guide: programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)).

## 5. Correct 80%-of-available memory contract

### 5.1 Why the current VRAM implementation is wrong for this requirement

Current code obtains both free and total memory, but then records:

```text
process_budget_bytes = total_bytes × budget_fraction
```

It passes the same fraction to `torch.cuda.set_per_process_memory_fraction`. PyTorch defines the resulting allowance as **total visible memory multiplied by the fraction**, not free memory ([PyTorch process-memory-fraction documentation](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.get_per_process_memory_fraction.html)). `torch.cuda.mem_get_info` reports global free and total device memory ([PyTorch `mem_get_info`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html)). Merely reading `free_bytes` does not make the limit availability-based.

If the 8 GB GPU has 5 GB free at admission:

- current conceptual limit: `0.8 × 8 = 6.4 GB`;
- requested additional-use limit: `0.8 × 5 = 4.0 GB`.

The current setting can therefore compete with the display driver and other processes for memory that was not free when the job started.

### 5.2 Recommended frozen admission formula

Use one GPU budget authority per physical device. At job-group admission time:

```text
F0 = global free VRAM reported immediately before admission
T  = total VRAM
R0 = memory already reserved by the admitted CALO process group
A  = floor(0.80 × F0)                  # new allowance from currently free memory
C  = min(T - driver_guard, R0 + A)     # process-group ceiling
f  = clamp(C / T, minimum_safe, 0.95)  # only if PyTorch fraction API remains useful
```

The important scientific/operational rule is `A`, not the API fraction. Freeze it for that admitted job group. Do not continuously recompute “80% of available” upward as memory is consumed, because that moving target can chase itself and undermine the cap. Re-evaluate only at a controlled admission boundary, after all previous leases are released.

Treat 80% as a **maximum allowance**, not a target to fill. NVIDIA explicitly advises against allocating all available memory merely to occupy it and recommends sizing allocations to the actual problem ([CUDA memory guidance](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-programming-guide/index.html)).

### 5.3 One device-level lease manager

The process-local Python lock is insufficient. The target needs one of:

1. a single persistent CUDA worker owning all CUDA tensors and accepting jobs over an internal queue; or
2. an interprocess GPU lease service with shared accounting, atomic reservations, worker identity, heartbeat, and stale-lease recovery.

For an 8 GB laptop GPU, the first design is simpler and safer. Cross-run microbatching should feed one CUDA owner. Multiple Python processes should not each create large CUDA contexts and allocator pools.

Every admission record should include:

- device UUID and runtime version;
- `F0`, `T`, `R0`, computed `A`, and final `C`;
- estimated working-set bytes and safety margin;
- chosen batch/microbatch;
- actual peak allocated/reserved;
- offload bytes and transfer time;
- OOM retries and final disposition.

### 5.4 Host RAM formula

At CPU-mode admission:

```text
H0 = psutil.virtual_memory().available
H_allow = floor(0.80 × H0)
```

This must be an application allocation ceiling for the admitted experiment group. It is not the same as `0.8 × total RAM - currently used`, which is what `compute/topology.py:452-455` implements.

Do not force the process to consume `H_allow`. Reserve/allocate on demand and stop admission before the projected working set exceeds it. A container memory hard limit should be a stable operator-defined ceiling; the application can use the live `H0` value as the stricter internal limit. Leave headroom for Windows/WSL2, the Docker VM, filesystem cache, GUI, and other applications.

### 5.5 Required memory tests

The corrected implementation is not accepted until tests demonstrate:

1. free=5 GB, total=8 GB produces a new allowance near 4 GB, not 6.4 GB;
2. two workers cannot each reserve the whole allowance;
3. concurrent external VRAM use cannot make admissions exceed the frozen cap;
4. allocator-reserved, allocated, driver-used, and global-free values are distinguished;
5. minimum microbatch failure transitions to the declared offload/fallback state;
6. host available-RAM changes do not make the current job's ceiling grow recursively;
7. job cancellation and worker death release leases;
8. telemetry values are consistent with NVML/`nvidia-smi` within documented accounting differences.

## 6. CUDA-first, CPU-RAM fallback/offload design

### 6.1 Scientist-facing modes

Expose only:

- **Accelerated (recommended):** use verified NVIDIA CUDA; keep active data in VRAM when it fits; use host RAM only for bounded staging/offload; fall back to CPU computation only under the declared conditions.
- **CPU:** execute entirely on CPU using the available-RAM ceiling.

Keep detailed modes, allocator diagnostics, deterministic parity switches, and tuning controls in an administrator CLI/config file, not the normal GUI.

### 6.2 Admission and runtime state machine

```text
PREFLIGHT
  ├─ CUDA unavailable or parity-unqualified ──────────────> CPU_COMPUTE
  └─ CUDA verified
       ├─ estimated active set fits allowance ────────────> CUDA_RESIDENT
       └─ does not fit
            ├─ reducible batch fits ──────────────────────> CUDA_MICROBATCHED
            ├─ bounded staged/offloaded plan fits ────────> CUDA_STAGED_HOST
            └─ no valid CUDA plan ────────────────────────> CPU_COMPUTE

CUDA_* OOM
  ├─ retry budget and smaller batch available ────────────> CUDA_MICROBATCHED
  ├─ safe staged plan available ──────────────────────────> CUDA_STAGED_HOST
  └─ minimum batch/device failure ────────────────────────> CPU_COMPUTE or FAIL_CLOSED
```

The fallback choice must be explicit in experiment provenance. For formal comparisons, silently changing compute/numerical backends mid-run can affect floating-point behavior and timing. Recommended policy:

- **scientific quality runs:** fail closed on unplanned backend change unless CPU/CUDA parity for that exact formulation has already passed and the protocol permits fallback;
- **exploratory runs:** allow CPU fallback, label the transition, and exclude heterogeneous runtime measurements from speed comparisons;
- **policy training:** checkpoint, release CUDA state, and resume a new CPU training segment only when exact continuation semantics are supported; never pretend an OOM-restarted trajectory is continuous.

### 6.3 What “offload” should mean here

Three distinct mechanisms must not be conflated:

1. **Staging:** immutable case/scenario inputs or completed outputs live in host RAM and are transferred in bounded chunks. CUDA still computes each chunk.
2. **Selective offload:** cold tensors, replay/history, checkpoints, or inactive branches live in pinned host RAM and are prefetched before use.
3. **CPU fallback:** numerical computation moves to CPU because even the minimum CUDA working set cannot execute reliably.

Mapped or unified host memory is not a free extension of VRAM. On a discrete laptop GPU it crosses PCIe and can introduce page migration/thrashing. NVIDIA notes that mapped host access is higher latency/lower bandwidth and not a performant replacement for correct data placement; unified memory also performs best with minimal migration ([CUDA programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html), [Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)). Prefer explicit microbatching and staged copies before managed-memory oversubscription.

### 6.4 Target-laptop recommendation

For the Lenovo LOQ with i7-14700HX, 32 GB DDR5-class RAM, RTX 4060 Laptop 8 GB, and 2 TB NVMe SSD:

- use the NVIDIA GPU for qualified batched FP64/FP32 tensor work where measurement proves benefit;
- keep one persistent CUDA owner to avoid duplicated contexts and allocator pools;
- expect less than 6.4 GB usable application VRAM when the GPU also drives the display; the exact allowance is 80% of free-at-admission, not a fixed number;
- use pinned host buffers only for planned transfers and cap them inside the host allowance;
- keep CPU orchestration threads bounded and leave capacity for WSL2/Windows;
- never use SSD swap as a scientific performance mechanism; it is an emergency safety net, not “CPU RAM offload”;
- benchmark FP64 power-flow kernels against the i7 CPU. Consumer Ada GPU FP64 workloads may not always beat a strong CPU at small population/case sizes, so auto-selection must use a qualified, formulation-specific profile rather than GPU marketing assumptions.

## 7. XPU removal audit and migration plan

### 7.1 Current coupling

XPU appears in at least these surfaces:

- bootstrap detection, repair decisions, sidecar provisioning, and prerequisite messaging;
- `compute/xpu_worker.py`, `compute/xpu_sidecar.py`, persistent sidecar protocols, topology discovery, binding, telemetry, resource scheduler, and experiment manager;
- policy-training device choices and CUDA/XPU/CPU percentage controls;
- `ExperimentConfig`, JSON schema, fingerprints, saved workspaces, benchmark campaign records, provenance, and release metadata;
- GUI readiness, dashboard, experiment manager, CALO Intelligence, and technical tooltips;
- tests, freeze manifests, requirements comments, documentation, historical release reports, and trained artifact metadata.

The current launcher even considers repair required when Intel hardware is detected but no XPU runtime is ready, independently of a healthy CUDA runtime (`calo_bootstrap/launcher.py:14-38`). That behavior must disappear in the XPU-free product.

### 7.2 Removal sequence

1. **Freeze a compatibility reader.** Old experiment/policy/provenance records containing XPU must remain readable. Map their lane to `legacy_xpu` display metadata, never to a runnable device.
2. **Define schema vNext.** Remove XPU fields and reduce execution mode to `cuda_preferred` / `cpu_only`. Add a schema migration from existing values:
   - `gpu_preferred`, `cuda_priority`, `cuda_only`, accelerator-capable `throughput_auto` → `cuda_preferred`;
   - `cpu_only` / `cpu_reference` → `cpu_only`;
   - XPU-only or weighted XPU saved plans → require explicit re-selection; do not silently reinterpret scientific execution history.
3. **Remove provisioning.** Delete Intel runtime repair/provisioning only after launcher and readiness tests no longer depend on it.
4. **Collapse scheduling.** Replace three-lane dictionaries, shares, executors, pools, work stealing, and utilization controls with a CUDA owner plus CPU executor.
5. **Remove sidecar code.** Only after no import, entry point, subprocess protocol, or test loads it.
6. **Migrate policy training.** Devices become `auto`, `cuda`, `cpu`; episode percentages disappear. Resource planning chooses concurrency from memory/CPU capacity, not arbitrary utilization percentages.
7. **Simplify GUI.** Remove all XPU labels and detailed device percentages from normal workspaces.
8. **Update fingerprints carefully.** Do not erase historical XPU fields from old fingerprints. Version the fingerprint schema so old scientific identities remain verifiable.
9. **Regenerate freezes/manifests from staged artifacts.** Never hand-edit a mutable root manifest.
10. **Run compatibility fixtures.** Open representative pre-removal workspaces, policies, campaigns, and result databases read-only and verify that scientific results are unchanged.

### 7.3 XPU removal acceptance gate

- `rg` finds no executable XPU imports, schedulers, configuration fields, GUI strings, or bootstrap actions;
- historical XPU records still load and clearly say “legacy runtime—view only”;
- CUDA and CPU paths pass parity, restart, cancellation, exact resume, and provenance tests;
- no policy checksum, experiment fingerprint, or stored result changes merely because runtime support was removed;
- dependency and container images contain no Intel/XPU runtime.

## 8. Containerization blueprint

### 8.1 Recommended deployment architecture

For the current desktop application, use a Linux OCI image with two Compose profiles:

```text
calo-gpu:  CUDA-enabled PyTorch runtime + CALO application + PyQt runtime
calo-cpu:  CPU PyTorch runtime + CALO application + PyQt runtime
```

They should share the same application source version and scientific dependency versions. Separate images avoid shipping a large CUDA stack in CPU-only deployments and make the provenance unambiguous. A single CUDA-capable image that falls back to CPU is operationally convenient, but less minimal and makes dependency identity harder to interpret.

On Windows, Docker Desktop GPU support requires the WSL2 backend ([Docker GPU support](https://docs.docker.com/desktop/features/gpu/), [Docker WSL2](https://docs.docker.com/desktop/features/wsl/)). The host needs a compatible NVIDIA Windows driver; the container uses NVIDIA Container Toolkit integration rather than bundling a host driver ([NVIDIA Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), [architecture overview](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html)).

### 8.2 GUI reality

PyQt inside a container needs a display channel. Under Windows 11/WSL2, WSLg can display Linux GUI applications, but GPU, audio, font, clipboard, and file-dialog behavior must be acceptance-tested. Three options exist:

1. **Near-term:** run the existing PyQt GUI in the WSL2 container through WSLg.
2. **More robust desktop:** keep a thin native GUI outside the container and call a versioned headless engine service inside it.
3. **Long-term multi-user:** make the engine an API/worker and replace the desktop with a browser UI.

Option 2 gives the best compromise for this product: immutable scientific execution in the container, reliable native desktop interaction, and a clean separation between scientist workflow and engine diagnostics. It is an architectural phase, not required for the first container proof-of-concept.

### 8.3 Image contents and build rules

The image should:

- use a pinned Python 3.11 or 3.12 base until 3.13/3.14 are explicitly qualified;
- pin a CUDA/PyTorch combination supported by the host driver;
- install from hashed lock files generated separately for CPU and CUDA;
- install the application wheel, not execute from a mutable source checkout;
- run as a non-root UID;
- set a read-only application filesystem where practical;
- contain no runtime `pip install` or first-launch hardware repair wizard;
- include licenses, source/version metadata, SBOM, and dependency vulnerability scan output;
- include a health check that verifies imports, database write volume, policy registry access, and—in GPU profile—CUDA device visibility and a tiny FP64 parity probe;
- use an init process and handle SIGTERM by safe-stopping/checkpointing active work;
- avoid embedding user policies, result databases, secrets, or machine-local HMAC keys in the image.

The current bootstrap wizard performs environment repair and hardware-specific package installation. That belongs at image build/deployment time, not application start. In-container startup should validate and fail with an actionable message; it should never mutate the Python environment.

### 8.4 Persistent volumes

Use explicit volumes or bind mounts for:

- `/data/db` — SQLite and migrations;
- `/data/results` — raw results, traces, validation, exports;
- `/data/policies` — deployable policies and authenticated resume envelopes;
- `/data/checkpoints` — active safe-stop/run checkpoints;
- `/data/config` — scientist configuration and protocol snapshots;
- `/data/logs` — operational logs;
- `/data/secrets` — machine/deployment-local signing secret, mounted with restricted permissions.

Each experiment record should store image digest, application version/commit, Python/PyTorch/CUDA versions, GPU UUID, mode, memory-admission record, policy SHA-256, formulation fingerprint, and configuration fingerprint.

### 8.5 Compose behavior

GPU profile requirements:

- request exactly the NVIDIA device(s) intended for use;
- default to one application CUDA worker;
- set a stable host-RAM container limit chosen for the machine (for example, not all 32 GB);
- use the application live-availability limiter as the stricter ceiling;
- never try to set a fixed Docker VRAM limit—enforce that inside the single CUDA owner and monitor at the driver level.

CPU profile requirements:

- no GPU device request;
- CPU affinity/quota optional and deployment-controlled;
- bounded worker count derived from physical cores and memory estimates;
- host-RAM allowance calculated from available memory at admission.

### 8.6 Container CI matrix

Minimum release matrix:

| Job | Required checks |
|---|---|
| Source | Ruff, formatting, mypy target, unit/scientific tests, dependency lock verification |
| CPU image | build, SBOM, vulnerability scan, non-root, read-only smoke, CPU parity, database migration |
| Headless GUI | Qt offscreen launch, core navigation, scientist wizard, screenshots/accessibility checks |
| CUDA image | build without device; physical runner verifies device visibility, FP64 parity, memory admission, OOM/offload, cancellation |
| Artifact | wheel/sdist/image filesystem manifest generated from staged output, not repository root |
| Compatibility | representative old databases/configs/policies load without mutation |

## 9. GUI redesign for an ordinary scientist

### 9.1 Principle

The normal GUI should present scientific decisions, not software architecture. Terms to remove from ordinary screens include CUDA/XPU percentages, backend, tensor, microbatch, schema, lineage internals, sidecar, provenance implementation, allocator, worker pool, checkpoint interval, runtime ABI, freeze manifest, and developer/audit wording.

Do **not** delete the underlying auditability. Move it to:

- an “Advanced diagnostics” window;
- downloadable machine-readable provenance;
- administrator CLI/config;
- log/export details shown only when troubleshooting.

### 9.2 Proposed scientist workflow

1. **Policy status** — active governing policy, scientific qualification status, integrity, and last qualified scope. Actions: train/manage policy in a separate studio; activate a qualified policy.
2. **Power system** — select case/data, validate network, inspect warnings.
3. **Study objective** — active loss, voltage deviation, stability, emissions/cost, multiobjective or robust target.
4. **Controls and constraints** — generators, taps, shunts, operating limits, contingencies/scenarios.
5. **Methods** — CALO and comparison methods, with scientifically fair budgets.
6. **Evidence strength** — choose a neutral evidence tier or customize it.
7. **Review protocol** — one complete summary/diff, estimated runs/storage/time, unresolved validity warnings.
8. **Run and results** — progress, failures, validated outcomes, comparisons, export/reproducibility bundle.

### 9.3 Remove journal/venue claims from the GUI

Use neutral labels:

- **Exploratory** — feasibility and method development;
- **Standard** — repeatable comparative study;
- **Strong** — broad, statistically powered, independently validated evidence;
- **Definitive** — preregistered frozen evaluation with external/repeated validation;
- **Custom** — expert design with live validity guidance.

Do not label these “Q1,” “Q2,” “Q3,” “Journal,” or “Transactions.” Quartiles describe a journal's position within a subject category and can change; they are not a property of one experiment. Clarivate describes Journal Citation Reports as journal intelligence based on multiple indicators, and Scopus describes CiteScore percentile/quartile as a serial title's relative subject-field standing ([Clarivate JCR](https://clarivate.com/products/journal-citation-reports/), [Scopus journal metrics tutorial](https://tutorials.scopus.com/EN/AnalyzeJournals/sc_AnalyzeJournals_textOnly.html)).

The documentation may explain the user's intended rough mapping—Definitive for the rigor often expected in very selective Transactions/Q1 work, Strong for substantial Q1/Q2 work, Standard for a sound scoped study, Exploratory for method development—but must explicitly state that no preset predicts acceptance or venue rank.

### 9.4 Evidence profiles and outputs

Run counts below are **starting floors**, not automatic adequacy. Final repetitions must be determined from a pilot variance and a declared smallest scientifically important paired difference. Sample-size methodology for multi-algorithm/multi-instance comparisons exists specifically because fixed counts are unreliable ([Campelo & Wanner sample-size paper](https://arxiv.org/abs/1908.01720)).

| Profile | Initial repeated-run plan | Required scientific outputs | Scope/gates |
|---|---:|---|---|
| Exploratory | pilot ≥10 paired seeds/case | feasibility rate, best/median objective, violation decomposition, convergence, runtime/memory, failure reasons | 2–3 cases; no broad superiority claim |
| Standard | ≥30 paired runs/case, then power check | median/IQR and confidence intervals, paired differences, Wilcoxon-Holm, effect size, feasibility probability, evaluations-to-feasible, independent PF validation | ≥4 diverse systems/conditions; equal FE and transparent timing |
| Strong | power-calculated, often ≥50/case | all Standard outputs plus Friedman/post-hoc ranking, robustness/CVaR, ablation, sensitivity, OOD holdout, deterministic/local-solver comparison, raw data and container digest | frozen dev/validation/test split; strong baselines; correction across hypotheses |
| Definitive | sequential/power-calculated, often ≥100 when effects are small | all Strong outputs plus preregistration, external replication, optimality/lower-bound gaps where possible, multi-hardware reproducibility, adverse-case analysis, complete artifact/SBOM | algorithm/policy frozen before test; multiple families of networks and conditions |
| Custom | calculated from user targets | live checklist derived from chosen claim | cannot apply until every required dependency is valid |

The wizard should ask for:

- primary claim: feasibility, solution quality, robustness, speed, scalability, or all;
- smallest practically important difference;
- desired confidence/power and family-wise error policy;
- cases and operating-condition population to generalize to;
- stochastic algorithms/baselines;
- maximum FE, wall-clock, and energy budgets;
- independent validation and holdout policy;
- required plots/tables/raw exports.

It should then recommend a 10-run pilot, estimate paired differences/variance, calculate additional runs, round up conservatively, and warn when compute/storage is insufficient. Optional sequential stopping must be preregistered with alpha-spending or an appropriate Bayesian decision rule; repeatedly checking ordinary p-values is not acceptable.

### 9.5 One global experiment protocol

“Apply” must create one immutable, versioned `ExperimentProtocol` snapshot containing:

- case/data identity and checksums;
- objective and aggregation;
- controls, bounds, discrete semantics, constraints and tolerances;
- power-flow options;
- scenarios/contingencies and probabilities;
- algorithms and exact parameter sets;
- policy dependency and SHA-256;
- FE/wall/energy budgets;
- paired seed plan and repetitions;
- evidence profile, tests, multiplicity correction and effect-size definitions;
- independent validation and holdout rules;
- outputs/storage/export plan;
- compute mode and numerical precision contract.

All Power System, ORPD, Methods, Scenarios, Experiment, Results, and Export panels should read this snapshot. Applying a new protocol should show a human-readable diff, validate it transactionally, update every dependent panel/state object, and invalidate only downstream stages. The user should never re-enter the same field in the Power System block.

The current `ExperimentConfig` is already a useful nucleus, but it mixes scientific formulation, execution engineering, publication intent, continuation metadata, GUI state, and XPU scheduling. Split it into:

```text
ScientificFormulation
ExperimentDesign
EvidencePlan
ComputePlan
ContinuationPlan
PresentationPreferences
```

Only the first three belong in the primary scientist wizard. Compute defaults should be automatic, continuation belongs to Resume, and presentation preferences must not affect scientific fingerprints.

## 10. Policy training must remain independent yet govern experiments

### 10.1 What is already correct

`ExperimentConfig.validate_policy_development()` validates only objective, variables, power flow, tolerances, robust objective, and scenarios. It explicitly ignores comparison portfolio run minima, device shares, and campaign constraints. The CALO Intelligence panel also constructs its own active-loss development template. This is a sound separation and should be preserved.

Power System navigation is already locked until `evaluate_governing_policy` reports an active, compatible, qualified, checksum-valid policy. New policy-assisted CALO experiments also fail closed when strict binding, policy ID/path, active status, compatibility, qualification, or SHA-256 is invalid. These are strong controls.

### 10.2 Current inconsistencies

1. `policy_training_active` activates a “Global Training Exclusive Lock” that disables almost every workspace. Training is scientifically independent but operationally blocks ordinary configuration work.
2. The experiment database binds policy only inside `if "CALO" in config.algorithms`. If the governing policy is a prerequisite for every power-system experiment, every experiment must record it—even a comparator-only study.
3. The normal GUI offers `No-AI CALO`, which can unbind a policy from the current CALO configuration. That is useful as an ablation comparator but inappropriate as a normal governing choice.
4. Resource sharing between independent policy training and experiments is not defined. Two independent workflows can still conflict for the same 8 GB GPU.

### 10.3 Target contract

- **Policy Studio is a separate lifecycle.** Train, resume, fork, qualify, compare, and activate policies without reading or mutating the current experiment protocol.
- **Training never auto-activates.** A candidate becomes governing only after completed real-runtime qualification and explicit user activation.
- **All experiments bind the active policy snapshot.** Store policy ID, SHA-256, schemas, qualification record/scope, and activation time even when CALO is not one of the comparison algorithms. For non-CALO runs, label this a governing-context dependency; do not imply it numerically modifies the baseline.
- **Experiment protocol application does not alter training.** It must not change PPO cases, curriculum, epochs, branches, device, or checkpoint.
- **Policy changes invalidate unstarted downstream setup.** Already-started/completed experiments remain bound to their original immutable policy.
- **No-AI CALO becomes a hidden validation comparator.** Keep it in qualification/ablation CLI and expert reports, not as a normal scientist path.
- **Resource leases are independent of scientific state.** Training and experiment configuration may coexist, but CUDA execution needs the same central device lease. On this 8 GB system, default to one heavy GPU task at a time; CPU-only work may coexist if host and thread budgets permit.

Suggested state model:

```text
Policy candidate -> qualified for declared scope -> explicitly active
                                              |
                                              v
ExperimentProtocol binds policy SHA -> run snapshot -> immutable result

Policy training lifecycle: independent configuration and checkpoints
Experiment lifecycle: depends only on an activated qualified snapshot, never on a live trainer
```

## 11. CALO scientific-strength audit

### 11.1 Strong foundations to retain

CALO is not a superficial wrapper around one random operator. The implementation includes:

- feasibility-first evaluation and adaptive epsilon;
- exact requested FE accounting, including cache-hit accounting;
- persistent personal best and HPEM Best-1/3/5/7 hierarchy;
- feasible and boundary archives;
- contextual success direction memory;
- mixed-variable decoding and variable-group masks;
- learning/discovery lanes, precision search, recovery intervention;
- deterministic policy mode and explicit raw/executed action traces;
- exact optimizer-state resume with formulation/policy compatibility fingerprints;
- policy checksum/qualification governance;
- paired seeds, independent PF validation, Holm correction, and stored raw records;
- protected case118/case300 holdouts in the existing workflow.

Those mechanisms provide a credible platform for a domain-specialized optimizer. They do not by themselves demonstrate superiority.

### 11.2 Material CALO logic inconsistency: per-learner operator claim

`docs/calo_methodology.md` states:

> CALO does not apply one operator to the entire population. Each learner independently samples one of six operators.

For the native v5.9 AI path, `optimizer.py` calls `controller.decide(...)` once before the population loop. Inside the loop, every ordinary learner receives `raw_operator = decision.operator`. Per-learner differences still occur through local regime adaptation, memory depth, variable group, discovery/learning lane, precision, and forced recovery, but the neural operator is generation-global.

The training environment follows the same global action semantics, so training/runtime parity is not the main issue. The issue is that the published method description and scientific interpretation are wrong. Choose one:

1. **Conservative correction:** document a generation-level neural operator with per-learner heuristic/controller adaptations and reframe claims accordingly.
2. **Stronger method:** implement a hierarchical policy that produces a global regime/continuous controls plus conditional per-learner operator (and preferably memory/group/lane) actions. Record joint log probabilities and intervention masks so PPO credit corresponds to what was sampled and executed.

The second option is scientifically more interesting but is a new algorithm version. It requires frozen ablations and fresh qualification; it cannot inherit old superiority evidence.

### 11.3 Contextual credit is disconnected in the native policy path

The code comment at `optimizer.py:1016-1020` explicitly says contextual credit, rule priors, and discovery priors remain diagnostics/learning memory and do not redefine the PPO action. Contextual credit affects the legacy/no-AI blended path and memory probabilities, but not native neural operator choice.

Therefore claims like “the online contextual operator credit adapts deployed native operator choice during a run” are false. To close the loop, use one of:

- policy logits + a calibrated contextual posterior/prior combined through a preregistered product-of-experts or residual-logit model;
- recurrent/meta-RL policy that consumes contextual reward history directly;
- contextual bandit head updated online, with the PPO policy selecting hyperparameters/priors;
- explicit two-level controller: neural strategy selection and online credit-based per-learner operator allocation.

Any online adaptation must be included in training/runtime equivalence, checkpoint state, deterministic replay, and ablation.

### 11.4 Partly learned, partly heuristic action space

The policy learns a global regime, one global operator, and continuous parameters. Memory depth and variable group are then selected per learner using fixed priors/contextual credit and variable-group intelligence. Lane assignment, precision, and recovery are environmental interventions.

This can be valid, but it must be represented as a **hybrid hierarchical controller**, not a wholly learned cognitive policy. A stronger version would use structured multi-head actions:

- global state head: regime and continuous exploration/exploitation controls;
- per-context head: operator distribution conditioned on learner context;
- memory-depth head;
- variable-group head or attention over variable groups;
- lane/precision proposal heads with safety masks;
- intervention critic/auxiliary loss to distinguish proposed from executed actions.

The action-space increase should be staged. First verify whether per-context operators improve paired feasible outcomes; do not add complexity without ablation evidence.

### 11.5 Environmental selection is dimension/scale sensitive

`environmental_selection.py` normalizes rank to roughly [0,1] but subtracts `diversity_weight × raw Euclidean distance`. Distance changes with decision dimension and variable normalization. Although decision vectors appear normalized in many paths, Euclidean distance still tends to grow with `sqrt(D)` and can overpower rank for larger problems.

Use a dimension-normalized distance (`distance / sqrt(D)`), a robust population-relative z-score/quantile, crowding distance, or a k-nearest-neighbor novelty score normalized within the candidate pool. Freeze the definition and re-run selection ablations across case dimensions.

### 11.6 Qualification statistics that require correction

#### A. Rank-biserial is not rank-biserial

`policy_qualification.py:211` computes:

```text
(number of negative differences - number of positive differences) / n
```

That is a sign-based effect, not the matched-pairs rank-biserial correlation associated with Wilcoxon signed ranks. A proper value uses positive and negative **rank sums**, with a declared sign orientation. Rename the existing metric or implement the correct estimator and confidence interval.

#### B. Unequal paired arrays are silently truncated

`statistics/wilcoxon.py` uses `n = min(len(a), len(b))` and slices both arrays. This can hide a missing run and pair the wrong observations. Require equal lengths or, better, accept keyed `(case, run_index, seed)` records, align exactly, and fail on duplicates/missing pairs.

The policy-qualification-specific path does key by case and run index, which is better, but it silently skips missing comparator rows. Formal qualification should require the predeclared complete pair set or explain every exclusion.

#### C. Relative difference changes meaning near zero

`(candidate - comparator) / max(|candidate|, |comparator|, 1)` is stable, but objectives below one use an absolute one-unit denominator. A “1%” non-inferiority margin is then not 1% of the objective. Define objective-specific practical margins in physical units or normalize every case by a frozen reference scale established before qualification.

#### D. AUC scale depends on the observed run

Pre-feasible AUC values are filled using `max(observed feasible) + 10% of observed scale`. Different algorithms/runs can therefore receive different pre-feasible penalties. Use a fixed case/formulation reference scale and FE horizon, or report separate time-to-feasibility plus conditional feasible-objective AUC. The latter is easier to interpret and avoids mixing two scientific concepts into a run-dependent scalar.

#### E. Development versus release qualification

The default qualification cases are case30 and case57. Keep them for development, but define two statuses:

- **development-qualified:** passed on development/validation systems and may be used for continued research;
- **release-qualified:** frozen before evaluation and passed protected multi-family holdouts/OOD conditions.

Only release-qualified policies should support broad performance claims. The product may allow a development-qualified policy to govern exploratory work, but must show its scope.

### 11.7 Stronger benchmark universe

IEEE 30/57/118/300 are useful but insufficient. Add PGLib-OPF, which was designed to improve and standardize AC-OPF benchmarking and includes typical and API/SAD stress cases ([PGLib-OPF repository](https://github.com/power-grid-lib/pglib-opf), [PGLib paper](https://arxiv.org/abs/1908.02788)).

Construct a frozen benchmark taxonomy:

- network size and topology family;
- light/nominal/heavy loading;
- renewable penetration and forecast-error distributions;
- N-1 branch/generator contingencies;
- tight voltage/reactive limits and Q-limit switching;
- discrete tap/shunt density;
- feasible, narrow-feasible, and deliberately infeasible instances;
- single objective, weighted multiobjective, and robust/CVaR variants;
- in-distribution validation and out-of-distribution final test.

Do not tune on the protected test. Use development → validation → frozen test, with case/condition family separation where possible.

### 11.8 Baselines that can support a leadership claim

A 20-metaheuristic campaign is broad but can still be weak if comparators are old or poorly tuned. Include:

- AC-OPF/ORPD deterministic references using mature nonlinear solvers (for example IPOPT/MIPS through a well-defined formulation);
- multi-start deterministic local optimization;
- PowerModels formulations and convex relaxations/lower bounds where applicable; PowerModels is explicitly designed to compare power-flow formulations on shared specifications ([PowerModels paper](https://arxiv.org/abs/1711.01728));
- modern differential evolution families such as SHADE/L-SHADE and a strong CMA-ES implementation;
- surrogate-assisted method for expensive robust evaluations;
- a carefully tuned mixed/discrete method appropriate to tap/shunt controls;
- CALO plus deterministic local polishing, reported separately as a hybrid;
- exact or mixed-integer/relaxation references on small instances when tractable.

Report optimality/lower-bound gaps when available, not only relative ranking among heuristics. Large-scale AC-OPF benchmark work emphasizes tools, formulations, and bounds rather than declaring one stochastic method universally best ([large-scale AC-OPF tools and bounds](https://arxiv.org/abs/2203.11328)).

Fairness needs more than equal FE:

- same physical formulation, tolerances, decoder, scenario samples, and validation;
- paired seeds/scenarios where meaningful;
- equal FE as the primary black-box budget;
- separately reported wall time, energy, memory, preprocessing, and parallel hardware;
- tuning budget disclosed and separated from final evaluation;
- deterministic solvers evaluated with iterations/evaluations and convergence/optimality criteria appropriate to their method;
- failed/infeasible runs retained, never filtered from feasibility statistics.

### 11.9 CALO upgrade program

#### Phase S0 — repair truth and measurement

- align methodology with actual global/per-learner semantics;
- correct qualification effect sizes, pairing, AUC, and scale definitions;
- freeze exact baseline implementations/tuning budgets;
- split development- and release-qualification status;
- add unit/property tests for FE accounting, constraint ordering, mixed variables, checkpoint replay, and statistics.

#### Phase S1 — high-value ablations

Run paired ablations for:

- global operator vs per-context/per-learner operator;
- neural-only vs neural+contextual residual;
- HPEM levels and personal best;
- contextual memory depth;
- variable groups;
- dual lane;
- precision and recovery;
- adaptive epsilon;
- each continuous policy parameter;
- policy vs no-AI hybrid;
- local-polish hybrid.

Use the same frozen development/validation bundle and correct for all planned comparisons. Keep only components with practically meaningful, repeatable benefit.

#### Phase S2 — hierarchical policy

- multi-head context-conditioned actions;
- intervention-aware PPO credit;
- permutation-invariant population/context encoding;
- recurrent or compact history features for nonstationarity;
- uncertainty/entropy calibration and safe fallback to transparent priors;
- explicit action masks for unavailable operators/variable groups;
- shared transition kernel or formal seeded equivalence between training and runtime.

#### Phase S3 — domain knowledge without leakage

- graph/network embeddings based on topology and electrical sensitivity;
- reactive-power/voltage sensitivity-informed groups;
- feasibility restoration operators derived from power-system structure;
- surrogate models with uncertainty and mandatory exact reevaluation;
- curriculum across topology/load/constraint difficulty;
- meta-learning across development systems while protecting final systems;
- deterministic local polishing of the best feasible candidates.

#### Phase S4 — definitive evaluation

- preregister protocol and smallest important effects;
- freeze code, image digest, policy SHA, cases, scenarios, and analysis scripts;
- determine runs by pilot/power calculation;
- execute independent replicated campaigns;
- publish all raw outcomes, failures, configurations, seeds, environment, and validation;
- report where CALO loses, not only where it wins.

### 11.10 Claim boundary

Wolpert and Macready's No-Free-Lunch result shows why universal superiority over all problem classes is not a valid objective ([DOI 10.1109/4235.585893](https://doi.org/10.1109/4235.585893)). Domain specialization can still yield a “free lunch” on a declared non-uniform problem distribution. Therefore phrase the research objective as:

> Build and demonstrate the strongest reproducible CALO variant for the declared constrained ORPD distribution and budgets, with explicit failure regions and independent holdout evidence.

## 12. Other repository logic and release issues

### 12.1 Configuration/schema drift

The code/schema mismatches listed earlier can cause different tools to accept different configurations. Establish one source of truth—preferably typed Python models capable of generating JSON Schema—and add round-trip contract tests:

```text
default model -> JSON -> schema validation -> model load -> identical scientific fingerprint
```

Unknown-field behavior must be the same everywhere. For scientific reproducibility, fail on unknown scientific fields; use explicit versioned migration for old fields.

### 12.2 Release manifest architecture

Current root-manifest tests consider every file under the checkout except a few names. That includes `.venv`, audit outputs, caches not listed in exclusions, and this requested report. A release manifest should be created from:

- built wheel contents;
- built sdist contents;
- container filesystem/application layer;
- separately declared scientific frozen bundle.

Do not assert that a developer checkout equals a release artifact. Historical release tests should validate fixtures/tags/artifacts, not assert that the current version remains 6.5, 6.6, 6.7, or 6.8.

### 12.3 Dependency reproducibility

Create:

- `requirements-lock-cpu-py311.txt` with hashes;
- `requirements-lock-cuda<version>-py311.txt` with the correct PyTorch index/artifact hashes;
- a separate dev/test lock;
- SBOM and license report;
- automated update policy with scientific regression gates.

Do not install PyTorch dynamically in a running container. The current active environment uses Python 3.14.6 while CI targets 3.11–3.13; qualify a supported version rather than relying on accidental compatibility.

### 12.4 CI and coverage

First make Ruff green. Then separate suites:

- fast unit;
- scientific invariants;
- statistics/reference-data;
- database migration;
- headless GUI;
- CPU integration;
- physical CUDA;
- container/end-to-end;
- historical compatibility.

A blanket 60% threshold is too weak for scientific decision paths and too blunt for GUI boilerplate. Add higher per-package thresholds and mutation/property tests for the evaluator, constraints, statistics, policy binding, fingerprinting, and checkpoint state.

### 12.5 Error handling

The 150 broad handlers should be classified:

- expected optional telemetry failure;
- recoverable worker/device failure;
- scientific validation failure;
- persistence/integrity failure;
- programmer defect.

Only the first two should commonly fail forward. Scientific/integrity failures should fail closed. Every catch must preserve exception type, context, device/job identity, and user-visible resolution; no empty `except`/`pass` on scientific state paths.

## 13. Recommended implementation roadmap (no implementation performed)

### Gate 0 — freeze and clarify

1. Preserve the current tree and create a clean implementation branch.
2. Decide the exact meaning of CUDA staged offload versus CPU compute fallback.
3. Adopt the neutral evidence-profile names and policy-governance contract in this report.
4. Mark current v6.9 as pre-release until release gates are credible.

**Exit:** approved architecture decision record and compatibility fixtures.

### Gate 1 — correctness before containerization

1. Correct shared VRAM/RAM admission semantics.
2. Implement one CUDA owner and explicit fallback state machine.
3. Fix the functional CPU-fallback test and statistical definition defects.
4. Resolve schema drift and release-harness design.
5. Make lint and CPU/headless-GUI suites green.

**Exit:** deterministic CPU reference and qualified CUDA parity with memory-pressure tests.

### Gate 2 — remove XPU

1. Add schema/provenance compatibility migration.
2. Collapse scheduler/configuration to CUDA-preferred and CPU-only.
3. Remove bootstrap and sidecar code.
4. Remove GUI/device percentage controls.
5. Verify historical read compatibility.

**Exit:** no executable XPU path or dependency; old records remain auditable.

### Gate 3 — container proof

1. Produce hashed CPU/CUDA locks.
2. Build non-root CPU and GPU images.
3. Add persistent volumes, safe stop, health checks, SBOM, image provenance.
4. Verify WSL2/WSLg GUI behavior on the target laptop.
5. Measure memory admission, parity, throughput, thermals, and power.

**Exit:** repeatable experiment from a clean machine using only documented host prerequisites and persistent data volumes.

### Gate 4 — scientist GUI and global protocol

1. Introduce the split configuration model and one immutable protocol snapshot.
2. Build the neutral evidence/customization wizard with pilot/power guidance.
3. Remove engineering and venue language from normal GUI.
4. Move diagnostics to advanced view/exports.
5. Make policy studio operationally separate and bind every experiment to active policy.

**Exit:** a new scientist can configure a valid study without touching compute internals or duplicating inputs.

### Gate 5 — CALO scientific strengthening

1. Repair claims/statistics and execute S0/S1 ablations.
2. Decide whether hierarchical per-learner policy earns a new algorithm version.
3. Expand benchmarks/baselines and establish deterministic bounds/gaps.
4. Run preregistered validation and protected release holdout.

**Exit:** scope-specific, independently reproducible CALO evidence—not a universal marketing claim.

## 14. Final acceptance checklist

### Compute/container

- [ ] 80% allowance is calculated from free/available memory at admission.
- [ ] One physical GPU cannot be overcommitted by multiple process-local governors.
- [ ] CUDA-resident, microbatched, staged-host, CPU-fallback, and fail-closed states are distinct.
- [ ] Fallback is scientifically permitted or blocks with an explicit reason.
- [ ] CPU and CUDA images are immutable, non-root, locked, scanned, and reproducibly built.
- [ ] No runtime package installation or XPU repair exists.
- [ ] Target-laptop WSL2/WSLg/GPU qualification is recorded.

### GUI/workflow

- [ ] No XPU, utilization percentage, backend, microbatch, schema, or developer text in normal GUI.
- [ ] No Q1/Q2/Q3/Journal/Transactions promise in normal GUI.
- [ ] Evidence profiles provide outputs, validity requirements, and adaptive run calculation.
- [ ] Applying a protocol updates all dependent stages atomically with a diff.
- [ ] Policy training configuration is untouched by experiment application.
- [ ] Every experiment binds the governing policy snapshot.
- [ ] No-AI CALO is restricted to expert qualification/ablation.

### Scientific

- [ ] Method documentation matches global/per-learner action semantics.
- [ ] Contextual credit authority is accurately described and ablated.
- [ ] Correct keyed pairing, effect sizes, margins, AUC, CIs, and multiplicity controls.
- [ ] Development and release qualification are distinct.
- [ ] PGLib/stress/OOD cases and modern deterministic/stochastic baselines are included.
- [ ] Run counts come from pilot precision/power, not venue labels.
- [ ] Code, policy, formulation, image, seeds, raw failures, and independent validation are frozen and published.
- [ ] Claims state the problem distribution and budgets where CALO was tested.

## 15. Sources consulted

### Repository sources

- `README.md`
- `pyproject.toml`, `requirements*.txt`, `.github/workflows/ci.yml`
- `calo_rpd_studio/accelerated/vram_residency.py`
- `calo_rpd_studio/compute/topology.py`, resource scheduler, persistent workers and XPU sidecars
- `calo_rpd_studio/experiments/experiment_config.py` and JSON schema
- `calo_rpd_studio/app/experiment_manager.py`, state/workflow manager
- `calo_rpd_studio/gui/panels/*`
- `calo_rpd_studio/portfolio/models.py`
- `calo_rpd_studio/algorithms/calo/optimizer.py`, training, heterogeneous/competitive training, policy qualification, environmental selection, memory/credit/controller modules
- `calo_rpd_studio/statistics/*`
- `docs/calo_methodology.md`, prior scientific/open-dispute/audit reports, architecture/reproducibility/validation documents
- `docs/CALO_vNext_Tensor_Native_Scientific_Upgrade_Plan.pdf`

### External primary/authoritative sources

1. PyTorch, [`get_per_process_memory_fraction`](https://docs.pytorch.org/docs/main/generated/torch.cuda.memory.get_per_process_memory_fraction.html).
2. PyTorch, [`torch.cuda.mem_get_info`](https://docs.pytorch.org/docs/stable/generated/torch.cuda.mem_get_info.html).
3. NVIDIA, [CUDA Programming Guide—programming model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html).
4. NVIDIA, [CUDA Programming Guide—Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html).
5. Docker, [GPU support in Docker Desktop](https://docs.docker.com/desktop/features/gpu/) and [WSL2](https://docs.docker.com/desktop/features/wsl/).
6. NVIDIA, [Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and [architecture](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/arch-overview.html).
7. Babaeinejadsarookolaee et al., [PGLib-OPF](https://arxiv.org/abs/1908.02788) and the [official repository](https://github.com/power-grid-lib/pglib-opf).
8. Coffrin et al., [PowerModels.jl](https://arxiv.org/abs/1711.01728).
9. Tanneau et al., [large-scale AC-OPF tools and bounds](https://arxiv.org/abs/2203.11328).
10. Hansen et al., [COCO performance assessment](https://arxiv.org/abs/1605.03560).
11. Campelo and Wanner, [sample-size calculations for comparing multiple algorithms and instances](https://arxiv.org/abs/1908.01720).
12. Wolpert and Macready, [No Free Lunch Theorems for Optimization](https://doi.org/10.1109/4235.585893).
13. Clarivate, [Journal Citation Reports](https://clarivate.com/products/journal-citation-reports/).
14. Scopus, [source metrics and CiteScore percentile/quartile tutorial](https://tutorials.scopus.com/EN/AnalyzeJournals/sc_AnalyzeJournals_textOnly.html).

---

**Decision:** do not begin broad GUI cleanup, XPU deletion, or container work as isolated edits. First approve the memory/fallback contract, configuration split, policy-binding contract, and compatibility migration. Those decisions determine almost every downstream file and prevent another release from claiming a behavior that the runtime does not actually provide.
