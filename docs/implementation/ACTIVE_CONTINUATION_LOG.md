# Active CALO-RPD continuation log

This is the durable, append-only working log for the active TSH-CALO continuation. It exists so work can resume without relying on chat context. Keep observed facts, artifact identities, commands or validations, decisions, commits, and the next action current. Corrections must be added explicitly; do not silently rewrite scientific outcomes.

## Scope and standing boundaries

- Active objective: continue from the user-specified `main` baseline `7ec5b840193a4fe347c42e2d9ea1796fcac929e6` through the documented gates.
- Architecture approval: TSH-CALO A–E are approved for careful production-candidate implementation; F is experimental, independently feature-flagged, evidence-gated, and disabled by default.
- Policy training remains independent of power-system experiments. Experiments cannot train, modify, qualify, register, or activate policy artifacts.
- An experiment can consume only a separately qualified, registered, explicitly activated, immutable, checksum-valid, compatible policy; otherwise it uses the deterministic safe baseline fallback.
- No push, merge, publication, release, final freeze, release manifest, SBOM, image digest, or release-ready claim is authorized.
- `Docker_Build.txt` is an untracked user-owned file and must remain untouched.

## Resume checkpoint — 2026-08-04 (Asia/Calcutta)

- Goal status: active; confirmed through the goal service.
- Branch: `main`.
- Current source commit: `33f29370ade972ee00ae07c22bf3d204a2dbaedd`.
- Working tree before this log: no tracked changes; only `?? Docker_Build.txt`.
- Repository-wide `AGENTS.md` instructions were previously created. Root, `docs/`, and `docs/implementation/` instructions were re-read before creating this log.

### Completed implementation commits

- `78e6f800b675670365bebf58f876e3da4fef117d` — preserve resumable training interruptions.
- `80286afdd24b124c0515dce6edbc17ae52184460` — add independently invoked, evidence-gated TSH-CALO qualification campaign.
- `33f29370ade972ee00ae07c22bf3d204a2dbaedd` — serialize qualification campaign writers and reject failed-integrity resumes.

### Qualified facts and retained failures

- The real v2 training campaign completed at `artifacts/tsh-calo-training-runs/tsh-calo-ieee30-57-v2-20260804`.
- Candidate ensemble: `ensemble.candidate.pt`.
- Candidate SHA-256: `3adf5017dc51f33d76214aeb505da598984b9da0e4263e1ee8fe59a667180ceb`.
- Training manifest SHA-256: `ded60598652d552a70f03c811969092ab243437f2d5adaf8d7f75f665bc80f33`.
- The candidate remains `candidate_unqualified`; it has never been registered or activated.
- The first screening output, `artifacts/tsh-calo-qualification-runs/tsh-calo-ieee30-57-v2-screening-v1-20260804`, is permanently retained as failed-integrity evidence. `campaign_integrity_failure.json` forbids scientific use, qualification, and resume.
- The v2 screening path contains only a refused launcher attempt caused by pre-created log paths. It is not a scientific campaign and must not be used as evidence.

### Active evidence audit

- Valid screening plan: `artifacts/tsh-calo-qualification-plans/tsh-calo-ieee30-57-v2-screening-v3-20260804.json`.
- Plan file SHA-256: `b519fa3014d97b2ecb93205dacb94d57f386604b2bc60a2c18514a2a5d9bd2bc`.
- Scientific-design hash: `9d1583703ee5ae211b00269dab6054676354d2d524ebf0085e92e17e63ce92a9`.
- Execution hash: `0a3ba70c35b1601b850af49da2772aa59c127e2ca8682fdfe5bd627b5657c46a`.
- Seed-manifest hash: `4cfe6933eea9abe26a4273db66a676458ff63450a53394c24c2aff730b8b22f1`.
- Screening output: `artifacts/tsh-calo-qualification-runs/tsh-calo-ieee30-57-v2-screening-v3-20260804`.
- External stdout/stderr logs use the same basename with `.stdout.log` and `.stderr.log` suffixes.
- Frozen design: IEEE 30 and IEEE 57, 10 paired runs per case, population 40, exactly 2,000 function evaluations per run, screening only, CPU policy inference, no device fallback, and eight development-only OOD calibration states per case.
- Last observed state before this checkpoint: 39 of 40 run records complete, zero recorded failures, final IEEE 57 candidate run executing. Completion and aggregation have not yet been audited in this log.
- Interim IEEE 30 observation only: neither arm produced a feasible run in 10 attempts. This cannot support qualification or an objective-superiority claim.

### Validation state at this checkpoint

- Qualification-focused tests: six passed after the single-writer fix.
- Active test tree: `582 passed, 63 skipped`, excluding only the intentionally stale `tests/unit/test_v690_release_integrity.py` freeze test.
- Ruff check and format check: passed across 406 files.
- Schema validation: current and passing.
- These are local development results, not container, hardware-matrix, independent replication, release, or superiority evidence.

## Exact next actions

1. Audit whether the v3 screening process exited and whether all 40 unique records were durably written.
2. Verify that screening produced `qualification_evidence.json` but no qualification receipt, registration, or activation artifact.
3. Check every record against the frozen plan, policy hash, case/seed identity, exact 2,000-evaluation budget, device provenance, and independent power-flow validation fields.
4. Hash and summarize the evidence and calibration artifacts; record the exact non-qualification decision and permitted claims here and in the handoff, gates, and traceability documents.
5. Commit that evidence documentation as a logically scoped commit.
6. Continue to the next incomplete gate in documented order without weakening acceptance criteria or regenerating release artifacts.

## Subsequent entries

Append timestamped entries below this line after each material action, validation, decision, or commit.

### 2026-08-04 — v3 screening completion and integrity audit

- The v3 campaign completed with state `completed_not_qualified`; both observed launcher/process IDs (`14920` and `30080`) exited.
- Output contains 40 unique records and zero failure records. The exact expected set is present: two cases × ten paired indices × baseline/candidate.
- All 40 records carry the frozen execution-plan hash `0a3ba70c35b1601b850af49da2772aa59c127e2ca8682fdfe5bd627b5657c46a`, candidate hash `3adf5017dc51f33d76214aeb505da598984b9da0e4263e1ee8fe59a667180ceb`, exact paired seeds, 2,000 candidate evaluations, 2,000 recorded evaluations, and 2,000 scenario power-flow calls.
- All 40 retained solutions passed the independent PYPOWER validation record. This establishes validation agreement for these retained solutions only; it is not a general CPU/CUDA equivalence claim.
- All 20 candidate-inference records requested, selected, and computed on CPU. Every estimated working set was admitted beneath the recorded 80%-of-currently-available-RAM allowance. No fallback occurred. Baseline records correctly have no policy device-admission record.
- OOD calibration contains 16 development-only samples (eight per case), 640 candidate evaluations, and 640 scenario power-flow calls. Logical calibration identity: `f90d93045cd31b918301ab801da701dc70b77ead151384c4f4080100628b485c`.
- Case30: baseline feasibility 0/10; candidate feasibility 0/10; no paired feasible objective observations; no objective inference is permitted.
- Case57: baseline feasibility 10/10; candidate feasibility 10/10; median paired relative objective improvement `0.011492392668353543`; 95% bootstrap CI `[-0.0019638316095621712, 0.01484160649928978]`; win rate `0.7`; rank-biserial effect `0.4`; one-sided Wilcoxon and Holm-adjusted `p=0.052734375`. The interval crosses zero and the multiplicity-controlled test misses the frozen `0.05` threshold.
- Final decision: `passed=false`, grade `U`, score `0.0`, claim scope `no qualification or policy-benefit claim`.
- Screening emitted no receipt and performed no registration or activation. The v2 candidate remains unqualified and inactive.
- Physical file SHA-256 values:
  - copied plan: `be297f2cd8584bc67ba14087085d9ed96eb69ebd7c8745983a4caa7e55fd9ef9`;
  - copied seed manifest: `dff66b39f9bec82c029b3508b9a15f0fd4153492551c44804d880e49f56d3e58`;
  - OOD calibration evidence: `a8c0c1fefcd918758e1ee2b8d8be7229b7fa1e2281256b9b4795cbf6015c3d55`;
  - qualification evidence: `039f2bfe31e39196e126da3961c65e4a248133ed09b009a93f64c933b2292778`;
  - stdout: `12d8cb400157e193bbbd28f98c082dcbc4046f05f02dd78a49527110d9099086`;
  - empty stderr: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Audit outcome: zero internal consistency errors. The campaign is valid negative screening evidence, not qualification evidence.
- Decision: do not launch formal qualification, do not open protected cases, do not register or activate the candidate, and do not weaken or reinterpret the preregistered thresholds.
- Synchronized `RELEASE_READY_CONTINUATION_HANDOFF.md`, `IMPLEMENTATION_GATES.md`, and `REQUIREMENT_TRACEABILITY.md` with the negative result, exact scientific statistics, evidence identity, prohibited claims, and next legal boundary.
- Corrected the G0 ledger to reflect the user-directed local `main` workflow and continued protection of `Docker_Build.txt`.
- Documentation validation: `git diff --check` passed; no documentation-test references exist for these ledgers.
- Next action: commit this logically scoped evidence ledger, then inspect the physical CPU/CUDA parity and memory-pressure gate without changing scientific semantics.

### 2026-08-04 — evidence-ledger commit and physical G2 audit start

- Committed the negative screening evidence and this continuation log as `867101e` (`docs(calo): record negative v2 qualification screen`). No push, merge, publication, or release occurred.
- Physical device discovery on this workstation observed one `NVIDIA GeForce RTX 4060 Laptop GPU`, CUDA UUID `GPU-6949bb11-4637-37cb-5038-18e7743453c8`, driver `581.86`, PyTorch `2.13.0+cu130`, CUDA runtime `13.0`, and one visible CUDA device.
- At the discovery sample, `torch.cuda.mem_get_info(0)` reported 7,441,743,872 free bytes of 8,585,216,000 total bytes; `psutil` reported 22,410,686,464 bytes of currently available system RAM. These are transient observations, not capacity or performance claims.
- Ran the existing fail-closed FP64 parity audit on development case30 only with physical `cuda:0`, seed `20260804`, 12 seeded random candidates plus deterministic boundary/corner probes (27 candidates total), and batch size 32. Result: pass; zero feasibility, convergence, bus-type, or scenario-count mismatches; maximum objective error `2.2515322939398175e-13`; violation error `5.882427878134422e-09`; voltage error `3.552713678800501e-15` p.u.; angle error `1.723066134218243e-13` degrees. Frozen tolerances were `1e-5`, `1e-6`, `1e-5`, and `1e-4` respectively.
- Ran the identical audit on development case57. Result: **failed**. Top-level feasibility/convergence/bus-type/scenario counts matched, but probes 5, 20, and 23 had large state divergence; probe 25 also produced non-finite component disagreement. Aggregate maxima were infinite component error, `0.4017950186797832` p.u. voltage error, and `84.28098606173475` degrees angle error. No CPU/CUDA-equivalence claim is permitted.
- Protected case118/case300 assets remain unopened.
- Diagnosis: all four failed probes were jointly nonconverged and both backends returned the same authoritative rejection (`value=inf`, `violation=inf`, infeasible). The device-resident path represented unused post-failure constraints as additional infinities and computed `inf-inf` for an objective standard deviation; the parity audit also compared discarded terminal Newton iterates after both solvers had rejected the case.
- Corrected the device-resident failure record to preserve a single authoritative `power_flow=inf`, zero unused derived constraints, and canonical infinite objective mean/standard deviation/robust value. This does not alter feasible or converged evaluations, evaluation counts, thresholds, or solver tolerances.
- Corrected the parity contract to require identical convergence outcomes and fail-closed evaluation semantics, count jointly nonconverged scenarios explicitly, and compare voltage/angle/bus-type state only when both solvers converged. A one-sided convergence result still fails closed. Converged-state tolerances are unchanged.
- Added a deterministic IEEE 57 alternating-boundary regression proving canonical failure metadata, joint-nonconvergence accounting, and zero comparison of discarded iterates.
- Focused parity/scientific validation: `25 passed`.
- Corrected physical case57 rerun on `cuda:0`: pass across 27 candidates; 23 converged scenarios compared; four jointly nonconverged scenarios; zero feasibility/convergence/bus-type/scenario-count mismatches; maxima: objective/component `7.815970093361102e-13`, violation/constraint `9.734435479913373e-13`, voltage `5.329070518200751e-15` p.u., angle `2.3803181647963356e-13` degrees.
- Complete active tree excluding only the deliberately stale v6.9 release-integrity file: `583 passed, 63 skipped`.
- Repository Ruff check: pass. Repository Ruff format check: 406 files already formatted.
- Committed the parity correction as `7a712c5` (`fix(accelerated): canonicalize nonconverged parity`).
- Extended `validate_accelerator` with an optional new-file-only durable evidence mode. It requires an explicit run ID and clean tracked Git source, records the exact source commit, strict-JSON parity report, parameters, UTC bounds, Python/platform/PyTorch/CUDA identity, pre/post currently available CPU RAM and CUDA VRAM, immutable 80%-of-currently-available admission calculations, physical-CUDA requirement/result, peak allocated/reserved CUDA memory, and a narrowly worded claim scope. It refuses overwrite and does not inspect protected cases implicitly.
- Added three evidence-writer/runtime tests covering strict non-finite JSON encoding, no-overwrite behavior, and the 80%-of-currently-available CPU RAM bound. Focused evidence/parity set: `12 passed`; focused Ruff check/format: pass.
- Commit interruption: the attempt to create `feat(validation): retain physical parity evidence` was rejected before Git ran because the Codex execution-credit limit was reached. No staging or commit occurred, and no bypass was attempted.
- Interrupted working-tree checkpoint: HEAD remains `7a712c579f424ccf5c761f4defb57db7eba824f3`; tracked modifications are `calo_rpd_studio/scripts/validate_accelerator.py` and this log; untracked repository work is `tests/unit/test_accelerator_evidence.py`; user-owned `Docker_Build.txt` remains untracked and untouched. `git diff --check` passes.
- Exact resume action after execution credit is available: stage only `calo_rpd_studio/scripts/validate_accelerator.py`, `tests/unit/test_accelerator_evidence.py`, and this log; verify the cached diff; commit as `feat(validation): retain physical parity evidence`. Do not stage `Docker_Build.txt`.
- After that clean commit, run case30/case57 physical audits into new artifact identities and audit their hashes/contents before memory-pressure/fallback and soak work.

### 2026-08-04 — release-readiness continuation resumed

- The user created a durable goal to continue all remaining non-tuning engineering, physical-validation, container/CI, packaging, traceability, and release gates until the repository is genuinely release-ready. Algorithm tuning, weakened scientific criteria, premature protected-case opening, fabricated evidence, push, merge, publication, and release remain out of scope unless separately authorized.
- Re-read the root, package, script, test, unit-test, documentation, and implementation-record instructions. The interrupted working-tree checkpoint matched the prior log exactly; `Docker_Build.txt` remains user-owned, untracked, and untouched.
- Revalidated the pending accelerator-evidence implementation with `9 passed` across `test_accelerator_evidence.py` and `test_v33_cuda_resident.py`; focused Ruff lint and format checks pass.
- Complete active tree excluding only the deliberately stale `tests/unit/test_v690_release_integrity.py`: `586 passed, 63 skipped` in 127.91 seconds.
- Next action: commit only the pending accelerator evidence writer, its focused test, and this continuation log, then run new clean-commit physical case30/case57 evidence audits before memory-pressure, fallback, lease, recovery, and soak validation.

### 2026-08-04 — clean-commit physical parity retained and G2 resource harness started

- Committed the durable parity-evidence writer as `63f56ad` (`feat(validation): retain physical parity evidence`). No push, merge, publication, or release occurred.
- Ran new physical FP64 parity audits from clean source `63f56adb4cf36e15210088eed92ff5325f76b02d` on the observed NVIDIA GeForce RTX 4060 Laptop GPU, development cases only, seed `20260804`, 12 random candidates plus deterministic boundary/corner probes, batch size 32, and explicit physical-CUDA requirement.
- Case30 evidence `artifacts/physical-accelerator-validation/g2-63f56ad-20260804/case30-parity.json` passed across 27 candidates with 27 converged scenarios, zero semantic mismatches, maximum objective error `2.2515322939398175e-13`, violation error `5.882427878134422e-09`, voltage error `3.552713678800501e-15` p.u., and angle error `1.723066134218243e-13` degrees. SHA-256: `20f1f0da3b837e54d071359ac375b419fe0f750192bd290efdaa5edebba15b53`.
- Case57 evidence `artifacts/physical-accelerator-validation/g2-63f56ad-20260804/case57-parity.json` passed across 27 candidates with 23 converged and four jointly nonconverged scenarios, zero semantic mismatches, maximum objective error `7.815970093361102e-13`, violation error `9.734435479913373e-13`, voltage error `5.329070518200751e-15` p.u., and angle error `2.3803181647963356e-13` degrees. SHA-256: `45ad11f4fff6bd045f4e8c9575bc19b818929f8a17768b2f4259dbfff07fd8e0`.
- Both evidence records bind the exact clean source, physical device, PyTorch/CUDA runtime, live CPU-RAM/CUDA-VRAM samples, and peak CUDA allocation/reservation. Each peak remained within its recorded 80%-of-currently-free-VRAM allowance. Claim scope remains evaluator parity for these development cases/candidates/source/device only.
- Added a non-tuning physical resource-recovery validation harness with a new `calo-rpd-resource-validate` entry point. It retains bounded real VRAM pressure/recovery, actual host-staged CUDA execution, controlled and explicitly labelled OOM microbatch backoff, controlled clean CPU restart plus CUDA recovery, and cross-process exclusive device-lease evidence. Controlled faults cannot be reported as naturally occurring hardware OOM evidence; protected cases are excluded by the CLI.
- Focused resource/evidence/VRAM suite: `22 passed`; focused Ruff lint and format pass.
- Complete active tree excluding only the deliberately stale `tests/unit/test_v690_release_integrity.py`: `593 passed, 63 skipped` in 127.17 seconds. Repository Ruff lint passes and all 409 Python files are formatted.
- Next action: commit the resource harness and synchronized ledgers, then execute the harness from that clean commit before starting the physical soak gate.

### 2026-08-04 — physical resource recovery retained and soak evidence hardened

- Committed the physical resource-recovery harness and synchronized ledgers as `d6a950c` (`feat(validation): retain physical resource recovery evidence`). No push, merge, publication, or release occurred.
- Ran `calo-rpd-resource-validate` from clean source `d6a950c519b6e3d586f546c60d97302ea3cd56a0` with explicit RTX 4060 `cuda:0`, development case30, eight candidates, batch size eight, seed `20260804`, 5% bounded pressure with a 256 MiB absolute maximum, and a 64 MiB recovery tolerance.
- The 268,435,456-byte physical allocation was observed as a 281,018,368-byte free-VRAM reduction. Free VRAM sampled 7,441,743,872 bytes before, 7,160,725,504 during, and 7,429,160,960 after cleanup; recovery passed. The 80%-of-currently-free allowances decreased from 5,953,395,097 to 5,728,580,403 bytes during pressure.
- Actual CPU-host input staging completed on CUDA with `cpu_inner_loop_participation=false`. The controlled OOM probe attempted microbatches `5,2,2,1`, recorded one retry and no CPU fallback, and completed numerically. The controlled typed-capacity probe performed the required full-request CPU reference restart and then recovered CUDA execution. These two faults are explicitly injected state-machine evidence and are not represented as naturally occurring hardware OOM.
- Cross-process lease evidence refused the contender while `cuda:0` was owned and admitted it after release. Overall physical resource qualification passed. Evidence path: `artifacts/physical-resource-validation/g2-d6a950c-20260804/resource-recovery-case30.json`; SHA-256: `391180a00e721ce028bcd09141e260e0fcf7ccd5c3af23e124fbf9d513d6d89f`.
- A short eight-second pre-change CUDA soak diagnostic confirmed `nvidia-smi + PyTorch CUDA` temperature and GPU board-power telemetry are available on this host; CPU temperature and GPU power-limit telemetry are unavailable and remain unfilled.
- Hardened the soak runner without changing algorithm or experiment semantics: optional clean source/run binding, strict physical-CUDA requirement, one-hour minimum that cannot be lowered for physical qualification, exclusive device lease for the soak duration, new-file-only evidence, verified hash-chained provenance, observed-only sensor summaries, and trapezoidal GPU-board-energy integration with an explicit exclusion of CPU/display/PSU/battery/whole-system energy.
- The hardened runner passed `34` focused soak/governor/resource tests and a five-second physical CUDA smoke run. The smoke remained correctly `physical_qualified=false` because it was below one hour; its provenance chain verified and it reported only observed telemetry.
- Complete active tree excluding only the deliberately stale `tests/unit/test_v690_release_integrity.py`: `594 passed, 63 skipped` in 125.80 seconds. Repository Ruff lint passes and all 409 Python files are formatted.
- Next action: commit the hardened soak evidence implementation, then start the clean-commit one-hour physical CUDA soak.

### 2026-08-04 — one-hour physical CUDA soak retained

- Committed the hardened soak-evidence implementation as `67bd18e` (`feat(validation): bind physical soak evidence`). No push, merge, publication, or release occurred.
- Ran the bounded one-hour physical CUDA soak from clean source `67bd18ea704a4614e282b7bda3e2d29a28273d99` with run ID `g2-67bd18e-rtx4060-soak-20260804`, explicit `cuda:0`, one-second sampling, a 3,600-second physical-qualification minimum, and the exclusive CUDA device lease held for the complete run.
- The run completed in `3600.000156299968` seconds with 3,600 samples, terminal state `GREEN`, no non-GREEN sample, no safe-stop request, no protection stop, and `physical_qualified=true`. Independent verification checked all 3,602 hash-chained provenance events and reproduced tail hash `71b89e4725c3fe4029ac8a7534d3d59d65fc3a1defd2f0db5b0cde8344283427`.
- Observed RTX 4060 telemetry over the retained run: temperature 46–60 °C, mean `56.934444444444445` °C; GPU board power 12.18–26.0 W, mean `24.341283333333333` W; utilization 0–60%, mean `49.08972222222222`%; and memory 1.6731802638006839–1.880801172447484%, mean `1.6732379362753085`%. CPU temperature and GPU power-limit telemetry were unavailable and remain null.
- Trapezoidal integration covered `3599.380295799987` seconds of observed GPU board-power samples and yielded `87619.64859865257` J (`24.33879127740349` Wh). This is GPU-board-energy evidence only and explicitly excludes CPU, display, PSU conversion, battery, and whole-system energy.
- Retained artifacts: result `artifacts/physical-soak-validation/g2-67bd18e-20260804/soak_g2-67bd18e-rtx4060-soak-20260804.json` SHA-256 `49b805c3019dadc2c97cafcff230b84c29c15ffd18f2bf5e54d5364edfa30800`; provenance JSONL SHA-256 `4a44f4f37821f64d4affb861acb047177bb1f6af9e8671c481bf0007a96d75f4`; stdout SHA-256 `2540b95883615629e2a2513c3216eb8d0f069cf094cd15d8778b4e6136b555fa`; empty stderr SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- G2 host parity, bounded resource recovery, lease behavior, and one-hour soak evidence are now retained. Container/WSL2 repetition, source-bound image builds, cross-container lease proof, whole-system energy, and any performance or scientific-benefit claim remain separate.
- Docker Desktop is now available on this workstation with a running linux/amd64 BuildKit builder and ample disk capacity. Existing user images, volumes, containers, caches, and `Docker_Build.txt` remain untouched. The next action is to close container build-context privacy, immutable source identity, shared-volume lease, runtime/CI evidence-retention, and local CPU/CUDA image qualification gaps before building commit-scoped images.

### 2026-08-04 — container privacy, source identity, and shared-lease hardening

- Audited the actual Docker build context and found generated trained-model/checkpoint/lineage files plus the user-owned `Docker_Build.txt` were not excluded even though image source is copied recursively. Hardened `.dockerignore` to exclude generated policy/checkpoint/lineage data, publication/results/temp/coverage/egg-info outputs, and the user file while retaining only the trained-model package marker.
- Added deterministic immutable build-source declarations with strict schema/40-hex validation, an explicit unavailable development placeholder, new-file-only writes, and a durable-evidence eligibility check. Runtime resolution always prefers live Git truth, so a clean image declaration cannot bypass a dirty mounted checkout; it falls back to `/opt/calo/.calo-source-identity.json` only outside a Git worktree. Durable parity, resource, soak, and container-smoke paths now record the identity kind and fail closed unless the commit is full and tracked source is clean.
- Docker builds now accept `SOURCE_COMMIT`/`SOURCE_TRACKED_CLEAN`, retain the declaration plus OCI revision/license metadata, copy `LICENSE`, and run `pip check`. Compose supports commit-scoped CPU/CUDA tags and isolated qualification volumes without overwriting existing resources.
- Moved the default container device-lease directory from private `/tmp` to `/data/device-leases` on the shared named volume, closing the static cross-container over-admission gap. Hardened x11vnc to container-loopback only while websockify remains the local bridge.
- Strengthened CI so both CPU and CUDA images are loaded under exact-commit tags, receive full source build arguments, retain build metadata/digests/image inspection, emit CycloneDX plus complete JSON vulnerability reports, and fail on fixable high/critical findings. The manual physical lane now retains environment identity, durable case30/case57 parity, resource recovery, a genuine one-hour physical soak, cancellation, and always-uploaded artifacts; its timeout is 180 minutes.
- Added five focused source-identity tests plus expanded container contracts for build-context privacy, source arguments, the shared lease location/volume, LICENSE/pip-check, and loopback-only VNC. Focused source/container/evidence/soak/resource/lease validation: `44 passed`. Both Compose profiles validate, repository diff check passes, and `docker buildx build --check` reports no warnings.
- Complete active tree excluding only the deliberately stale `tests/unit/test_v690_release_integrity.py`: `601 passed, 63 skipped` in 114.08 seconds. Repository Ruff lint passes and all 411 Python files are formatted.
- Claim boundary: a build declaration is operator-supplied metadata corroborated by retained BuildKit provenance, not a signature or independently trusted attestation. No image has been built from this change yet, and no runtime, vulnerability, GUI, WSL2, performance, or scientific-benefit evidence is claimed.
- Next action: commit only this hardening set and synchronized ledgers, then build uniquely tagged CPU/CUDA images from the resulting clean commit and retain all evidence without changing existing user Docker resources.

### 2026-08-04 — first source-bound CPU image rejected by filesystem audit

- Committed the initial container hardening as `3b43c74` (`feat(container): bind source and shared device leases`). The default Docker driver then correctly refused an attested `--load` build because that exporter cannot preserve attestations; it created no image tag. Created the isolated `calo-rpd-g4-3b43c74` Buildx container builder and separated the attested OCI export from the cache-identical runtime-loaded image instead of dropping provenance silently.
- The attested CPU build succeeded from the exact 9.00 MB filtered context, passed `pip check`, generated BuildKit SBOM/provenance, and exported OCI index digest `sha256:dd1f9f89caf42454e6abb8710a17088211acbe079618fd8c71696b203c6f36c6` with attestation manifest `sha256:d4e2bfa48c9e32d9770eaedce07108434b35c8dbc4689e38c2cd56a36e1f15d7`. The OCI archive SHA-256 is `ecea77eeaa5588ba4342531e43b0ec6b42da7db75d80f267d14711211fd1d9b5`.
- Runtime image `calo-rpd-studio:cpu-3b43c74` passed CPU-only, non-root UID 10001, read-only root, writable data, schema/config/database round-trip, and durable source identity for full clean commit `3b43c74c14dee4cc825a8ff6614a28421e815665`.
- The mandatory image-filesystem audit then found 490 recursively nested `__pycache__`/`.pyc` files, including stale CPython 3.13/3.14 bytecode, because the prior root-only Docker ignore pattern did not exclude nested directories. Generated `.pt` files, the user-owned `Docker_Build.txt`, and extra trained-model data were absent as intended.
- Decision: reject this image as a release candidate despite its successful build/smoke. Harden recursive bytecode and repository-instruction exclusions, add static assertions, commit a new clean source, and rebuild both images under new commit-scoped identities. Do not reuse `3b43c74` image evidence as release qualification.

### 2026-08-04 — corrected CPU image passes privacy but fails vulnerability gate

- Committed recursive context exclusions as `698407d` (`fix(container): exclude recursive build residue`) and rebuilt a distinct source-bound CPU artifact. Its context transfer fell to 23.36 kB on the cached build boundary. Attested OCI index digest is `sha256:53788944b05cf281469a401b9a1a0eb80f003f69f8e3b41319e25f9c8bd9e722`; attestation manifest is `sha256:232f6ea6d63f0a04bfa91f96c4578e85e5f873d130d80bf957975b1df90016c2`; OCI archive SHA-256 is `8535575d0231dba18fb8b0a4dceaee3ffaf85d8c337da89ecc1ac65cc1b6fe86`.
- Runtime image `calo-rpd-studio:cpu-698407d` passed durable source identity for full clean commit `698407dc89550cc8644612651726c0b0943b64aa`, CPU-only execution, UID 10001, read-only root, writable data, database/schema/config round-trip, and image-manifest generation. The 336-file privacy audit passed with zero nested bytecode/instruction files, checkpoints, policy hashes, user build notes, or unexpected trained-model files.
- Docker Scout execution was rejected before invocation because it can disclose image-derived metadata to an external Docker service; no bypass was attempted. Pulled official Trivy 0.70.0 locally at immutable scanner digest `sha256:be1190afcb28352bfddc4ddeb71470835d16462af68d310f9f4bca710961a41e`, saved the CPU image locally, and used a dedicated database cache without uploading the CALO image.
- The first local scan downloaded the 103.35 MiB database but hit Trivy's default five-minute analysis timeout and produced no conclusion. A documented 15-minute rerun completed CycloneDX and full JSON output. It found 708 advisories total, including 23 critical and 130 high under upstream/vendor severity sources; the frozen release gate failed on exactly two high findings with fixes.
- Both fixable findings are vendored inside runtime-unneeded `setuptools` 79.0.1: `jaraco.context` 5.3.0 / CVE-2026-23949 (fixed 6.1.0) and vendored `wheel` 0.45.1 / CVE-2026-24049 (fixed 0.46.2). The separately installed `wheel` is already 0.46.3; the vulnerable copy is the setuptools vendor bundle.
- Decision: reject `cpu-698407d` for release. Preserve hash-locked installation and `pip check`, then remove runtime-unneeded `pip`, `setuptools`, and `wheel` build tools from the final layer, verify application imports/smoke, rebuild from a new clean commit, and rerun the unchanged local security gate. No allowlist, severity downgrade, ignored fixed finding, or algorithm change is permitted.

### 2026-08-04 — source-bound CPU/CUDA image gates retained; container soak running

- Committed removal of runtime packaging tools as `1f02a94` (`fix(container): remove runtime packaging tools`). A disposable build-stage probe removed `pip`, `setuptools`, and `wheel` and then imported the core numerical, PyQt, PYPOWER, CMA and CALO application modules successfully. The image build itself still performs hash-locked installation and `pip check` before removal.
- CPU attested OCI index digest: `sha256:463be6ab66d55721409aeb51dffac1a1e18d3d11d86a18814a0757044a2b5517`; attestation manifest: `sha256:95dd93d45e7921aa801ea2fe3b35cdabef211484ede8a1128a06848fc6897b18`; OCI archive SHA-256: `f7ca430797d335586404a3b1b7def99414a6b7b76cb4b50e0d6ada7d9783d662`; loaded runtime image/config identity: `sha256:32617d700f2c0401f89d5a20e7c60798c79a0c7e50d241311480610b9e271484`.
- CPU runtime `calo-rpd-studio:cpu-1f02a94` passed CPU-only visibility, UID 10001, read-only root, writable data, schema/config/database round-trip, full clean build-declared source `1f02a94ba4c13484bcd48c740fde6981cf0354ac`, and the 336-file zero-forbidden-file privacy audit. The pinned local Trivy rerun retained CycloneDX and complete JSON with 700 advisories, 23 critical and 128 high by recorded upstream/vendor severity, but zero HIGH/CRITICAL findings with a nonempty fixed version; the unchanged release gate passed.
- CUDA attested OCI index digest: `sha256:218ec1cc494eb6249315e32c8a63d35e98c5c78c06ee921f3a8e682f8a006fbe`; attestation manifest: `sha256:d003abbe0d906d336d763d6dfcda7a56effde153bfdcbcee4aba249625c7eae0`; OCI archive SHA-256: `5688d4043f18f76098f54aaf936bab462d26af503aa440362ce7fb4f95f50285`; loaded runtime image/config identity: `sha256:be9a836c66709ffb8fadd4c24450df572c2d5999e43e9038713eabaa34dc0db0`.
- CUDA runtime `calo-rpd-studio:cuda-1f02a94` passed UID/read-only/source/privacy/schema smoke on exactly one physical NVIDIA GeForce RTX 4060 Laptop GPU with PyTorch `2.10.0+cu128` and CUDA runtime `12.8`. Its 336-file privacy audit passed. Pinned local Trivy retained the complete same-size advisory inventory and passed with zero fixable HIGH/CRITICAL findings. These counts are scanner observations, not a claim that unfixable advisories are harmless.
- Physical shared-volume device-leasing passed across independent containers: holder ready, contender exited 1 with `DeviceLeaseUnavailable`, and a new container acquired the lease after holder release. Evidence SHA-256: `c13946dc5799054e64a90eef4bc8f0ce797646ceb6b445a0de6c37c2d8e9cdcf`.
- Durable physical validation inside the exact CUDA image passed for case30 parity (SHA-256 `11a5754a8b97403d7be7fbff71f1d37c2b97dde0a18226b999f979857631fe17`), case57 parity (`01a2eb9a28316385a7b09b12b244b2a1c84d1430268f0298c08545430b359a46`), and bounded resource recovery (`5507116b0851d1ce000bebce237bcf1cf12020b8a0cbf2e27c7bc459cb693e77`). All records use `source_identity_kind=build-declared` for the full clean image commit. Direct validator invocations emitted a nonsemantic missing `/tmp/calo-cache` warning; the one-hour command explicitly creates its writable cache/runtime directories before Python and currently has empty stderr.
- The exact CUDA image one-hour physical soak is now running with one-second samples, shared device lease, read-only root, retained hash-chain evidence, and no concurrent GPU workload. It remains pending until the full 3,600-second duration, protection state, source identity, telemetry summary, independent chain verification, and hashes pass.

### 2026-08-04 — counted Change E completed; suspended container soak rejected and rerun

- The first exact-image container soak was rejected rather than resumed as qualification. Its retained UTC chain contains 1,203 samples but a `3198.61298`-second host-suspend gap after sequence 974. `container-soak-rejected-host-suspend/REJECTED_ATTEMPT.json` marks `scientific_use_permitted=false`; the partial JSONL SHA-256 is `1a1a7ca4a9155b9cc6917dbf0c81479523ff4f64c23942738563836737797c5e`. No duration, continuity, or hardware-qualification claim is made from it.
- A uniquely named detached rerun now uses the same exact `calo-rpd-studio:cuda-1f02a94` config digest `sha256:be9a836c66709ffb8fadd4c24450df572c2d5999e43e9038713eabaa34dc0db0`, run ID `g4-1f02a94-container-soak-continuous-rerun1`, retained bind evidence, shared lease volume, and a temporary inspectable Windows keep-awake guard. It remains pending until 3,600 continuous active samples, independent chain/timestamp audit, result validation and hashes close.
- Implemented explicit counted AC linearization retention for optional Change E only. Ordinary evaluation and topology-only `evaluate_with_context` calls do not compute or retain it. An E-enabled immutable runtime/training design retains the final Newton Jacobian, analytic relaxed-control sensitivity, and analytic active voltage, angle, generator P/Q and branch-thermal constraint projection from the same already-counted converged solve. No second power flow or evaluator call is made; unavailable, switched-out, zero-residual, nonfinite, oversized or ill-conditioned contexts mask before policy action.
- Integrated the counted proposal-only operator into production and independent training paths with exact candidate-FE/scenario-call accounting, per-generation availability, linear-algebra timing, no-feasibility-authority provenance, exact resume and dynamic fail-closed masks. Change E remains disabled by default. Change F remains independently experimental and rejected by the fixed-population production/training paths.
- Fixed a prerelease trust-region defect found by a real development-case probe: snapping several discrete controls after the continuous bound could exceed the final norm radius. Whole lattice moves are now admitted only while they fit the radius; continuous controls receive only the remaining norm budget. The real case30 counted context produces a lattice-valid proposal within `0.08`, with zero hidden solver/evaluator calls and no feasibility claim.
- Versioned the changed candidate contracts as runtime `tsh-calo-v1.1.0-counted-physics-candidate`, training environment `tsh-calo-training-v4-counted-physics-safe80-receipts`, and environment checkpoint `tsh-calo-independent-environment-v2-counted-physics`. Earlier candidate v2 evidence remains immutable historical negative evidence and is not reinterpreted under the new ABI.
- Direct evidence: 18 counted-linearization/repair tests, 46 combined runtime/training/campaign/session tests, and the complete TSH-CALO plus counted-ORPD unit family `136 passed, 481 deselected`. Case30 analytic control sensitivity matched central power-flow finite differences across every relaxed control with maximum absolute error below `1.6e-8`. The complete active tree excluding only the deliberately stale v6.9 release-integrity file passes `608 passed, 63 skipped` in 125.20 seconds. Repository Ruff lint passes and all 411 Python files are formatted.

### 2026-08-04 — safe PGLib source boundary and non-protected validation corpus implemented

- Implemented a fail-closed PGLib import boundary that reads local bytes only, verifies a strict
  source-manifest SHA-256 and exact asset SHA-256, accepts only the official repository, pinned
  `vYY.MM` release plus 40-hex commit, declared typical/API/SAD path and role, retained
  CC-BY-4.0 attribution, and parses a restricted literal MATPOWER v2 grammar without `eval`, MATLAB,
  Python import, network access, or executable expressions. The known numeric `mpc.areas` field is
  accepted but explicitly recorded as unused; arbitrary fields remain rejected.
- Retained unmodified official PGLib-OPF v23.07 case14 typical/API/SAD validation assets from tag
  commit `dc6be4b2f85ca0e776952ec22cbd4c22396ea5a3`. Asset SHA-256 values are respectively
  `bd5c568621de65e4b0922317010868bc7fa94173807faa10ea8fdbbe77c28106`,
  `6e007be95df3f7171d0c9494c8cc3db1aca1a3a0f2073c3ffbfa43e7b0cd49a2`, and
  `f79873ebac589619c45540e7b1020c1be1eae13f19b47af0925042c6a01f8f0a`. The exact upstream
  license SHA-256 is `95b1cd9fee1676221d74f7c0cbba622d98ac098e9b317b3848113ef4356ab4fd`.
- All three real assets passed structural/physical case validation as 14-bus, five-generator,
  20-branch cases. Their distinct physical checksums are typical
  `14488f24f83576bfb80179434f27ae036ccec4e3ba69000cb3ec45ed8d3376d2`, API
  `d845a7205808a982edac7cccae71b768872bdc5b76ad212e3bc116af45a1c421`, and SAD
  `e2e92121499920b0048c920d9db7c2375e2014d7b5f96c7e0f04f43c0d56e62e`.
- AC-OPF import does not create an ORPD formulation. Added a separate strict, checksum-loaded,
  `review_status=reviewed` profile contract that binds explicit generator-voltage bus numbers,
  transformer branch indices and shunt definitions to the exact source asset and physical case.
  Invalid/dead controls fail closed. Both protected source import and protected ORPD conversion
  require explicit test-only authorization. No profile is trained, inferred, auto-reviewed or
  fabricated; actual external profiles still require independent human/domain review.
- Preserved bundled IEEE semantics: case provenance is absent from legacy serialization and
  fingerprints when not applicable. Independent comparison against an archived pre-change HEAD
  produced identical default scientific fingerprints for case30
  (`d94f33c8e5e7c00489f2811a2ab5a8df64cfe2db766278bb84024ba0da32a646`) and case57
  (`532db81bf01aeaa4064903ecd01526ea550ddd2ac67dbe4ae18055cd8ebecb67`).
- Focused import/config/scientific/ORPD regression set: `35 passed`; focused Ruff and generated-
  schema checks pass. A fresh development wheel/sdist build included eight PGLib data/license/
  manifest entries; loading all three cases from extracted wheel contents reproduced the exact
  physical checksums. Dirty-tree packaging-smoke hashes are wheel
  `71fda5b7bb595482b60e95d74397331a8f86dbb89425cbe31eb7e5e08af84fef` and sdist
  `47083a089d8ca48d796c4f310e6da82c1f9b97236cc8085a71f4fbce081a413d`; they are development
  evidence only and are not release artifacts.
- The complete active tree excluding only the deliberately stale v6.9 release-integrity file
  passes `615 passed, 63 skipped` in 125.63 seconds. Repository Ruff lint passes, all 414 Python
  files are formatted, the generated experiment schema is current, and `git diff --check` passes.
- The continuous exact-CUDA-image soak remained GREEN with no safe-stop while this independent CPU
  development proceeded. It remains pending until the complete continuous hour and independent
  chain/timestamp/result audit close; no qualification conclusion is recorded early.
- Next action after committing this non-tuning G10 boundary: implement disclosed mathematical
  reference-solver adapters and their separation of continuous-relaxation bounds from feasible
  mixed-variable ORPD solutions while the isolated container soak completes.

### 2026-08-04 — continuous exact-image soak accepted; first GUI image rejected

- Committed the safe PGLib boundary and non-protected corpus as `39ad382`
  (`feat(scientific): add verified PGLib import boundary`). The tracked tree then returned clean;
  only user-owned `Docker_Build.txt` remained untracked and untouched.
- Exact CUDA image `calo-rpd-studio:cuda-1f02a94` config digest
  `sha256:be9a836c66709ffb8fadd4c24450df572c2d5999e43e9038713eabaa34dc0db0`
  completed the continuous rerun with container exit 0, `3600.000632077` seconds, 3,600 GREEN
  governor samples, zero non-GREEN states, zero safe-stop requests and no protection stop. The
  result is `physical_qualified=true` for this exact source/image/device/duration scope only.
- Independent audit reproduced all 3,602 contiguous events (one start, 3,600 samples, one terminal),
  one session identity and tail hash
  `e00f680688ba366cf823a2d1984d20be27028e48583059d656e00ea4682cf815`.
  Maximum UTC and monotonic inter-sample gaps were `1.105776` and `1.0012514019999799` seconds;
  the earlier suspend gap did not recur. Result SHA-256 is
  `aab8b13c7e01a27260e7ca0934ac472e0e845103c0c07356d9468f031bb391b5`; JSONL SHA-256 is
  `eb3d5d6a3e03c259ea02b7c4e832a7acfbb74258bbc8f5bba9613ac6c5eca15c`.
- Observed scoped telemetry: RTX 4060 temperature 46–59 °C, board power 10.72–25.22 W, utilization
  0–55%, and 3,600 available samples for each. Trapezoidal board-power integration covered
  `3599.376110686` seconds and yielded `23.769249787905427` Wh. CPU temperature, GPU power limit,
  CPU/display/PSU/battery and whole-system energy remain outside the evidence. The temporary
  keep-awake process was command-line identity checked and stopped after verification.
- Started isolated CPU image `calo-rpd-studio:cpu-1f02a94` on localhost port 16080 with its own
  retained volume. The GUI gate rejected it before browser acceptance: the Qt application exited
  because `libxcb-shape.so.0` was missing. Direct `ldd` and `QT_DEBUG_PLUGINS=1` QApplication probes
  reproduced that exact missing library; `libxcb-cursor0` itself was installed. Compose restart
  count reached one while the old web-page-only health probe reported healthy during the restart
  window. The rejected container was stopped with timeout 20 and retained rather than treated as
  GUI/restart evidence.
- Corrected source adds Debian `libxcb-shape0`, a build-time `ctypes.CDLL` load of PyQt6
  `libqxcb.so` to verify the complete shared-library closure, and a health probe that requires both
  the noVNC endpoint and a live Qt PID. The launcher now publishes readiness only after every
  desktop child remains stable and exits fail-closed if Xvfb, Openbox, x11vnc, websockify or the Qt
  app exits. Focused contract/supervision tests: `10 passed`; Ruff, both Compose profiles and
  `docker buildx build --check` pass with no warnings.
- Next action: commit the GUI dependency/health correction, rebuild an exact clean CPU image, rerun
  direct QApplication plus in-app browser GUI interaction, persistence, restart and graceful
  cancellation. Do not accept or reuse the rejected `cpu-1f02a94` GUI attempt.

### 2026-08-04 — corrected source-bound Linux GUI runtime accepted

- Committed the fail-closed Qt dependency/health correction as `31a4713`
  (`fix(container): require live Qt application health`). From that clean tracked commit, built
  `calo-rpd-studio:cpu-31a4713` as image/config identity
  `sha256:f241c14c69d7896833e5805090d495f4ea14299de585cfb238ea13527b0deb5b`.
  Build-time `ctypes.CDLL` loading of PyQt6's `platforms/libqxcb.so` passed after installing
  `libxcb-shape0`; `pip check` also passed before runtime packaging tools were removed. The embedded
  source declaration is the full commit `31a47136d2a6f497ec6da6107a9623d243b67654` with
  `tracked_source_clean=true`.
- Started the exact image in isolated Compose project `calo-rpd-gui-31a4713` on loopback port 16080
  with retained named volume `calo-rpd-gui-31a4713-runtime`. Health reached GREEN only with both the
  noVNC endpoint and live `/tmp/calo-app.pid`; restart count remained zero. Independent
  `QApplication` construction reported platform `xcb`. Direct Qt screen capture showed the real
  1600x1000 CALO-RPD Dashboard with system protection READY, policy correctly NOT READY and gated
  downstream workflow.
- Retained persistence marker SHA-256
  `8ddd6fb1d67b6840d1b9a9887f2c0a522ad1de4696760d872a2461eedf7ea6c3` before and after an explicit
  container restart. The supervised application returned on PID 40 after PID 39, with a distinct
  process start tick, healthy state, zero automatic restart count and no OOM. Stable pre/post-restart
  renders are byte-identical at SHA-256
  `28108327353d3a491f8d92daf3f081d3e8bfb8b8a0d53bd9540d1a2484025187`; the 192,662-byte PNGs are
  retained under `artifacts/container-validation/g4-31a4713-20260804/cpu-gui/`.
- The desktop Browser integration could not initialize: its browser-control kernel failed with a
  missing-path bootstrap error before even a minimal call executed. This is recorded as a tooling
  limitation; no browser interaction claim is made. Direct X11/Qt rendering, live-process health,
  local noVNC endpoint health, restart, volume persistence and bounded cancellation remain valid
  independent observations.
- Compose stop with a 20-second bound completed. Final container state was exited, exit code 143,
  `OOMKilled=false`, restart count zero; the evidence volume and stopped container remain retained.
  The earlier rejected `cpu-1f02a94` GUI attempt remains rejected and is not reused.
- Next non-tuning development action: implement disclosed mathematical reference-solver adapters
  that distinguish continuous-relaxation bounds from feasible mixed-variable ORPD solutions. G4/G6
  still require browser interaction when tooling is available and full immutable final-candidate/CI
  repetition; this clean GUI runtime is development qualification, not a release image.

### 2026-08-04 — disclosed mathematical-reference boundary implemented

- Committed `07f9476` (`feat(scientific): add mathematical reference adapters`). The new
  `calo-rpd-math-reference` path is outside the stochastic optimizer registry and never receives an
  artificial equal-FE budget. It requires clean full source identity, a hashed frozen
  ExperimentConfig and, for SLSQP, a hashed explicit start vector; it refuses protected case118/300,
  runs independent PYPOWER checks by default and writes new-file-only JSON evidence.
- The SciPy adapter clones the exact task while removing only tap/shunt snapping, records installed
  backend/version, numerical derivative mode, settings, termination, iterations, backend calls,
  distinct common-evaluator calls/cache hits and independent-validation requests. It always labels
  the result a local nonconvex continuous-relaxation point, with
  `certified_lower_bound=null`, `optimality_gap=null` and `gap_claim_permitted=false`. The same
  normalized point is separately decoded and evaluated on the original lattice and is called an
  incumbent only when feasible. Complex-step is rejected because the real common evaluator cannot
  preserve its imaginary perturbation.
- The exhaustive adapter accepts only genuinely all-discrete declared lattices, rejects continuous
  controls and candidate counts above an explicit ceiling, evaluates every Cartesian point with the
  common feasibility-first rule, and scopes any exactness only to that complete finite lattice. A
  no-feasible-point result is retained as a screen, not promoted to a physical infeasibility proof.
- Focused success/rejection/boundary/fallback/independent-validation suite: `11 passed`; combined
  ORPD/PGLib/config/IEEE/distribution regression set: `37 passed`. Complete active tree excluding only
  the deliberately stale v6.9 release-integrity file: `629 passed, 63 skipped` in 127.65 seconds.
  Repository Ruff passes and all 418 Python files are formatted.
- Clean-source case30 SLSQP development probe used commit
  `07f9476dae41eba27074ef82bc3309fa5b70088d`, task fingerprint
  `d94f33c8e5e7c00489f2811a2ab5a8df64cfe2db766278bb84024ba0da32a646`, relaxation fingerprint
  `eb1e70045a91aa252c07fe96e118bb1960fa4b2e5c8601fa62c4e4ea865f3f18`, config SHA
  `c05f2a5a96cd3fe22dc47c44e7981c5a43b58e23d7559df862d688507e9218ea` and start SHA
  `431ac5bd061eac7ceefce93a8b62d16aa9912ee680cc066097167b11acdc0496`. The deliberately bounded
  three-iteration run stopped at status 9/iteration limit after 63 distinct common solver
  evaluations, four derivative evaluations and three explicit validation evaluations. Relaxed and
  mixed points remained infeasible (violations `0.05900671777581266` and `0.0590648537434835`);
  the mixed controls were lattice-valid and both AC states passed independent PYPOWER comparison.
  Report SHA is `8be27e3bb467a78d524930422bafa372729c3527782e06803c949f04449763dc`;
  envelope-file SHA is `7d765109841d9ddec0d28860e2f94c160655ead7356ba485085bf073b75a7018`.
- A separate development-only case30 one-shunt task exhausted all six declared 0–5 MVAr points.
  It found no feasible evaluator point; best feasibility-first screen value used `Qsh@10=0.0`,
  objective `2.4438031297412985`, violation `0.08832537034242222`, and passed the independent
  PYPOWER state comparison. No exact optimum scope, lower bound or gap was emitted. Task SHA is
  `32b6f870172ac8a75d039fc8da453909edcf45a5ebb730de8ddf18f8cac3fe98`, report SHA is
  `abff7f42274c5f9ad347a2d1af67bf8a585c478de7655438cd503aac13ae4ee5`, and envelope-file SHA is
  `71ced2f2212f9db7c052e8cff4b7fb0c71f0d7aaf6504a111b52f223ea35ad72`.
- These probes validate implementation and honest negative-result handling only. No solver
  superiority, feasible case30 optimum, certified lower bound, imported PGLib ORPD result or
  protected-case evidence is claimed. Next action: close the independently reviewed external ORPD
  profile gap if review material exists; otherwise continue non-tuning release engineering with
  packaged Linux/CI and final-candidate reproducibility gates while keeping G9/G10 scientific
  campaign execution blocked on the absent fresh qualified candidate and design freeze.

### 2026-08-04 — installed-wheel Linux GUI lane implemented and exercised

- Found that the prior `headless-gui` job rendered from the checkout, so it could not prove the
  built wheel contained a runnable GUI. Committed `383e5bc`
  (`ci(gui): validate the installed wheel on Linux`). The wheel now contains
  `validate_packaged_gui.py`; the artifact job installs the built wheel, changes to `/tmp`, clears
  checkout `PYTHONPATH`, rejects package imports below `$GITHUB_WORKSPACE`, renders the Dashboard,
  retains evidence and only then generates the artifact manifest. The distribution verifier now
  requires the mathematical-reference module/CLI and packaged-GUI validator in both wheel and sdist.
- The validator checks the installed distribution/version and package path, Qt/Python/platform
  identity, initial Dashboard, all 16 non-empty sidebar labels, window dimensions, screenshot bytes,
  and the normal-view forbidden-language contract. It writes screenshot/report new-file-only and
  records `source_checkout_imported=false` only when a caller supplied and passed the forbidden-root
  check. Focused packaged/distribution/CI contracts: `11 passed`; combined GUI/package/CI set:
  `44 passed`. Complete active suite: `632 passed, 63 skipped` in 130.17 seconds; Ruff passes and all
  420 Python files are formatted.
- From clean commit `383e5bc`, built fresh development stage `python-dist-383e5bc`. Strict stage
  verification passed with one 349-member wheel and one 401-member sdist. Wheel SHA-256 is
  `27b94fecbf7ecdba85837c9c790d3d0d99a25f4bc07c62e2da73dc32f4e93479`; sdist SHA-256 is
  `274bcbc078ad41da638264f6bce68268352104c566ce1dd4cf7d8379ddf69d20`.
- Extracted that exact wheel into `/tmp/wheel` inside hardened, network-disabled, read-only Linux
  runtime image `calo-rpd-studio:cpu-31a4713`, UID/GID 10001 with all capabilities dropped and
  no-new-privileges. The validator imported `/tmp/wheel/calo_rpd_studio/__init__.py`, not image source
  `/opt/calo`, and rendered the real 1440x900 Dashboard. Evidence reports CPython 3.11.15, Linux
  WSL2/glibc 2.36, Qt 6.9.0, PyQt 6.9.1, offscreen platform, 16 workspaces, zero forbidden visible
  terms, and clean session shutdown. PNG size is 199,957 bytes with SHA
  `adc340f602011436ded5f321a55e5cb3855a8a0e1e50fe613032c1089789ca1f`; report SHA is
  `12e7ca4e5fb921b4c58d9d7434c87fb58fe8c77b3e3dbd6eca4717b674052181`.
- The first nested-shell invocation failed before application execution due Windows/Linux quoting;
  its empty attempt directory is not evidence. The fresh `rerun1` above is accepted. A supplemental
  direct Python check initially resolved `/opt/calo` because the image working directory contributes
  `sys.path[0]`; it was rejected, diagnosed and rerun with Docker working directory `/tmp`. The
  corrected wheel-only check loaded all three PGLib assets with exact physical checksums and exposed
  the packaged mathematical-reference CLI. No result from the shadowed import is used.
- Copied the accepted GUI evidence into the development distribution stage and generated a six-file
  manifest covering wheel, sdist, GUI JSON/PNG, SQLite and clean-session journal. Manifest SHA-256 is
  `8c47c03b42c63a09d747e922803a1b0ca812399dea320a1f19df45377deb1e4a`.
  The retained extraction volume is `calo-rpd-packaged-wheel-383e5bc`.
- This closes the local installed-wheel Linux GUI development gap only. It is not a final release
  artifact, was not installed on an independent clean machine, and does not replace actual GitHub
  Actions, interactive browser, final-candidate image/SBOM/attestation, G9 qualification, reviewed
  external ORPD profiles or protected G10 campaign evidence. Next safe action: audit and execute
  remaining CI/reproducibility gates that do not require tuning or protected-case opening.

### 2026-08-04 — CUDA-residency priority inserted before remaining release work

- Expanded the pinned mypy 1.20.2 safety boundary from nine to twelve release-critical modules by
  adding the mathematical-reference implementation/CLI and installed-wheel GUI validator. The
  first exact check found an unchecked `Any` return and an untyped PyYAML boundary; both were
  corrected without changing solver semantics. The repeated twelve-file check passed, and the
  focused regression set passed `20` tests before the user subsequently prohibited further tests
  for the CUDA-residency milestone. The full coverage attempt was interrupted and is not evidence.
- The user inserted a new engineering priority: eliminate recurring CPU↔CUDA traffic in the
  scientific hot paths, process power-system work in CUDA-resident windows centred on 100 function
  evaluations, retain policy-update work on CUDA across ten-epoch reporting windows, and target
  more than 95% CUDA share of steady-state accelerator-eligible numerical work. Startup, UI,
  SQLite, filesystem I/O, orchestration and final serialization remain CPU responsibilities and
  are excluded from that metric; no 100%-of-application CUDA claim is permitted.
- A live idle sample found no residual Python/pytest/coverage process and `nvidia-smi` reported the
  RTX 4060 Laptop GPU at 0 MiB application use and 0% compute/memory utilization. This is an idle
  observation only, not workload evidence and not proof of fallback.
- Audit found that the CUDA AC evaluator already uses a fixed-shape masked Newton loop with no
  per-iteration host early exit or history materialization, and the heterogeneous PPO learner keeps
  minibatches, losses and optimizer tensors on its selected CUDA device through each update block.
  Remaining recurring boundaries include host-origin population staging, construction-only NumPy
  duplicates, conversion of completed CUDA evaluation batches into Python `Evaluation` objects for
  optimizer decisions, rollout/environment coordination and per-epoch policy snapshot distribution.
- Began the non-tuning residency repair in `device_resident_orpd.py`: construction-only NumPy case
  arrays are discarded after CUDA tensor preparation, and a complete host-origin candidate request
  is now uploaded once and retained on CUDA when it fits the immutable 80%-of-currently-free-VRAM
  process ceiling. A typed CUDA OOM at that initial upload retains explicit host staging and the
  existing CUDA microbatch policy rather than silently entering a CPU scientific inner loop.
- Per explicit instruction, do not execute tests or benchmarks during this CUDA implementation
  milestone. Consequently the `>95%` threshold must remain an unmeasured acceptance target, not a
  result or release claim. Complete the residency/cadence code and static traceability first, report
  that implementation boundary to the user, and only resume profiling/tests and the broader release
  goal after the user permits the next phase.
- Completed the source-level power-system residency change. CUDA evaluators now discard seven
  construction-only NumPy duplicates after their device tensors exist, attempt one complete
  population upload, retain that population and every scientific output tensor on CUDA through the
  request, and perform one packed final population materialization. Explicit OOM staging remains a
  separately labelled capacity path under the immutable Safe-80 ceiling. Runtime residency metadata
  records a target of 100 evaluations per host boundary and zero CPU↔CUDA inner-loop transfers.
- Changed new experiment/training execution defaults from 64 to 100 candidates and changed automatic
  CUDA calibration candidates to `100/200/400`; the schema and both GUI surfaces are synchronized.
  This is an execution batching change, not an optimizer population-size or scientific-budget
  change. Historical immutable policy artifacts that record 64 remain unchanged.
- Completed the independent TSH-CALO PPO data-residency change. Each validated topology state and
  action is uploaded once before the PPO block and reused for every configured PPO epoch. Group
  masking/log-probability work is CUDA tensor work without availability scalar reads; stochastic
  group sampling deliberately retains its original three-group RNG call order. PPO loss and gradient
  metrics remain on CUDA and cross to CPU once as one packed vector after the complete configured
  PPO epoch block. Thus a ten-epoch configured block has one metrics transfer, not ten.
- Collapsed the policy/environment action boundary from separate group-operator, parameter,
  learner-operator, mask, log-probability and value transfers into one packed CUDA→CPU transfer.
  The regime scalar remains the required action-delivery boundary, and CPU-side action validation
  happens only after that boundary. Bootstrap values, environment transitions, UI/database work and
  durable checkpoint/export remain legitimate CPU responsibilities; they are not relabelled CUDA.
- Changed the independent training campaign from one full durable checkpoint per transition to a
  grouped durability window targeting 100 counted candidate evaluations. A status marker is written
  before the window; only a completed window receives an exact resume checkpoint. If a process stops
  inside the window, resume marks the campaign failed and refuses the identity, preventing partial
  uncommitted evaluations from being silently replayed as exact evidence.
- Strengthened `validate_accelerator` to schema v2. A CUDA parity result now fails qualification
  unless PyTorch peak allocation rises above its pre-workload baseline, allocator accounting is
  consistent, and additional allocation/reservation stay within the admission allowance. The record
  explicitly states that dedicated-VRAM evidence does not claim zero host-RAM use.
- No tests, imports, compilation, profiler, CUDA workload or benchmark were executed after these
  CUDA changes, per user instruction. `git diff --check` is clean and was used only as a source-text
  consistency inspection. Therefore source implementation is complete for this milestone, while
  runtime correctness, exact replay/parity, the `>95%` steady-state CUDA numerical-time threshold,
  and the requested 100,000-evaluation/1,000-epoch throughput remain unverified and cannot yet be
  release claims. Do not commit or resume the broader release goal until the user authorizes the
  verification phase. User-owned untracked `Docker_Build.txt` remains untouched.

### 2026-08-04 — immediate development, step 1: bounded CUDA timing evidence

- Added a reusable, fail-closed CUDA event/wall-clock measurement boundary in
  `accelerated/cuda_timing.py`. It synchronizes only before and after the complete measured window,
  retains the raw share and rejects non-positive or materially contradictory samples. The frozen
  engineering targets are 95% accelerator-eligible numerical time, 100 power-system evaluations
  per host boundary and ten policy epochs per reporting boundary.
- Added the development-case-only `calo-rpd-cuda-hot-path` validator. It refuses protected cases,
  requires physical CUDA and a durable clean source identity, creates one FP64 population directly
  in VRAM, performs warm-up outside the measurement, then measures repeated device-resident ORPD
  batches without intermediate host materialization. Qualification additionally requires every
  result tensor to remain on CUDA, full-request VRAM admission, zero declared CPU↔CUDA inner-loop
  transfers, zero CPU fallbacks and the 100-evaluation boundary contract.
- Evidence output is new-file-only strict JSON and explicitly excludes startup, UI, database,
  filesystem I/O, orchestration and final serialization from the numerical-share metric. This is a
  timing/placement harness, not scientific parity, tuning, throughput superiority or release
  evidence. No result is claimed until it is run on a clean committed source with physical CUDA.

### 2026-08-04 — immediate development, step 2: focused invariant coverage

- Added small deterministic invariants for the timing decision boundary, including acceptance at
  or above 95%, rejection immediately below it and rejection of materially contradictory CUDA-event
  and wall-clock samples. The same test locks the 100-evaluation/ten-policy-epoch constants and the
  synchronized experiment, training and generated-schema batching defaults.
- Extended accelerator evidence coverage so dedicated-VRAM proof requires an observed incremental
  PyTorch CUDA allocation, consistent allocator peaks and both allocation and reservation within the
  admission allowance; zero incremental allocation and over-allowance peaks remain rejected.
- Added exact prepared-policy equivalence coverage: a validated topology-aware policy state and its
  one-time prepared tensor representation must produce bit-identical CPU outputs. The existing PPO
  update test remains the focused execution check for prepared states/actions and packed metrics.
- Added a direct campaign-integrity invariant proving that a persisted non-null grouped CUDA-window
  marker makes the campaign identity non-resumable and records a failed status. This complements the
  existing safe post-checkpoint interruption/resume test; it does not weaken exact replay rules.
- No test has been executed yet in this step. The next action is limited to Ruff, the affected mypy
  target and the five focused unit modules; the full suite remains deferred to a later release gate.

### 2026-08-04 — immediate development, step 3: qualification-lane integration

- Registered `calo-rpd-cuda-hot-path` as an installed console entry point and added both the timing
  primitive and validator to the CI typed-module boundary. The manual self-hosted physical-CUDA lane
  now runs separate case30 and case57 100-candidate, ten-batch measurements after CPU/CUDA parity and
  retains their new-file-only JSON evidence with the other physical qualification artifacts.
- The hot-path commands do not run in ordinary CPU CI and do not open case118/case300 or any protected
  qualification cases. They fail the manual lane when the measured share, tensor placement, residency,
  transfer, fallback or 100-evaluation boundary contract is not directly satisfied.

### 2026-08-04 — immediate development, step 4: ten-epoch policy timing boundary

- Added the separate development-only `calo-rpd-cuda-policy-hot-path` command. It constructs a
  topology-aware rollout from case30 or case57 setup state, admits an independent trainer with CPU
  fallback disabled, warms up outside measurement and times complete ten-epoch PPO update blocks.
  Qualification requires CUDA-resident model parameters, nonzero allocation within the admitted
  process ceiling, finite packed metrics, the declared one-transfer-per-ten-epoch provenance and at
  least 95% bounded CUDA event-time share.
- The command records that it does not export, register, qualify or activate the resulting diagnostic
  policy state. Both development cases are wired into the manual physical-CUDA lane alongside ORPD
  timing; protected cases remain closed. Setup power flow, rollout construction and durable JSON are
  outside the timed learner window and remain truthfully CPU work.

### 2026-08-04 — immediate development verification checkpoint

- Focused Ruff lint passes. Ruff formatting initially identified five changed source files and one
  contract test; those files were mechanically formatted and the repeated focused formatting check
  passes. `git diff --check` passes.
- The first focused pytest invocation reached `26 passed` and `11` setup errors caused solely by the
  sandbox-denied default Windows pytest temp directory; it had no assertion failures. Repeating the
  identical six-module set with an explicit workspace-local temporary base passed `37` tests in
  `7.98s`. After policy-lane integration, both validator `--help` imports succeeded and the affected
  contract/residency/trainer subset passed `19` tests in `4.40s`; temporary test data was removed.
- A local mypy rerun is not claimed: neither available existing Python environment contains mypy,
  and no dependency was installed or changed. The CI typed-module list now includes the two timing
  validators and timing primitive, so the pinned CI environment remains the required type gate.
- These short checks establish source-level regression confidence only. No physical CUDA workload was
  run in this checkpoint, so the 95% target, dedicated-VRAM execution of the new validators and the
  requested 100,000-evaluation/1,000-epoch throughput remain pending direct evidence. User-owned
  untracked `Docker_Build.txt` remains untouched.
- A post-verification source audit confirmed the seven discarded NumPy arrays are referenced only
  during CUDA evaluator construction. It also found that the prepared PPO probability path should
  neutralize unavailable group rows before softmax, rather than only zeroing them afterward; this was
  repaired to exclude latent NaN gradients without changing available-group equations or RNG order.
  Focused independent-training and hierarchical-policy regression then passed `18` tests in `5.28s`.
- Immediate development checkpoint is ready to commit as one source milestone: CUDA populations and
  PPO state/action data remain resident across their bounded numerical windows, campaign durability
  is grouped without silent partial replay, both numerical paths have fail-closed physical timing
  validators, and the manual qualification lane retains their evidence. The broader release goal is
  not complete; physical timing/VRAM execution, pinned CI typing, full regression, packaging and final
  release gates remain separate evidence work.
