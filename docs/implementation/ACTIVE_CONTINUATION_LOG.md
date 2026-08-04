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
