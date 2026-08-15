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

### 2026-08-04 — post-commit short physical-CUDA smoke

- Committed the immediate source milestone as `f1a90fb`. Tracked source was clean afterward and the
  user-owned untracked `Docker_Build.txt` remained excluded. The RTX 4060 Laptop GPU was visible with
  8,188 MiB total, 7,957 MiB free and no application allocation at the pre-smoke sample; installed
  PyTorch reported `2.13.0+cu130`, CUDA runtime 13.0 and CUDA availability true.
- The first one-window case30 ORPD validator invocation stopped before workload execution because the
  runtime canonicalized the selected device as `cuda:0` while the validator compared it with the
  alias `cuda`. This is a validator device-identity defect, not parity, timing, VRAM or fallback
  evidence. The target is corrected to explicit `cuda:0`; the same short smoke must be repeated.
- After committing that identity correction as `d352eba`, the repeated invocation reached CUDA
  memory-stat initialization but PyTorch `2.13.0+cu130` rejected a `torch.device('cuda:0')` argument
  for `reset_peak_memory_stats`, requiring numeric index `0`. No numerical workload ran and no timing
  result exists. The shared timing primitive plus both validators now use numeric indices for CUDA
  runtime synchronization/memory/name APIs while retaining `cuda:0` for tensor placement.
- Committed the index normalization as `18c019b`. The next repeat showed the remaining condition:
  this PyTorch build rejects allocator-stat reset before its first CUDA context initialization,
  regardless of device argument form. A one-scalar diagnostic allocation then proved the default,
  integer, string and `torch.device` forms all work after initialization. No ORPD workload ran. Peak
  reset is moved after construction of the resident CUDA problem tensors, before warm-up/measurement.
- Committed the initialization-order fix as `b88d9f3`. The short physical case30 ORPD smoke then
  qualified one 100-evaluation device-resident window: all inspected result tensors were on `cuda:0`,
  full-request residency was admitted, declared CPU↔CUDA inner-loop transfers and CPU fallbacks were
  both zero, peak PyTorch CUDA allocation/reservation were 27,984,896/35,651,584 bytes, and bounded
  CUDA event time was 2.946866 s of 2.947119 s wall time (99.9914%). Retained JSON is
  `artifacts/cuda-hot-path-b88d9f3/case30-smoke.json`, SHA-256
  `54c56a5553b1b0ec1f171337866af28e5508302627ec1a8324832d85ec263d75`.
- The matching short physical TSH-CALO smoke qualified one measured ten-epoch PPO update after one
  warm-up update. Model/optimizer work remained admitted on `cuda:0`, packed metrics were finite,
  allocated/peak allocated VRAM were 18,059,264/18,602,496 bytes within the 5,953,395,097-byte
  process ceiling, the per-epoch host metric-transfer flag was false, and bounded CUDA event time was
  0.238431 s of 0.238479 s wall time (99.9799%). No policy was exported, registered, qualified or
  activated. Retained JSON is `artifacts/cuda-hot-path-b88d9f3/case30-policy-smoke.json`, SHA-256
  `a4bad813823e4553f9ee38feb59fd8e9bd810db15c627759d21100541d47cea4`.
- These are source-bound development smokes for case30 on the observed RTX 4060/PyTorch stack, not
  complete case30/case57 manual-lane qualification, GPU-utilization proof, an overall-application
  95% claim or evidence for 100,000 evaluations/1,000 epochs in minutes. The immediate development
  requirement is implemented and smoke-verified; broader release testing remains open without tuning.

### 2026-08-04 — current-commit case30/case57 CUDA hot-path qualification

- Audited `IMPLEMENTATION_GATES.md`, `REQUIREMENT_TRACEABILITY.md` and the release handoff against
  clean commit `c1cf911a5a14eb25df916d77c4dd6f1ba8994388`. The next legal non-tuning boundary was
  development-case physical CUDA evidence; protected cases and a new scientific candidate remain
  closed. Executed the exact manual-lane residency windows on the observed RTX 4060 with PyTorch
  `2.13.0+cu130`/CUDA 13.0.
- Case30 ORPD passed ten measured batches of 100 candidates after two warm-ups: 1,000 measured
  evaluations, all inspected output tensors on `cuda:0`, full-request residency admitted, zero
  declared CPU↔CUDA inner-loop transfers/fallbacks, peak allocation/reservation
  28,027,392/35,651,584 bytes, and 29.309986 s CUDA event time over 29.310196 s wall time
  (99.999283%). Evidence SHA-256:
  `517039bc326585fa4555e339291fc49cb745359a3345618ccbccf49fcf78d462`.
- Case57 ORPD passed the same 1,000-evaluation window with peak allocation/reservation
  79,052,288/121,634,816 bytes and 47.867195 s CUDA event time over 47.867493 s wall time
  (99.999377%). Evidence SHA-256:
  `c50c18fa4fc0cb946032ef310d326286c2e611d23afd5068883de3bd68284dc2`.
- Case30 and case57 independent policy validators each passed three measured ten-epoch updates after
  one warm-up update (30 measured epochs per case). Both retained CUDA model/optimizer allocation,
  finite packed metrics, no per-epoch host metric transfer, no fallback and no export/registration/
  qualification/activation. Case30 recorded 0.737026/0.737100 s CUDA/wall (99.989917%), peak
  allocation 18,602,496 bytes; case57 recorded 0.768172/0.768263 s (99.988173%), peak allocation
  19,051,008 bytes. Evidence SHA-256 values are
  `0b13b15849fb086da3be4119d045c135bb0ef6c68e6aa1bff7a31bbc9b467c02` and
  `b4c75efa5cce21925a2afe217305e1fe57ff5fcd4695f1ba408324ae8e0d5758`.
- Retained paths are under `artifacts/cuda-hot-path-c1cf911/`. This closes the new source-bound
  case30/case57 hot-path placement/timing development gate only. It does not prove GPU utilization,
  whole-application CUDA share, CPU/CUDA scientific parity for commit `c1cf911`, requested extreme
  throughput, final-container/CI qualification, or scientific benefit.

### 2026-08-04 — exact-source parity and resource-recovery follow-up

- At clean commit `5b50095054f312de8e1aec8548c6fde6cd27c453`, case30 and case57 each passed
  the 27-candidate physical FP64 CPU/CUDA parity battery with dedicated-VRAM allocation proof, zero
  feasibility/convergence/bus-type/scenario-count mismatches and unchanged scientific fingerprints.
  Evidence SHA-256 values are `424b8de56ce3cdb9b52c4bee4a38583fcf40a72c2a86ad5cb5144465eee81881`
  and `8708f691929e3931b838fc5c80b313895cac012435ebb83b438684245b1d67ff`.
- The subsequent bounded resource-recovery run correctly failed qualification. Pressure/recovery,
  controlled microbatch backoff, controlled CPU restart plus CUDA recovery and cross-process lease
  exclusion/release all passed. Its host-staging probe failed because it still assumed an ordinary
  small host-origin request must remain staged; the new residency contract correctly uploaded and
  admitted that full request instead. Failed evidence is retained at SHA-256
  `6ff947381c48c648d1bf6017f5ea0679f721655a3673b3b31aac01f40a2d5b02` and is not a gate pass.
- Repair preserves both requirements: ordinary fitting requests remain fully resident, while a new
  narrow evaluator upload seam lets the validation harness inject a typed CUDA OOM on the initial
  full-request upload only. The staged-host probe now requires that injection, explicit attempted-but-
  not-admitted metadata, the declared staging reason, CUDA outputs, zero CPU inner-loop participation
  and no microbatch OOM retry. The injection is labelled and is not natural-hardware-OOM evidence.
- Focused resource/evidence/VRAM regression after this repair passed `24` tests in `2.86s`; focused
  Ruff lint passes and the touched files are formatted. The failed resource record remains retained,
  and the repaired probe must be repeated from a clean committed source before this gate can close.
- Committed the repaired probe as `74268e343a50103e20b04160f61d055c05733648` and repeated the exact
  bounded physical resource run. Qualification passed: observed/recovered 256 MiB pressure and
  Safe-80 admission contraction, exactly one injected initial-upload OOM followed by explicit
  host-staged CUDA execution with the declared reason and no CPU inner loop, controlled `5→2`
  microbatch backoff, controlled whole-request CPU restart plus subsequent CUDA recovery, and
  cross-process lease refusal then acquisition after release. Evidence SHA-256 is
  `73bb1f1bf6905f221b7993a2ec5d1bde50ccb6a074e22337a759234f5c64f13a` at
  `artifacts/physical-parity-74268e3/resource-recovery.json`. Controlled faults remain explicitly
  outside any natural-hardware-OOM claim.

### 2026-08-04 — pinned typed boundary restored

- Installed repository-pinned mypy `1.20.2` plus its runtime dependencies into the ignored local
  `.venv`; no project dependency lock or declared runtime dependency changed. The exact CI 15-module
  command initially found seven issues: a PyQt static-method stub mismatch, one missing policy-array
  annotation, one NumPy return inferred as `Any`, and three L-SHADE typing gaps.
- Applied typing-only repairs: a narrowly scoped `QTest.qWait` stub ignore preserving the already
  validated runtime call, explicit NumPy array/result annotations, `Evaluation` parameter types for
  the existing constrained-improvement adapter, and typed L-SHADE memories/epsilon. Operator
  equations, RNG order, scientific fingerprints, thresholds and runtime choices are unchanged.
- Repeated `mypy --follow-imports=skip --check-untyped-defs` over the exact 15 CI modules: success,
  no issues. Focused Ruff passes, and the affected L-SHADE, mathematical-reference, packaged-GUI,
  CUDA-residency and independent-training regression set passed `32` tests in `12.41s`.

### 2026-08-04 — exact source-stage and complete active-tree regression

- On committed source `05cd1b8`, all three hash-complete requirement-lock verifiers passed: CPU
  PyTorch 2.10.0+cpu, CUDA 12.8 PyTorch 2.10.0+cu128, and the matched CI lock with pinned Ruff
  0.15.22, mypy 1.20.2 and uv 0.11.29. `compileall`, generated experiment-schema check and
  repository Ruff lint passed.
- Ruff's complete format gate identified seven previously changed files; applied only its mechanical
  formatting and repeated the repository check successfully across `422` Python files. No intended
  behavior or scientific value changed.
- Executed the complete active test tree once with checkout source coverage, excluding only
  `tests/unit/test_v690_release_integrity.py` exactly as documented because that historical freeze is
  regenerated only at final G11. Result: `638 passed, 63 skipped` in `148.81s`; coverage was `68%`
  over 35,252 statements, passing the configured 60% threshold. This supersedes the prior 632-pass
  development checkpoint for local regression evidence. It does not replace Linux/CI/container,
  final-freeze, protected-case or scientific-campaign evidence.

### 2026-08-04 — post-regression G9/G10 development audit

- Verified that the worktree is clean apart from the user's untouched untracked
  `Docker_Build.txt`. The persistent release-readiness goal remains active; no release, publication,
  registration or protected-case authority has been inferred.
- Re-read the current gate, traceability and handoff boundaries. Local source development and the
  shortened active regression gate are green. The next substantive boundary is G9: the historical
  v2/v3-ABI policy candidate failed its frozen development screen and cannot be reused or
  reinterpreted under counted runtime v1.1/training v4.
- Located the existing v4 training, qualification and component-evidence implementation, including
  `tsh_calo_training_campaign.py`, `tsh_calo_qualification_campaign.py`, the explicit train/qualify
  CLIs and their focused test families. The immediate task is therefore a narrow completeness audit
  of the frozen-plan and A–E paired-ablation contracts before any fresh development-only candidate
  is launched. Protected cases remain closed and algorithm tuning remains prohibited.
- Test policy for this phase: run only touched-module lint/type checks and focused unit/invariant
  tests after each change; rerun the full active suite only at a release-gate boundary. This shortens
  development feedback without weakening any final acceptance criterion.

### 2026-08-04 — G9 component-evidence completeness finding

- Completed the narrow source audit of the counted-v4 training and qualification boundaries. The
  training campaign already freezes source, curriculum, seeds, hyperparameters, resource routing and
  A–F flags; the qualification campaign already retains paired equal-FE cells, independent solution
  validation, anytime endpoints, deterministic intervals, Holm correction and non-promotion
  semantics.
- Found one immediate development gap: formal qualification accepts checksum-bound A–E component
  evidence references, but the repository contains no producer for the declared
  `tsh-calo-component-ablation-evidence-v1` schema. Its verifier currently checks only schema,
  component, acceptance, policy identity and protected-case exclusion. Therefore a fresh candidate
  cannot lawfully earn the evidence prerequisite using repository-owned code.
- The existing production feature validator correctly refuses disabling graph, hierarchy,
  uncertainty or bandit components, and the fixed production optimizer correctly refuses
  experimental F. Those safeguards will remain unchanged. The repair will add a separate
  non-serializable, development-only ablation capability and frozen paired campaign; it must have no
  registry, activation, protected-case or production-experiment authority. A–E will be attributable;
  F remains separately experimental, disabled and ineligible for formal production qualification.

### 2026-08-04 — repository-owned A–E component-evidence development completed

- Added `tsh_calo_component_ablation.py` and the explicit `calo-rpd-ablate-tsh` command. A frozen
  plan now binds exact source/candidate/cases, at least 30 paired runs per development case, seeds,
  FE/population budget, calibration, anytime checkpoints, statistical/practical thresholds and
  execution route. Start requires a new directory; resume requires the identical plan and retained
  cell identity.
- The matrix executes frozen CALO, the canonical-refactor identity, graph-only, hierarchy-only,
  graph+hierarchy, +uncertainty, +bandit and the full approved A–E candidate under paired seeds and
  equal exact FE. It reuses independent PYPOWER validation and retained feasibility-first/anytime
  records. Change A requires exact seeded record parity; B/C/D/E require their frozen incremental
  comparisons to pass feasibility non-inferiority, paired-feasible coverage, practical improvement,
  win rate, rank-biserial effect, global Holm control and anytime non-regression. Standalone B/C
  comparisons remain explicit falsification records.
- Component removal is available only through a non-serializable development capability. Ordinary
  production/qualification still require the immutable candidate's full approved flags; the
  production optimizer still rejects F. The A–E preflight requires an ensemble trained with E,
  rejects F/experimental authority, and emits no training, qualification, receipt, registry,
  activation or deployment action.
- Strengthened formal qualification verification: accepted component evidence must now match the
  exact candidate and source, exact development-case design, protected-cases-closed declaration,
  frozen plan/scientific/seed hashes, direct non-empty analysis and the component-ablation-only
  authority boundary. A minimal accepted Boolean is no longer sufficient.
- Preserved the full production network path bit-for-bit while adding explicit graph/hierarchy
  removal forward paths. Focused Ruff formatting/lint and compilation pass. New/touched ablation,
  topology, qualification, inference and optimizer tests pass `33` cases in `10.95s`; the new
  campaign/inference/CLI typed boundary passes mypy with no issues. No real ablation was run and no
  component or policy-benefit claim was made.
- Committed this development boundary as `ae7b304` (`feat(tsh): add frozen component evidence
  campaign`). The only unrelated worktree item remains the user's untouched untracked
  `Docker_Build.txt`.

### 2026-08-04 — candidate-bound CPU/CUDA equivalence development completed

- Added the `calo-rpd-tsh-device-equivalence` validator for one exact immutable unqualified
  ensemble. It requires a clean durable source identity and physical CUDA, uses only case30/case57
  development states, fits a checksum-bound development-only OOD calibration, and evaluates the
  candidate through the existing non-serializable qualification capability. It cannot register,
  qualify, activate, export or deploy a policy.
- The validator loads the same candidate separately on CPU and CUDA with fallback forbidden, uses
  the same deterministic seed, state, action mask, learner groups/contexts, fresh identical bandit
  state and safety envelope, and requires exact regime, learner operators, action masks and
  intervention reasons. Group parameters, operator probabilities, uncertainty, mixture weights,
  value and OOD values must be finite, shape-identical and within frozen `rtol=1e-5`/`atol=1e-6`
  defaults.
- A CUDA pass additionally requires selected NVIDIA computation, a positive dedicated-VRAM
  allocation increase while loading the ensemble, peak allocation at least the live allocation and
  every ensemble parameter resident on CUDA. Evidence retains the candidate/source/design/
  calibration identities, both controller provenance records, per-case maximum absolute
  differences, runtime/device identity and explicit protected/lifecycle boundaries.
- Short verification only: Ruff format/lint, compilation and mypy pass; the device-equivalence and
  adjacent inference tests pass `11` cases in `5.42s`. No physical candidate equivalence run was
  attempted because no fresh counted-v4 A–E candidate has yet been frozen; the historical negative
  candidate must not be reinterpreted.
- Committed this validator boundary as `e77431e` (`feat(validation): add TSH device equivalence
  gate`).

### 2026-08-04 — release-ledger reconciliation after immediate development

- Updated `IMPLEMENTATION_GATES.md`, `REQUIREMENT_TRACEABILITY.md` and
  `RELEASE_READY_CONTINUATION_HANDOFF.md` to the same boundary. They now record the latest 638-pass
  complete active source gate, current-source physical evaluator/recovery/hot-path evidence, and
  commits `ae7b304`/`e77431e` as implemented harnesses rather than executed scientific evidence.
- G9 remains in progress. Immediate non-tuning repository development is complete through counted-v4
  training, qualification, A–E paired evidence production and candidate device equivalence. What is
  pending is execution: freeze a new A–E/F-off counted-v4 plan, train one fresh ensemble, run its
  physical equivalence and frozen A–E matrix, then apply the unchanged qualification criteria.
- The ledgers continue to reject the historical v2/v3-ABI candidate, threshold changes, post-hoc
  reinterpretation, protected-case opening, fabricated hardware evidence and premature release/
  benefit claims. F remains a separately experimental study and cannot enter formal A–E evidence.
- Kept the short-test policy explicit: focused tests after development changes; one complete active
  suite at the next source/release gate. No extra long regression or scientific campaign was run for
  this documentation-only reconciliation.
- Committed the synchronized ledgers as `d78d5f2` (`docs(gates): reconcile immediate G9
  development`).

### 2026-08-04 — counted training CUDA-path audit exposed a blocking development gap

- Audited the historical counted-v2 campaign and the current training call chain before freezing or
  launching a fresh counted-v4 candidate. Although policy training can select CUDA, the training
  environment currently calls `ORPDProblem.evaluate_with_context` once per candidate. The default
  campaign factory constructs the CPU reference problem, so every counted power-system evaluation,
  topology context and retained control-linearization request still traverses a Python/CPU loop.
- Consequently, the old campaign demonstrates policy-model CUDA residency only; it is not evidence
  that more than 95% of the eligible end-to-end training workload ran on the GPU. No fresh campaign
  was launched and no CUDA-utilization, throughput, qualification or release claim was made.
- The immediate development repair is now a hard prerequisite: retain the completed tensor power-
  flow state for a whole population on CUDA, materialize counted evaluation contexts only at the
  outer batch boundary, add an accelerated batch `evaluate_population_with_context` API with exact
  FE/scenario accounting and no hidden CPU power-flow solve, route CUDA campaign sessions through
  that API, and prove bounded CPU/CUDA evaluation/context parity. Focused tests will be used while
  implementing it; the complete active suite remains reserved for the next source-gate boundary.

### 2026-08-04 — batch device-resident counted-context development completed

- Extended the device-resident ORPD population result to retain final complex voltages, convergence
  and Newton diagnostics, final PV/PQ bus types, actual generation and branch complex flows/loading
  on the selected tensor device. Evaluation records and all retained context arrays now use one
  packed population-level host materialization after the CUDA work completes; microbatch
  concatenation remains on-device and no candidate-level CPU↔CUDA transfer loop was introduced.
- Added `AcceleratedORPDProblem.evaluate_population_with_context`. It reconstructs the ephemeral
  public `PowerFlowResult`/topology contexts from the already-counted tensor state and never invokes
  another CPU power-flow solve. Optional Change-E derivatives are derived from that final counted
  state at the outer boundary; this retains exact FE/scenario accounting and proposal-only physics-
  repair semantics. The production TSH optimizer and independent training environment now submit a
  complete bounded population through this API instead of calling one scalar evaluator per row.
- The default campaign now constructs an accelerated ORPD problem on the trainer's admitted CUDA
  device, requests a population boundary of at least 100 evaluations, and carries the frozen CPU-
  fallback choice through to the evaluator. Explicit CPU campaigns and injected test/reference
  factories remain supported. Exact-resume compatibility was advanced to campaign v2/status v2,
  training environment v5, environment checkpoint v3 and session v2 so historical CPU-loop
  campaigns cannot silently resume under the changed execution ABI.
- Advanced the signed episode receipt to v2. Every fresh candidate now binds the counted ORPD
  computation class and selected device, whether the batch-context API was used, the target host
  boundary, inner CPU-CUDA transfer count and hidden context power-flow rerun count. CUDA receipts
  fail validation unless they declare a CUDA device, batched contexts, zero inner-loop transfers
  and zero hidden power-flow reruns. This is structural provenance, not a substitute for physical
  utilization/timing evidence.
- Short development verification passed Ruff/compile checks and `76` focused accelerator, VRAM,
  counted-environment, session/campaign, lifecycle, inference, optimizer and qualification tests in
  `24.52s`. A bounded context parity test checks evaluation, voltage, bus-type, branch-flow,
  Jacobian and control-sensitivity equivalence and forbids a reference-evaluator rerun. The isolated
  two-module accelerator typed boundary is clean. No full suite or scientific campaign was run.
- No greater-than-95% end-to-end CUDA claim is made yet. The computational power-flow population is
  now structurally CUDA-resident, but context-object reconstruction and optional Change-E outer-
  boundary linear algebra still touch the CPU. The next gate is physical NVIDIA case30/case57
  candidate-bound VRAM/timing evidence; if eligible CUDA share is not greater than 95%, those
  measured remaining boundary costs must be ported before any fresh training campaign starts.

### 2026-08-04 — release ledgers reconciled to the batch-context boundary

- Synchronized `IMPLEMENTATION_GATES.md`, `REQUIREMENT_TRACEABILITY.md` and
  `RELEASE_READY_CONTINUATION_HANDOFF.md` to training environment v5/campaign v2/session v2/receipt
  v2 and the completed population-level counted-context implementation. Removed the stale direction
  to start a fresh counted-v4 campaign immediately.
- The documented next legal order is now explicit: first bind the current source to physical NVIDIA
  dedicated-VRAM/timing evidence and require greater than 95% eligible CUDA work with zero inner
  transfer loop; port any measured remainder if it fails; only then freeze and execute the fresh
  A–E/F-off plan, candidate equivalence, frozen A–E matrix and eligibility gates. Protected cases
  remain closed and no tuning, benefit, qualification or release claim was introduced.
- This was a documentation reconciliation after the `76`-test focused development gate; no second
  regression suite or scientific campaign was run. The locally created `.codex-pytest-temp`
  directory could not be removed because the managed approval service rejected the cleanup after
  reaching its usage limit; it is untracked generated test state and must not be committed. The
  user's unrelated untracked `Docker_Build.txt` remains untouched.

### 2026-08-06 — v12.0 Phase 1 coding completed; validation explicitly deferred

- Created the active v12 development identity `12.0.0.dev1` / `12.0.0-dev.1` across runtime,
  package metadata, CLI `--version`, GUI labels, README, container labels and CI. Added
  `ACTIVE_DEVELOPMENT_STATUS.json` and `STATUS_RECORD_INDEX.json`; RC/final/freeze/qualification and
  protected-case flags remain false. Historical v6.9 records were not rewritten.
- Implemented `calo-paired-analysis-v2-exact-keyed-signed-rank`: exact preregistered keyed pairs fail
  on duplicates/missing/extra ambiguity; exploratory incompleteness is explicitly nonqualifying;
  positive means candidate improvement; matched-pairs rank-biserial uses signed rank mass with
  average ties; and formal Wilcoxon evidence records SciPy/version/alternative/zero method without
  changing test families on failure.
- Migrated both policy qualification engines to the shared v12 analysis definition and new campaign
  plan/evidence schemas. Separated time-to-first-feasible from versioned post-feasibility AUC. No
  optimizer, diversity-pressure, policy architecture, training transition or protected-case
  behavior changed.
- Added an additive historical statistical correction record. The retained v3 screen is untouched;
  its effect is legacy/unverifiable without complete immutable tracked raw pair values. The negative
  result cannot reverse because the retained case57 interval crossed zero and Holm-adjusted
  `p=0.052734375` failed the frozen mandatory threshold. Requalification is required for any future
  promotion.
- Freeze verification now emits invalid/missing/changed/extra categories together in human and
  machine-readable output. The v6.9 release-integrity test is scoped to a v6.9 checkout.
- Added `tests/unit/test_v120_phase1_contracts.py` and updated affected historical compatibility
  expectations. At the user's instruction, **no tests were executed** and no Python, policy
  training, policy evaluation, qualification, ablation, benchmark, campaign or protected-case
  command was invoked. Prior pass counts do not validate this v12 working tree.
- Saved exact later validation/evidence-capture instructions in the local-only, Git-ignored
  `validation/Validate-Phase1.ps1` harness. It writes detailed hashed evidence under
  `validation/logs/`. Phase 1 coding is complete; its exit gate remains validation-pending until
  those outputs are returned for review. Do not begin Phase 2 or any policy workflow without a new
  instruction.
- Repository ownership correction: `Docker_Build.txt` is tracked. Untracked
  `.codex-pytest-temp/` and generated `calo_policy_candidate` branch/artifact state remain user/runtime
  files and must not be staged, packaged or modified.

### 2026-08-06 — local-only Phase 1 manual validation harness prepared

- At the user's direction, added the repository-local `validation/Validate-Phase1.ps1` manual
  harness and excluded the entire `/validation/` tree in `.gitignore`. The script itself, every
  command log, evidence JSON, summary, transcript and SHA-256 manifest are therefore local-only and
  must never be staged or committed.
- Replaced the earlier untracked documentation command log with the executable local harness and
  updated all Phase 1 handoff references. Future phases use the same convention: a phase-specific
  local validation script plus detailed ignored logs returned manually for review.
- The Phase 1 harness records Python/Git/source identity, complete working-tree inventory, ignore
  enforcement, diff integrity, CLI/GUI/package/container version consistency, broad-exception
  governance, compilation, generated-schema consistency, Ruff diagnostics/format, Phase 1 contract
  tests, affected compatibility tests and historical-v6.9 test scoping. It contains no training,
  policy evaluation, qualification, ablation, benchmark or protected-case entry point.
- The harness writes a complete run under `validation/logs/phase1-YYYYMMDD-HHMMSS/`, including
  per-command exit codes/output, a source SHA-256 manifest, JSON and Markdown summaries, a transcript
  and a SHA-256 manifest of the validation evidence. Return that complete directory for review.
- The harness was **not executed**. Only the PowerShell parser and Git ignore/static source checks
  were used while preparing it; PowerShell parsing passed, both the script and example logs resolve
  through `.gitignore`, and the ignored validation tree is absent from `git status`.

### 2026-08-06 — Phase 1 validation-log findings corrected; fresh manual rerun prepared

- Reviewed the complete `validation/logs/phase1-20260806-222429` run. Its evidence hashes all
  verified. It reported 12/16 commands passed: active v12 identity, exception classification,
  compilation, generated schema, 12 Phase 1 contracts, five affected regressions and five expected
  historical-v6.9 skips passed. No policy training/evaluation, campaign, benchmark or protected case
  ran.
- Classified three failures as harness/environment defects: Windows PowerShell promoted native
  stderr warnings to terminating `RemoteException` for Git status/diff; root-wide Ruff encountered
  the inaccessible `.codex-pytest-temp` runtime directory; and mixed `Set-Content`/`Tee-Object`
  encodings produced NUL-interleaved command logs. Ruff format was the one confirmed source failure:
  12 files required formatting.
- Corrected the validator to capture native stdout/stderr under nonterminating handling, restore the
  caller's error policy, use the native process exit code as pass/fail authority, and append all
  retained output consistently as UTF-8. Git warnings remain visible without creating false
  failures. Ruff diagnostics now scan only `calo_bootstrap`, `calo_rpd_studio` and `tests`.
- Added Git exclusions for `.codex-pytest-temp/` and the three generated
  `calo_policy_candidate` branch/artifact paths. The validation tree, runtime test state and generated
  candidate state are absent from `git status` and remain prohibited from source/release content.
- Applied Ruff's mechanical formatter to the exact 12 files identified by the log; it reported
  `12 files reformatted`. This was a formatting correction, not a test or scientific execution.
- Prepared the next local run. The revised PowerShell harness parses successfully, every local-only
  path resolves through `.gitignore`, complete Git status now reads without the prior permission
  warning, and `git diff --check` returns success while retaining non-failing LF/CRLF notices.
- The revised Phase 1 validator was **not run**. The next action is for the user to execute
  `validation/Validate-Phase1.ps1` manually and return the new complete
  `validation/logs/phase1-YYYYMMDD-HHMMSS/` directory for review.

### 2026-08-06 - Phase 1 functional validation passed; source-identity closure corrected

- Reviewed the complete `validation/logs/phase1-20260806-224036` rerun without executing any
  validator or test command. All 21 listed evidence artifacts matched the retained SHA-256 manifest,
  and all 29 files in the source manifest still matched the current workspace at review time.
- The rerun reported 16/16 commands passed: active v12 identity, Git integrity, broad-exception
  classification, compilation, generated schema, Ruff diagnostics and formatting, 12 Phase 1
  contracts, five affected regressions and five expected historical-v6.9 skips. It recorded no
  policy training/evaluation, qualification campaign, benchmark or protected-case execution.
- The functional Phase 1 checks passed, but final evidence acceptance remained conditional because
  the captured Git inventory contained 32 changed/untracked source files while the explicit source
  manifest contained 29. The omitted files were `calo_rpd_studio/app/main_window.py`,
  `calo_rpd_studio/gui/navigation/sidebar.py` and
  `calo_rpd_studio/gui/panels/application_settings_panel.py`. The manifest also identified the
  validator only by path, without binding its exact content.
- Corrected the local-only validator source manifest to include all three GUI files. Upgraded the
  manifest to `calo-phase1-source-manifest-v2` and added the validator script size, SHA-256,
  declared source-path count and missing-source-path count. No application semantics changed.
- The corrected validator was **not run**. The next action is one final user-executed Phase 1 rerun;
  return the new complete ignored evidence directory for hash, source-coverage and exit-gate review
  before Phase 2 source changes begin.

### 2026-08-06 - Phase 1 validation accepted; Phase 2 coding handoff prepared

- Accepted the user-executed `validation/logs/phase1-20260806-230256` evidence after checking its
  retained summary and hashes: 16/16 commands passed, the source manifest covered 32/32 declared
  paths, and the validator content was hash-bound. The log states that no policy training,
  policy evaluation, qualification, benchmark, campaign, or protected-case workflow executed.
- Implemented the versioned `calo-runtime-execution-contract-v2` resolver shared by direct,
  sequential, parallel, GUI, ordinary CLI, benchmark, and final-campaign entry points. Scientist
  modes remain `cuda_preferred` and `cpu_only`; `execution_purpose=formal` makes CUDA mandatory and
  fallback forbidden, while exploratory fallback is one explicit full-request CPU restart.
- Separated requested mode/device, physical UUID-or-PCI identity, logical CUDA index, runtime
  device, fallback policy/reason, actual evaluator device, host/container lease scope, and
  CUDA-only claim eligibility. Final campaigns now require formal CUDA and disable CPU fallback.
- Bound CUDA ownership to physical UUID with normalized PCI fallback, retained logical indices
  separately, added queued lease wait/cancel/timeout semantics, froze scheduler routing to the
  pre-resolved device, preserved Safe-80 admission, and separated per-request from governor-lifetime
  VRAM telemetry.
- Added central batch cardinality and candidate-identity validation before any FE registration.
  Short, long, empty-mismatched, partially identified, reordered, or duplicated results fail with a
  typed invariant error while the FE count remains unchanged.
- Added `calo-partial-run-failure-v2` envelopes preserving exact FE count, iteration, incumbent,
  feasibility/violation state, convergence state, numerical metadata, runtime/fallback resolution,
  and checkpoint reference. Existing database transactions retain the envelope and count together.
- The dense-case and capacity paths now obey the same fallback flag across ordinary and
  counted-context APIs, record actual CPU execution, and exclude fallback results from CUDA-only
  timing, energy, parity, utilization, and equivalence strata.
- Topology records now name `torch_fp64_tensor_matmul_smoke_v1` as the direct CUDA FP64 authority.
  Active status exposes only CUDA-preferred/CPU-only execution and leaves XPU nonexecutable;
  historical XPU evidence remains immutable/view-only.
- Added versioned runtime-resolution and partial-failure JSON schemas plus
  `tests/unit/test_v120_phase2_contracts.py`. No test, policy training/evaluation, qualification,
  benchmark, campaign, protected case, or scientific evidence workflow was run by Codex.
- Prepared `validation/Validate-Phase2.ps1`. It is local-only under the already ignored
  `/validation/` tree, records every command and source SHA-256, dynamically includes all current
  changed/untracked source paths, and contains only identity, compile, schema, lint/format, Phase 2
  contract, and affected regression checks.
- Phase 2 coding is complete but its exit gate is **validation-pending**. The next legal action is
  for the user to run `validation/Validate-Phase2.ps1` and return the complete new
  `validation/logs/phase2-YYYYMMDD-HHMMSS/` directory. Do not begin Phase 3 or make a release claim
  until that evidence is reviewed and corrections, if any, are closed.
- Coding-time tooling record: two schema-generator attempts with the system/bundled Python failed
  before generation because `yaml` was unavailable, and the repository venv launcher was initially
  sandbox-blocked. The authoritative generator and derived schema were synchronized in source for
  the manual `--check` gate. The repository Ruff executable was later used only as a mechanical
  formatter on the Phase 2 Python paths (19 reformatted/4 unchanged, then 7/2 and 6/2 on the two
  final scoped passes, followed by 1/1 on the final two-file cleanup); no Ruff validation check or
  test was run by Codex.
- Static handoff-only checks passed: the Phase 2 PowerShell script parses, both script/log paths are
  ignored by Git, the active/status/schema JSON documents parse, and `git diff --check` succeeds
  with only expected LF-to-CRLF notices. These checks do not close the manual validation gate.

### 2026-08-07 - Phase 2 first validation reviewed; two corrections applied

- Reviewed the complete retained `validation/logs/phase2-20260807-001858` evidence without running
  the validator or any test. The run passed 13/15 commands: 23/23 dedicated Phase 2 contracts passed,
  and the affected runtime regression set reported 43 passed/1 failed.
- Evidence integrity was intact: all 20 retained artifact hashes matched, the v2 source manifest
  covered 33/33 declared paths with current hashes matching at review time, and the validator size
  and SHA-256 matched its manifest. The run recorded no policy training/evaluation, qualification,
  benchmark, campaign, or protected-case execution.
- The schema failure was text-only: `runtime_compute_device` preceded the two generated assigned-
  device properties even though the authoritative generator emits physical, logical, then compute.
  Reordered those three JSON property blocks without changing names, types, defaults, or semantics.
- The sole runtime-regression failure was a stale historical assertion expecting the old
  `between 0.10 and 0.95` message. The Phase 2 implementation correctly rejects 100% VRAM against
  the Safe80 ceiling, so the regression now expects `no greater than 0.80`; runtime policy code was
  not weakened or changed.
- Active status and its source assertions now truthfully bind the failed run as corrections-applied,
  rerun-pending. Codex performed no test, schema-generator check, policy workflow, benchmark,
  campaign, protected-case operation, or scientific evaluation while applying the corrections.
- Phase 2 remains open. The next action is a full user-run `validation/Validate-Phase2.ps1`; return
  the complete new `validation/logs/phase2-YYYYMMDD-HHMMSS/` directory for review before Phase 3.

### 2026-08-07 - User-directed phase execution protocol strengthened

- Added a repository-wide rule that every numbered development phase must receive a new concrete
  goal through the goal service before the agent announces or begins phase source development.
- Restricted phase work to coding: production source, test source, schemas, documentation, and
  validation harness creation. The agent must not execute any manual-capable test or validation
  command unless the user later names and explicitly authorizes that command.
- Required every phase coding pass to end with a detailed PowerShell validator under the ignored
  `validation/` tree. The user runs it and returns the complete timestamped logs; the agent reviews
  those logs read-only, applies evidence-backed corrections, and prepares an unexecuted rerun.
- Validators and logs remain prohibited from Git and release artifacts. Focused reads and minimal
  command output are now an explicit token-conservation requirement.
- This entry changes workflow governance only; it does not start Phase 3. Therefore no new phase
  goal was created, and no test or validation command was executed for this documentation update.

### 2026-08-07 - Phase 2 second validation reviewed; formatting corrected

- Reviewed `validation/logs/phase2-20260807-003024` read-only. It passed 14/15 commands: generated
  schema, Ruff diagnostics, 23/23 Phase 2 contracts, and 44/44 affected regressions all passed.
- Verified all 20 retained artifact hashes, exact 34/34 captured-path manifest coverage, validator
  size/hash identity, zero missing manifest paths, 163 classified broad handlers, zero parse
  failures, and zero scientific-priority broad handlers. No prohibited workflow was recorded.
- The only failure was Ruff format reporting `tests/unit/test_v690_vram_residency.py`. Focused byte
  inspection showed 316 CRLF endings and three bare LF endings introduced around the corrected
  assertion. Normalized the file to its established CRLF style without executing Ruff or tests.
- Four governance documents had legitimately changed after the run while implementing the user's
  new phase workflow, so the old evidence could not bind the final current source even without the
  formatting failure. Active status and source assertions now bind the second failed run as
  formatting-corrected/rerun-pending.
- No validator, test, lint, formatter, compile, schema, benchmark, policy, campaign, qualification,
  or protected-case command was executed. The user must rerun `validation/Validate-Phase2.ps1` and
  return the complete new timestamped log directory before Phase 2 or transition to Phase 3 passes.

### 2026-08-07 - Phase 2 accepted; Phase 3 scientist-GUI coding completed

- Accepted the user-run `validation/logs/phase2-20260807-003828` evidence read-only: 15/15 commands,
  23/23 Phase 2 contracts, and 44/44 affected regressions passed. All 20 retained artifacts, all 35
  current source paths, and validator identity matched; no prohibited workflow was recorded.
- Created the required goal before any Phase 3 source development. Its concrete scope is modern
  scientist GUI coding, ledgers/traceability, and an unexecuted ignored manual validator.
- Preserved all sixteen stable workspace keys and legacy migration behavior while implementing five
  collapsible presentation groups: Home, Model, Study, Evidence, and System. Added persisted compact
  navigation, persisted group state, SVG icons, Ctrl+K search, textual state badges, hidden locked
  children, and accessible blocked-step explanations.
- Reorganized the Dashboard around the next legal action, five readiness categories, active study
  context, retained/recent/resumable/failure/evidence summaries, and progressive activity details.
  The long protocol form is no longer visible on the Dashboard; compatibility methods remain atomic.
- Added a single-scroll seven-step Study Setup in Experiment Manager, routing case, formulation,
  algorithm, and scenario decisions to their authoritative panels and retaining existing budget,
  audit, and launch controls without duplicating state. Advanced continuation and queue details are
  collapsed by default.
- Added a reusable compact-input/accessibility policy: ordinary text/select/scalar limits of
  480/420/240px, a 900px Study Setup column, compact/comfortable 40/44px targets, form label buddies,
  accessible names/descriptions, explicit long-text expansion, and structured integer chips.
- Added named light/dark semantic tokens, consistent focus treatment, tabular scientific numerals,
  8px-based spacing, and text labels so status is not communicated by color alone.
- Added `validate_phase3_gui_render.py` and Phase 3 test source for grouping, migration, compactness,
  accessibility, Dashboard separation, seven-step setup, semantic themes, glyph availability,
  clipping, scale, and presentation-layer preservation of the accepted Phase 2 runtime contract.
- Prepared ignored `validation/Validate-Phase3.ps1`. It dynamically hashes all changed/untracked
  source, binds its own identity, records detailed command logs, and requests local light/dark
  1280x720, 1440x900, 1920x1080, and 200%-scale renders. It explicitly records that Linux rendering
  and scientist acceptance are separate pending evidence.
- Codex did not run the validator, tests, renders, Ruff, compilation, schema, type checks, package
  smoke tests, GUI/browser interaction, Docker, benchmarks, campaigns, policy workflows,
  qualification, or protected cases. Phase 3 coding is complete; manual validation remains pending.

### 2026-08-07 - Phase 3 first validation reviewed and corrections applied

- Reviewed user-run `validation/logs/phase3-20260807-045558` read-only. It completed 18 commands:
  11 passed and 7 failed. Compilation, Ruff diagnostics, active v12 identity, Git integrity/ignore
  rules, and all 35 Phase 2 presentation regressions passed. The two GUI test groups totaled 60
  passes and two formatting-sensitive assertion failures.
- All four requested Windows offscreen renders failed safely. Qt could not discover a font in the
  PyQt runtime; screenshots visibly contained tofu for ordinary Latin text. Each report recorded
  zero compact-input violations, zero oversized editable long-text violations, and zero Unicode
  replacement characters, but 1,344-1,380 missing-glyph instances and 4-7 clipping candidates.
- Verified the 30-file source manifest against current source with zero mismatches and zero missing
  paths. Of 47 retained evidence hashes, 34 readable durable artifacts matched; 13 pytest-temp
  artifacts were inaccessible after the run, exposing a durability defect rather than an accepted
  hash result. No policy, qualification, benchmark, campaign, or protected-case workflow ran.
- Corrected the two brittle tests to use behavioral/AST contracts. Applied Ruff formatting as a
  mechanical source rewrite to the twelve reported files; this was formatting, not a check.
- Added license-safe runtime font resolution: native fonts are retained when usable; otherwise Qt
  explicitly registers an existing Windows, Linux, or macOS system font. Font family, source,
  registration, and validation-sample support are now retained in GUI evidence. No font file was
  copied or redistributed.
- Removed unsupported `font-variant-numeric` QSS, strengthened packaged-GUI font rejection, and
  changed render failures to report separate glyph, replacement, clipping, compact-input, and
  long-editor counts.
- Upgraded the ignored validator source manifest to v2 with exact commit, dirty flag, Git-status
  line count/hash, validator hash, and source-manifest hash in the final summary. Ephemeral
  `pytest-temp` contents are explicitly excluded from durable evidence hashes.
- Active status binds the failed run and records corrections as revalidation-pending. Codex did not
  execute Ruff checks, pytest, compilation, rendering, the validator, policy workflows, benchmarks,
  campaigns, qualification, or protected cases while correcting it.
- Next action: manually rerun `validation/Validate-Phase3.ps1` and return the complete new
  `validation/logs/phase3-YYYYMMDD-HHMMSS/` directory. Do not overwrite or reinterpret the failed
  run. Linux rendering, keyboard interaction, and scientist acceptance remain separate open proof.

### 2026-08-07 - Phase 3 Windows evidence accepted; remaining blockers corrected in tooling

- Accepted `validation/logs/phase3-20260807-052047` read-only: all 18 commands passed, including
  62 Phase 3/GUI tests, 35 Phase 2 regressions, current identity/compile/Ruff gates, and four
  readable Windows light/dark/high-DPI render cells with zero glyph, replacement, clipping,
  compact-input, or long-editor failures.
- The run retained commit `00b8ee07a6d59c0d805d0c043c91ae5ea73d45d0`, dirty state, a 31-line
  Git status and its hash, 34/34 durable evidence hashes, and 32/32 current source hashes. It
  recorded no policy training/evaluation, qualification, benchmark, campaign, or protected case.
- Updated active status and its exact source contracts so the accepted Windows run is no longer
  misreported as pending. The failed `phase3-20260807-045558` record remains immutable history.
- Added a tracked non-scientific all-workspace keyboard/accessibility evidence collector and enabled
  explicit Linux xcb selection in the existing render CLI. The collector exercises presentation
  controls only and records that it neither executes nor qualifies scientific or policy work.
- Prepared ignored Windows and Linux remaining-gate validators plus detailed instructions. The
  Windows lane combines current source checks with mandatory human keyboard, screen-reader, and
  scientist attestations. The Linux lane uses xcb under Xvfb for light/dark dashboard and all-sixteen
  workspace evidence. Both bind current source identity and retain detailed logs/hashes.
- No test, validator, compiler, Ruff command, GUI interaction, render, policy/scientific workflow,
  benchmark, qualification, campaign, or protected case was executed by Codex.
- Next action: the user runs both commands documented in
  `validation/PHASE3_REMAINING_VALIDATION.md` and returns both complete timestamped directories.
  Phase 3 and Phase 4 remain blocked until both pass evidence review.

### 2026-08-07 - Remaining Phase 3 clipping corrected; reviewer prompts removed

- Reviewed `phase3-remaining-windows-20260807-092741` without rerunning it. The source-bound run
  failed at 5/8 automated commands: three files required Ruff formatting and Results Explorer plus
  Application Settings failed clipping in both light and dark/200% all-workspace cells.
- Verified 45/45 durable hashes, 32/32 current source hashes, exact validator and Git-status hashes,
  and no policy/scientific evaluation, benchmark, qualification, campaign, or protected case. The
  supplemental Phase 2 rerun reported only the same formatting failure and does not reopen Phase 2.
- Reorganized Results filters into a four-column labeled grid with a separate action row, preventing
  the 193px `Open experiment workspace` action from shrinking to 113px. Replaced the Settings
  database form row with a dedicated accessible, copyable, read-only full-width path field.
- Made clipping records self-describing with displayed text, accessible name, tooltip, and actual/
  preferred dimensions. Added source and focused behavior contracts for both responsive layouts.
- Replaced the ignored Windows validator with a fully noninteractive v2 lane. It collects no
  reviewer name, role, screen-reader declaration, PASS/FAIL selection, or evidence note. Objective
  keyboard, semantic accessibility, contrast, clipping, glyph, input, scroll, terminology and render
  checks are the only acceptance basis; human screen-reader/scientist acceptance is not inferred.
- Updated repository instructions to prohibit interactive reviewer answers in future phase
  validators. Preserved the failed run and its `screen_reader=NA` answers as invalid historical
  evidence rather than carrying them forward.
- Used Ruff format only as a mechanical rewrite: two files were formatted before it exposed an
  invalid hyphenated test identifier; that identifier was corrected and the remaining file was
  formatted. A final two-file GUI-test formatting pass changed nothing. No lint/format check or test
  was executed.
- No validator, test, compile, Ruff check, GUI render/interaction, policy/scientific workflow,
  benchmark, qualification, campaign, or protected case was executed by Codex. Run the corrected
  ignored Windows validator and return its complete timestamped directory before Linux xcb closure.

### 2026-08-07 - Phase 3 automated Windows pass accepted; tabbed layout refinement implemented

- Accepted `phase3-remaining-windows-20260807-112621` for its exact source: 10/10 commands passed,
  including 13 Phase 3 contracts, two focused GUI regressions, and light plus dark/200% evidence for
  all sixteen workspaces. All 47 durable hashes and 36 current source hashes matched; no prohibited
  policy, scientific, benchmark, qualification, campaign, or protected-case workflow ran.
- Recorded the accepted run as pre-refinement evidence after the user requested another layout pass.
  It remains valid history but cannot prove the subsequently changed GUI source.
- Added a reusable accessible `WorkspaceTabs` surface and replaced stacked sections in ORPD
  Formulation, Robust Scenarios, Portfolio Manager, Application Settings, and Benchmark & Evidence.
  Related compact controls now occupy balanced side-by-side groups; long database and manifest paths
  remain dedicated full-width values.
- Added runtime/source contracts for tab titles, accessible metadata, and adoption across all five
  workspaces. Updated the workspace evidence schema to v2 so every shared tab is keyboard-selected,
  rendered, hashed, and recorded in light plus dark/200% Windows and Linux xcb lanes.
- Updated the Git-ignored noninteractive validator and instructions. No tests, validators, Ruff,
  compilation, renders, GUI interaction, scientific/policy workflows, benchmarks, qualification,
  campaigns, or protected cases were executed by Codex. Current-source Windows and Linux evidence
  remain pending.

### 2026-08-07 - Phase 3 tabbed-layout validation reviewed; table-width gate corrected

- Reviewed `phase3-remaining-windows-20260807-120240` read-only. The source-bound noninteractive run
  passed 9/10 commands: 14 Phase 3 contracts, five responsive-layout regressions, and both light and
  dark/200% all-tab evidence cells passed. All 79 retained evidence hashes and all 17 source records
  matched, with no missing or mismatched file and no prohibited workflow.
- Preserved the run as failed evidence because Ruff formatting rejected five files. Visual review
  also found that Portfolio Manager's Requested outputs tree retained a large unused right region
  while its Minimum evidence header and cells were shortened; the v2 collector did not inspect tree
  column allocation.
- Corrected the Portfolio tree with a fixed selection column, a stretching Output column, and a
  content-sized Minimum evidence column. Added stable object/accessibility identity and a rendered
  GUI regression for the resize policy, viewport use, and header fit.
- Advanced the all-workspace evidence schema to v3. Each visited tab now audits visible tree widgets
  for unused width, horizontal overflow, header/cell fit, and retained failure details. Advanced the
  ignored Windows summary to v4 and included tree-layout counts/failures in each evidence cell.
- Mechanically formatted only the affected Python implementation/test files; no format-check gate,
  validator, test, compilation, render, GUI interaction, policy/scientific workflow, benchmark,
  qualification, campaign, or protected case was executed. A fresh user-run Windows directory is
  required, followed by the outstanding Linux xcb evidence.

### 2026-08-07 - Phase 3 corrected Windows automation accepted; correction goal closed

- Accepted `phase3-remaining-windows-20260807-121530` read-only for its exact dirty source at commit
  `bf3a51a970555e44411121e145e6f5a24f41989d`: all 10 commands passed. The run includes 15 Phase 3
  contracts, six responsive-layout regressions, and light plus dark/200% v3 workspace evidence.
- Verified the evidence boundary independently: the current validator matches retained SHA-256
  `e49d3b1d125b0753e8849e5200c48885a038ca73868e40686d69452273f93eec`; current Git-status identity
  matched the retained status hash before this acceptance-ledger update; all 17 returned source
  entries and all 79 evidence hashes matched with zero missing or mismatched files. The status,
  verification contract, test expectation, and ledgers now deliberately postdate that manifest.
- Both workspace cells passed five tab sets, sixteen section screenshots, keyboard/contrast checks,
  and the Portfolio tree-width gate. The tree uses all 1054 rendered viewport pixels with zero
  unused or overflow pixels; all three columns meet their retained content widths in light and
  dark/200%. Screenshot review confirms the former blank-right-space and text truncation are gone.
- The Phase 3 Windows correction goal is complete and closed. The overall Phase 3 gate remains open
  solely for a separate returned Linux xcb evidence directory; Windows evidence does not infer Linux
  rendering or human screen-reader/scientist acceptance and does not qualify scientific, policy,
  performance, protected-case, release-candidate, or final-release claims.
- No test or validation command was rerun after the post-evidence acceptance-ledger update. The next
  user-run Linux xcb lane will bind those ledger-only identity changes together with current source.

### 2026-08-07 - Phase 4 redesigned around development completion; policies deferred

- Recorded the user's controlling boundary: no old policy will be treated as final, reused as the
  final candidate, trained, evaluated, qualified, registered, activated, or packaged. Old policies
  remain development-only/unqualified/inactive until a separately authorized post-development
  deletion action.
- Replaced the training/qualification-oriented Phase 4 design with coding-only production completion:
  final A-E/F-off-capable semantics, empty-policy operation, runtime/CUDA/container/package/CI
  hardening, old-policy inventory and dry-run deletion tooling, and a source-bound development freeze.
- Phase 4 explicitly prohibits policy training/evaluation, candidate creation, qualification,
  protected campaigns, policy deletion, registration, activation, release-manifest generation, and
  release claims. Its validator must be noninteractive, Git-ignored, and limited to development proof.
- Added a separately controlled post-Phase 4 transition: review and authorize exact old-policy
  deletion, verify an empty policy store, freeze a new A-E/F-off plan against the completed source,
  train and independently qualify a completely new policy, then choose a newly-qualified-policy or
  policy-free Phase 5 scope.
- Updated Phase 5 to package no old policies and to bind one immutable source identity plus, only when
  included, a separate newly qualified policy checksum. Phase 4 now produces no RC; Phase 5 owns RC
  and final-release identities.
- This was a design-only documentation/status update. Phase 4 did not start, no Phase 4 goal was
  created, no policy file was deleted, and no test, validator, training, evaluation, qualification,
  protected case, benchmark, container, packaging, or release command was executed.

### 2026-08-07 - Repository instruction and Markdown alignment completed

- Audited all 60 live `AGENTS.md` files and separated them from 60 copied instruction files under
  retained baseline/wheel-smoke artifacts. The copied files remain immutable evidence; root and
  relevant live scoped instructions now consistently enforce Phase 4 development-only work,
  empty-policy fixtures, non-executable XPU, no old-policy reuse/deletion, and the separately
  authorized post-freeze transition.
- Reviewed the repository's 252 Markdown files by current/historical role, including the new status
  index added by this alignment pass. Dated audits, versioned
  implementation reports, patch notes, release validation, and built/baseline copies remain
  historical evidence rather than being rewritten as current claims.
- Added `docs/DOCUMENTATION_STATUS.md` as the routing and precedence index. Updated current-facing
  README, user guide, architecture, methodology, reproducibility, container, portfolio, throughput,
  scientific-protocol, and architecture-proposal documents. Ambiguous v5/v5.6/v6.9 operational
  records now carry historical-status or current-scheduling notices.
- Reclassified the old candidate-training paragraph in requirement traceability as a superseded
  historical route. Current next work remains: obtain the separate Phase 3 Linux xcb evidence, then
  create a Phase 4 goal before any Phase 4 source development.
- This was instruction/documentation maintenance only. Phase 4 did not start; no goal was created;
  no policy was changed or deleted; and no test, validator, lint, format, compile, schema, GUI,
  container, packaging, training, evaluation, qualification, campaign, benchmark, protected-case,
  or release command was executed.

### 2026-08-07 - Separate Phase 4 and Phase 5 new-chat prompts prepared

- Added `PHASE_4_NEW_CHAT_PROMPT.md` with a fail-closed Phase 3 prerequisite check, mandatory
  phase-specific goal creation, the complete development/empty-policy/old-policy-removal-preparation
  scope, coding-only restrictions, four-ledger updates, and a noninteractive ignored validator
  handoff that Codex must not execute.
- Added `PHASE_5_NEW_CHAT_PROMPT.md` with a strict Phase 4 and post-freeze transition start gate. It
  cannot silently perform old-policy deletion, new-policy training/qualification, or the release
  policy-scope decision; it accepts only an already recorded policy-free or exact newly-qualified-
  policy scope.
- The Phase 5 prompt separates release-engineering implementation from user-executed validation and
  requires explicit authorization before tag, push, publication, or release.
- Prompt drafting did not start either phase, create a phase goal, modify/delete a policy, or run a
  test, validator, build, container, policy, scientific, protected-case, or release command.

### 2026-08-12 - Phase 3 Linux boundary accepted by project owner; Phase 4 started

- The project owner explicitly accepted the manually validated Linux xcb boundary and directed the
  project to continue to Phase 4. No automated `phase3-remaining-linux-*` directory was returned or
  retained, so this is an owner-approved scope/gate decision rather than reproducible automated
  Linux evidence. It does not infer human screen-reader, usability, or scientist acceptance and does
  not qualify policy, scientific, hardware-performance, release-candidate, or release claims.
- Phase 3 is closed by that explicit owner decision. The accepted Windows evidence remains the
  retained automated GUI evidence; Linux xcb remains manually accepted without a retained automated
  evidence bundle.
- Created the required Phase 4 goal before starting source development. Objective: complete v12
  production development, empty-policy hardening, engineering/release-infrastructure completion,
  controlled old-policy removal preparation without deletion, authoritative record updates, and a
  source-bound manual validator.
- Phase 4 started from clean commit `f800119cd3a14e2965c91040d0a8392013532089`. Phase 4 remains
  coding-only: no manual-capable test/validator/build command and no policy training, evaluation,
  qualification, registration, activation, protected campaign, or deletion is authorized.

### 2026-08-12 - Phase 4 coding completed; manual validation pending

- Completed the production integration gap that kept policy-gated TSH-CALO outside the ordinary
  experiment validator/runner/Algorithms workflow. TSH-CALO is now a separate gated algorithm and
  requires an immutable qualified active binding; Change F and experimental flags remain disabled.
  Primary CALO remains rule-only when the policy store is empty.
- Hardened empty-policy behavior. Application state removes stale CALO and TSH-CALO checkpoint,
  checksum, qualification, calibration, receipt, ensemble, and training-provenance fields; forces
  both policy-use flags off; and never fabricates or auto-selects a policy. The GUI no longer scans
  bundled/checkout `.pt` files. Existing/pre-freeze records remain inspectable but cannot activate,
  bind, qualify through the historical paired-CALO action, resume/fork as initialization, or delete.
- Added new-policy provenance fields for exact development-freeze commit, retained freeze payload
  SHA-256, and empty initialization checksum. Only completely new post-development TSH-CALO
  ensembles pass registry readiness. A future training plan/command must supply and validate the
  exact clean, empty-policy, post-transition freeze report; a dirty Phase 4 evidence candidate
  cannot authorize training.
  The production inference loader independently repeats this source/initialization gate, so a
  hand-authored serialized binding cannot bypass the registry; historical artifacts remain
  inspectable only and non-production qualification authorities retain their explicit scope.
  Experiment policy inference is bound to the scheduler's resolved CPU/CUDA device and forbids
  internal or baseline fallback; exploratory CUDA failure must be a separately recorded full-request
  CPU restart.
- Added exact policy-lifecycle database snapshots plus inventory-bound transactional cleanup, and
  added `policy_retirement.py` / `manage_policy_retirement.py`. Phase 4 uses their read-only inventory,
  dry-run plan, and disabled authorization template only. The later execution path requires unchanged
  files/database, matching plan hashes, clean full source identity, matching retained authorization,
  the accepted Phase 4 freeze payload SHA-256, explicit freeze/irreversible/database
  acknowledgements, path confinement, and a receipt outside the policy store. No policy record,
  row, or file was deleted or altered in this coding pass.
- Added the development-only `calo-rpd-development-freeze` report. It hashes the source identity,
  Phase 4 validator, policy inventory, A-E/F-off interfaces, schemas, dependency locks, containers,
  and exclusion rules while recording zero release-scope policies and explicit prohibited claims.
  It is not a final freeze, release manifest, qualification receipt, RC, or release-ready statement.
- Extended distribution-required source, kept trained policies/validation excluded, and changed the
  physical-CUDA CI label/scope to policy-free engineering evidence. Removed policy hot-path
  evaluation from the Phase 4 physical lane.
- Added proportional Phase 4 test source for empty-policy state, gated TSH-CALO configuration,
  Change-F/internal-fallback refusal, new provenance, inventory/path confinement, authorization and
  synthetic retirement receipts, and development-freeze source binding. Updated affected historical
  activation fixtures without weakening old negative evidence.
- Kept the generated experiment schema structurally open for registry-defined algorithm parameter
  maps while publishing the safe rule-only CALO default. Legacy GUI training, recovery,
  continuation, resume, fork, and paired-qualification entry points now fail closed in their
  callbacks as well as remaining disabled, so worker completion cannot re-enable an old-policy path.
- Created the noninteractive Git-ignored `validation/Validate-Phase4.ps1` and
  `validation/PHASE4_VALIDATION.md`. The harness creates a new timestamped directory and covers
  source/schema/style/type gates, targeted Phase 4 and non-policy engineering/GUI tests, actual
  old-policy inventory and dry-run, disabled authorization, wheel/sdist/clean-install checks, CPU/CUDA
  containers, physical case30/case57 parity and batching, resource recovery, source/interface freeze
  identity, and detailed command/source/evidence hashes. It explicitly excludes Linux xcb because
  the project owner accepted that boundary manually before Phase 4; it does not infer human
  accessibility, usability, or scientist acceptance.
- The final source-to-validator audit added direct built-image-manifest policy/training/validation
  exclusions, broader artifact-format and Docker-context exclusions, complete tracked plus
  untracked source hashes, start/end Git-state stability, and nested instruction/package-marker
  protection in retirement plans. Broad policy-training and qualification-analysis regression files
  were removed from the Phase 4 validator to preserve the owner's no-policy-test boundary. Durable
  source identity and future training/qualification/ablation entry points now treat non-ignored
  untracked files as dirty source; ignored validation/log/policy artifacts remain excluded.
- The retained development-freeze candidate now embeds its own sorted SHA-256 manifest of every
  Git-tracked and non-ignored untracked source file, plus the raw Git-status SHA-256/clean state. Its
  parser requires the complete declared interface/dependency/container/exclusion contract, exact
  policy-empty counts and authority boundaries, and the full prohibited-workflow record. A
  structurally incomplete but self-hashed JSON file can no longer be used as later training
  authority. The ignored validator and its instructions remain separately declared and hashed.
- Added a separate `calo-rpd-accept-development-freeze` decision command. It cannot run validation
  or self-accept a candidate: it accepts only a complete, fully passing `phase4-*` directory with
  all 32 exact result IDs produced by the 30 numbered stages and whose every retained file matches
  its hash manifest, then writes a non-overwriting
  receipt outside that immutable run under an explicit decision ID. The receipt binds a production-
  source content contract that excludes only development-policy artifacts, allowing the separately
  authorized old-policy cleanup without losing the accepted code identity. Future training plans,
  candidates, registry readiness, and experiment bindings require the receipt SHA-256 in addition
  to the later clean/empty post-transition freeze SHA-256. No acceptance receipt was created in this
  coding pass; it is available only after returned-log review and an explicit closure decision.
- Codex used Ruff format only as a mechanical source rewrite. It did not run the validator or any
  Ruff check, test, compilation, schema check, lint, mypy, package, GUI, browser, Docker, CUDA,
  policy, benchmark, campaign, protected-case, qualification, or release command. Phase 4 and its
  goal remain open until the user runs
  `& .\validation\Validate-Phase4.ps1`, returns the complete new `phase4-*` log directory, and its
  source/hash-bound evidence is reviewed and accepted.

### 2026-08-12 - Phase 4 final source-only completion audit

- Corrected the governing-policy readiness record construction so a pre-freeze active record fails
  closed as `development_only` without a positional-argument exception. All active-record status
  branches now use one keyword-labelled constructor, and regression source covers the pre-freeze
  boundary. No policy workflow was executed.
- Changed the GUI removal-plan export to open a dedicated SQLite read-only database handle rather
  than borrowing the application's writable handle. The GUI remains inventory/dry-run only and
  cannot execute deletion.
- Tightened the accepted production-source contract: only generated records beneath the designated
  `calo_rpd_studio/data/trained_models/` store may be excluded during a later authorized retirement.
  A policy-looking filename elsewhere remains ordinary source content and cannot evade the accepted
  source digest. Regression source records that boundary.
- Updated `ACTIVE_DEVELOPMENT_STATUS.json`, the active-version verifier, Phase 3/4 contracts, the
  five-phase plan, gates, and traceability so the current state is Phase 4 with Phase 3 closed by
  owner manual Linux/xcb acceptance. The records still state that no automated Linux evidence bundle
  was retained and do not infer human accessibility, scientist acceptance, or release evidence.
- Completed the final static requirement-to-source inspection and applied Ruff formatting only as a
  mechanical rewrite. `git diff --check` reported no whitespace errors. No validator, pytest,
  compile, schema, Ruff check, type check, package build, GUI test, container, CUDA check, policy
  workflow, protected case, or release command was run by Codex.
- The Phase 4 development implementation and its ignored manual harness are complete. The Phase 4
  acceptance gate remains intentionally unevidenced until the user runs
  `& .\validation\Validate-Phase4.ps1` and returns the complete new log directory. A passing run is
  still not self-acceptance; the separate receipt is created only after read-only review and an
  explicit acceptance decision.

### 2026-08-12 - Phase 5 release-preparation development completed; combined validation pending

- The project owner explicitly changed sequencing: complete Phase 5 coding first, then manually
  validate Phase 4 and Phase 5 together. The Phase 4 development goal was closed and a distinct
  Phase 5 goal was created before Phase 5 source edits. This does not mark Phase 4 or Phase 5
  validation accepted.
- Added `release_policy_scope.py`. Its disabled template authorizes nothing. An approved decision
  must choose exactly `policy-free` or `newly-qualified-policy`; forbids old-policy reuse and F;
  binds Phase 4 acceptance, clean post-transition freeze, and production-source contract; and, for
  a policy scope, verifies the exact confined artifact, policy manifest, and immutable parsed
  qualification receipt.
- Added `create_release_preparation.py`, distinct wheel/sdist member-manifest generation, and CI
  contract verification. The preparation candidate hashes Python distributions, both archive
  member manifests, CPU/CUDA image identities, Buildx metadata, SBOMs, complete vulnerability
  reports, actual image filesystem manifests, scanner/database identity, independent wheel/sdist
  clean installs, packaged GUI evidence, and CI scope. It always records RC, final, release-ready,
  publication, policy-benefit, superiority, protected-case, and human-acceptance claims as false.
- Declared PyTorch as a package runtime dependency and required every new Phase 5 trust module in
  wheel/sdist verification and CI's typed boundary. CI generates distinct distribution manifests;
  existing CPU/CUDA maximum-provenance, SBOM, security, GUI, compatibility, and explicit-dispatch
  physical-CUDA jobs remain the execution authorities.
- Added `finalize_release_records.py`. Development may create its disabled authorization template,
  but final metadata/source-manifest generation requires a clean already reviewed `12.0.0` release
  identity, passing combined evidence on that exact commit, release preparation rebuilt for the
  exact approved scope, and a self-hashed explicit authorization. It cannot version-promote,
  commit, tag, push, publish, upload, or release.
- Added deterministic Phase 5 test source without executing it. Added ignored noninteractive
  `Validate-Phase5.ps1` and `Validate-Phase4-And-Phase5.ps1`. The wrapper runs full Phase 4 first,
  stops on failure, then runs Phase 5, requires identical Git commit/dirty state, and retains two
  full phase directories plus a combined hash-bound directory.
- Codex used Ruff format only as a mechanical rewrite. No test, compile, schema, Ruff check, mypy,
  build, install, GUI, Docker, Trivy, CUDA, policy, protected-case, final-record, tag, push,
  publication, or release command was executed. The exact next action is
  `& .\validation\Validate-Phase4-And-Phase5.ps1` and return all three new directories.

### 2026-08-12 - First combined run interrupted; source and harness corrections applied

- The owner-executed combined run created `phase4-20260812-165006` and
  `phase4-phase5-20260812-165006`. Phase 5 never started. The owner stopped the run while the Phase
  4 CUDA image command was still buffered; the partial directory has no complete acceptance
  summary and is not accepted evidence.
- Retained command logs showed one Ruff-format mismatch, 15 typed-boundary diagnostics, two
  engineering contract failures, and three GUI contract failures. Source/test corrections now
  cover the typed returns and heterogeneous SQL values, the active Phase 5 status contract, a
  deterministically tampered retirement plan, the seven-step stacked workflow, requested-output
  header width, and ordinary scientist wording.
- The original PowerShell command wrapper incorrectly converted normal native stderr (including
  Docker progress) into terminating exceptions, continued after failures, buffered all output, and
  passed Compose v5 an unsupported `run --read-only` option. Both phase validators now stream
  stdout/stderr to the terminal and per-command log, decide success by native exit code, and stop at
  the first failure. The combined wrapper streams both child validators and uses exact phase
  directory matching. Phase 4 Compose smoke inherits `read_only: true` from `compose.yaml`.
- Packaging and CPU/CUDA results from the interrupted run are not reusable proof because the faulty
  wrapper aborted native commands on stderr and later commands ran after failed prerequisites.
  No validator, test, format/lint/type/schema/compile check, package build, Docker/CUDA/Trivy
  command, policy workflow, protected case, or release action was executed by Codex during these
  corrections. A fresh owner-executed combined run is required.

### 2026-08-12 - Corrected fail-fast validator confirmed; eight files mechanically formatted

- Owner run `phase4-20260812-182252` used corrected validator SHA-256
  `c61e7c5a920f4ad420835d6fb99b06df014e46acb18e54daf6eba575be92e1f3`. Commands 01-05 passed:
  environment capture, active-version identity, compile, generated schema, and Ruff source checks.
  Command `06-format` failed and stopped Phase 4 immediately; Phase 5 did not start.
- The complete Phase 4 and combined summaries/hash manifests were retained. They record no policy
  workflow, real-policy deletion, protected case, release generation, RC, final release, or
  publication. The run is failed evidence, not Phase 4 acceptance.
- Ruff's formatter was applied mechanically to exactly the eight files named by `06-format`:
  policy registry, experiment configuration/runner, CALO Intelligence, Portfolio Manager, result
  database, distribution verifier, and GUI startup test. Codex did not execute the follow-up format
  check or any test, validator, build, Docker/CUDA/Trivy, policy, protected-case, or release command.
  A fresh combined owner run is the next authority.
- The combined wrapper was additionally corrected to embed a failed child's retained path, exit
  code, hashes, source identity, command counts, and explicit `passed: false` state. The prior
  `phase4: null` summary did not lose the child evidence, but made the combined record less useful.
  Passing requirements are unchanged.

### 2026-08-12 - Type and engineering gates pass; one scientist-facing GUI phrase corrected

- Owner run `phase4-20260812-182752` passed commands 01-08: environment, version, compile, schema,
  Ruff, formatting, all 15 typed-boundary files, and 112 Phase 4/non-policy engineering tests.
  Command `09-gui` passed 36/37 tests and stopped on the single ordinary-interface language gate.
  Phase 5 did not start.
- The only hit was `development` in the disabled qualification button tooltip: “independent
  post-development qualification authority.” It now says “independent post-freeze qualification
  authority.” Related normal policy/campaign tooltips and dialogs now consistently use historical,
  pre-freeze, post-freeze, or accepted-source wording. Internal provenance/field names are unchanged.
- The combined summary correctly embedded the failed Phase 4 child path, hashes, source identity,
  exit code, and command counts. No policy workflow, deletion, protected case, release, RC, final
  release, or publication occurred. Ruff mechanically formatted the three GUI files affected by the
  language correction; the status-contract test already matched formatter output. Codex did not
  execute any test, check, or validator after the correction; a fresh combined owner run remains
  required.

### 2026-08-12 - Distribution verifier package-namespace false positive corrected

- Owner run `phase4-20260812-184454` passed 14 result IDs through the fresh wheel/sdist build,
  including all 37 GUI tests and the read-only policy inventory/plan/disabled-authorization
  boundaries. Command `14-distribution` then rejected the legitimate packaged source
  `calo_rpd_studio/validation/__init__.py` as though it were the Git-ignored root `validation/`
  evidence directory. Phase 5 did not start.
- Distribution and container filesystem gates now reject root `validation/`, `validation_logs/`,
  and sdist-prefixed root equivalents while allowing the application namespace
  `calo_rpd_studio/validation/`. Distribution verification additionally requires its initializer
  and GUI contract module. Focused regression source covers accepted application paths and rejected
  local-evidence paths for both archives and images.
- The failed run remains non-acceptance evidence and records no policy workflow, deletion,
  protected case, release, RC, final release, or publication. Codex did not rerun packaging,
  tests, containers, or validation after correction. Ruff mechanically formatted the two corrected
  release-boundary modules and affected regression source (four files changed; the status-contract
  test was already formatted). A fresh combined owner run remains required.

### 2026-08-12 - Clean-install smoke assertion corrected after fifth combined attempt

- Owner run `phase4-20260812-185135` passed 17 result IDs, including distribution verification and
  installation of the built wheel with runtime dependencies into the isolated clean environment.
  Its first failure was `17-clean-smoke`; Phase 5 did not start.
- The command already ran from `validation/logs/.../clean-install`, but the assertion rejected every
  path beneath the repository. Because the intentionally Git-ignored clean virtual environment is
  itself beneath that directory, its valid `Lib/site-packages/calo_rpd_studio` installation was
  rejected even though it was not the checkout source package.
- The ignored Phase 4 validator now requires the imported module to reside beneath that run's clean
  environment and separately forbids resolution beneath the checkout `calo_rpd_studio` source
  directory. The installed entry-point assertions remain unchanged. Codex did not execute the
  corrected smoke, tests, or validator; preserve the failed evidence and perform a fresh combined
  owner run.

### 2026-08-12 - Physical-CUDA development evidence tier corrected after sixth attempt

- Owner run `phase4-20260812-190643` passed 24 result IDs, including clean installation, the
  checkout-independent lifecycle CLI, locked CPU and CUDA image builds, both container smokes, and
  physical NVIDIA discovery. The CUDA container reported CUDA available with no packaged policy or
  local validation content; the host reported an RTX 4060 Laptop GPU. The first failure was
  `23-cuda-parity-30`, before numerical parity executed, because the dirty development worktree did
  not qualify for the command's durable-evidence source tier. Phase 5 did not start.
- The strict default remains unchanged: retained durable qualification evidence requires a full
  clean source identity. Physical parity, CUDA time-share, and resource-recovery tools now expose a
  separate explicit `--allow-dirty-development-evidence` option. It still requires a full commit,
  records dirty state, emits `evidence_tier=development-only`, keeps
  `durable_evidence_eligible=false` and `qualification_passed=false`, and permits exit success only
  when the underlying engineering checks pass.
- The Git-ignored Phase 4 validator alone opts into that development tier because its freeze binds
  the complete tracked plus non-ignored source manifest and later proves source stability. Formal
  CLI/CI callers retain clean-source qualification semantics. Codex did not execute formatting,
  tests, CUDA work, containers, or either phase validator after this correction; fresh combined
  owner evidence remains required.
- A forward audit corrected the same repository-root assumption in the ignored Phase 5 wheel,
  sdist, and packaged-GUI checks before another long run could encounter it. Wheel/sdist imports must
  be inside their exact isolated environments and outside the checkout source package; packaged-GUI
  validation forbids that source package rather than the enclosing Git-ignored validation storage.
- The combined ignored wrapper now resolves Python, Docker, NVIDIA-SMI, and Trivy before Phase 4
  starts and retains a preflight log. This makes the owner's already observed missing-Trivy blocker
  fail immediately instead of after expensive container builds; installation remains a separate
  manual environment action.

### 2026-08-12 - Phase 4 passes; Phase 5 typed boundary corrected

- Owner run `phase4-20260812-195901` passed all 32 required Phase 4 result IDs with zero failures.
  The retained summary binds commit `f800119cd3a14e2965c91040d0a8392013532089`, records the exact
  dirty development source state, and confirms that no policy workflow, real-policy deletion,
  protected case, RC, final release, or publication occurred. It is a successful checkpoint for
  that exact source state; the subsequent Phase 5 typing correction changed the source identity, so
  the current tree still needs a fresh combined run. The separate explicit acceptance receipt and
  clean final-source gates remain pending.
- Phase 5 run `phase5-20260812-201822` passed its first five gates: active development identity,
  compilation, generated schema, Ruff lint, and formatting over 458 files. It failed first at
  `06-types` with two findings: PyYAML lacks inline typing metadata, and the Phase 5 summary JSON
  loader returned mypy `Any`. Combined run `phase4-phase5-20260812-195901` therefore failed and is
  not combined acceptance evidence.
- The CI workflow import now has a narrow `import-untyped` annotation with its existing runtime
  mapping checks retained. Final-record evidence loading now uses one fail-closed JSON-object loader
  that rejects malformed JSON, arrays, and scalars before returning a typed mapping; all six final
  record inputs use it. Regression source covers the non-object rejection.
- Codex did not execute mypy, tests, validation, builds, containers, policy work, or release work
  after correction. Phase 5 and combined validation require a fresh owner-executed run.

### 2026-08-12 - Active-version status contract aligned after eighth attempt

- Owner combined run `phase4-phase5-20260812-202511` passed the prerequisite preflight and Phase 4
  environment record, then stopped at `02-version`. The version, display version, project metadata,
  release boundaries, and policy boundaries all passed; only `active_status_runtime_contract` was
  false. Phase 5 did not start.
- The cause was an internal source mismatch: the status ledger had correctly advanced to record the
  seventh combined attempt and post-correction revalidation state, while `verify_active_version.py`
  still required the earlier pre-validation Phase 4 wording. The verifier now requires the current
  evidence-accurate core values and has a separate exact validation-attempt history check covering
  the seventh and eighth attempts. No release or scientific semantics changed.
- The failed run records one passed command and first failure `02-version`; it is retained negative
  evidence. Codex did not execute the corrected verifier, tests, or validators. Fresh combined
  owner validation remains required.

### 2026-08-12 - Phase 5 attestation prerequisite exposed after ninth attempt

- Owner combined run `phase4-phase5-20260812-202852` passed all 32 Phase 4 commands. Phase 5 run
  `phase5-20260812-204823` then passed 23 commands, including types, 543 tests, distribution
  manifests, clean wheel and sdist installation, packaged-GUI smoke, and isolated sdist smoke.
  Its first failure was `22-cpu-build`; Buildx rejected provenance/SBOM attestations because Docker
  Desktop was using its classic image store and `docker` driver.
- The Phase 5 build contract remains unchanged: both local images still require maximum provenance
  and SBOM attestations. The ignored combined wrapper now checks Docker's driver status before
  Phase 4 and fails immediately unless `io.containerd.snapshotter.v1` is present. It gives the exact
  Docker Desktop setting and restart instruction and records the result in `combined-preflight.txt`.
- This is an environment prerequisite plus a fail-fast harness correction, not a scientific or
  release-scope change. Codex did not switch Docker Desktop settings, execute tests/builds, run
  validation, or perform policy/release workflows. The owner must enable the containerd image store
  and run the combined validator again.

### 2026-08-13 - Combined Phase 4/5 validation passes; Phase 6 GUI goal created

- Owner combined run `phase4-phase5-20260813-000340` passed. Its Phase 4 child
  `phase4-20260813-000340` passed 32/32 commands and its Phase 5 child
  `phase5-20260813-010531` passed 41/41 commands with zero failures. Both children bind source
  commit `f800119cd3a14e2965c91040d0a8392013532089` and the same retained dirty source identity.
  The result is direct combined development/release-preparation validation; it does not choose the
  release-policy scope, authorize policy work, create a final release, or authorize publication.
- The project owner requested a new GUI-modernization goal after the combined validator finished.
  The current Phase 6 baseline is clean commit `2d7130fb63c13d35d0419dd63b1d68e2050dcf72`.
  The required goal was created before Phase 6 source development.
- Phase 6 will implement the saved `PHASE_6_NEW_CHAT_PROMPT.md` ribbon-workspace design using one
  command registry, compact contextual left editors, central result/preview tabs, and truthful
  jobs/logs/progress/status presentation. The supplied screenshot directly demonstrates two target
  defects: a disabled `PrimaryButton` is visually blue because the ID selector overrides the generic
  disabled theme, and a long single-row action layout clips training controls at constrained width.
- The owner also requires the training module to be usable and the application to run without
  Docker. This means an enabled entry to the existing independent completely-new-policy workflow,
  with explicit plan/check/start gates and no restoration of the obsolete embedded legacy trainer.
  Navigation must not itself start training. No auto-training, evaluation, qualification,
  registration, activation, policy selection, or deletion is permitted. Native Windows execution
  must be a first-class documented and packaged path alongside optional Docker CPU/CUDA operation.
- Phase 6 remains coding-only. Codex will implement production/test/documentation/validator source
  but will not execute manual-capable tests, GUI checks, Docker, policy training/evaluation, or the
  Phase 6 validator. One consolidated ignored noninteractive validator will be prepared at the end
  for owner execution.

### 2026-08-13 - Phase 6 development complete; owner validation pending

- The Phase 6 source pass now provides one immutable command registry and generated eight-tab icon
  ribbon, compact validated contextual editors, the pinned scientific workspace plus singleton
  documents, and dockable jobs/logs/warnings/device/provenance activity with truthful
  determinate/indeterminate progress and configured-versus-actual compute status.
- The visible legacy-training action row now routes to a separate independent new-policy
  plan/check/start document. Navigation performs no process action; a successful result is bound to
  the exact checked path fingerprint, later path edits fail closed, existing output requires an
  explicit authenticated-resume choice, and completed output remains unqualified and inactive.
  The obsolete embedded trainer stays hidden and disabled.
- Native Windows operation now has a repository launcher, direct installed-wheel entry point, and
  current guide while Docker remains optional. Distribution checks require the Phase 6 GUI/native
  members and reject root validation evidence and generated policy artifacts.
- `ACTIVE_DEVELOPMENT_STATUS.json` records
  `phase_6_coding = development_complete_manual_validation_pending`. Validation remains
  `pending_user_executed_noninteractive_phase6_validator`; no RC, final-release, release-policy,
  publication, protected-case, policy qualification, or activation state was advanced.
- The ignored `validation/Validate-Phase6.ps1` and `validation/PHASE6_VALIDATION.md` are complete.
  Codex did not execute Python, Qt, tests, lint, formatting, compilation, builds, Docker, the
  validator, or any policy/scientific workflow. Automated evidence has not yet been returned, and
  human usability, screen-reader, and scientist acceptance are not inferred.

### 2026-08-13 - First two Phase 6 runs stop in environment capture

- Owner runs `phase6-20260813-031026` and `phase6-20260813-031046` both stopped before command `01`
  with the same Windows PowerShell error: the chained `RuntimeInformation.ProcessArchitecture`
  access was parsed as an unavailable property. Both partial summaries are retained failed harness
  evidence, not Phase 6 validation.
- Each bundle binds commit `2d7130fb63c13d35d0419dd63b1d68e2050dcf72`, records identical
  before/after dirty-status SHA-256
  `beabf2d918c4717e61e3b9c12ba449fdf2e59a38ccd549da54e2fb74cbabe9bf`, and proves that
  nonignored source did not change. Every policy, protected-case, Docker, CUDA-campaign,
  publication, release, and inferred-human-acceptance field is false.
- The ignored validator now captures process architecture through the Windows-compatible
  `PROCESSOR_ARCHITECTURE` process environment value with an `Is64BitProcess` fallback and accesses
  `OSVersion` through a local object. Codex did not execute the correction. A fresh complete owner
  run is required; the two pre-command failures cannot be combined or accepted.

### 2026-08-13 - Phase 6 reaches format gate under Codex authorization

- After the owner authorized Codex to execute validation, sandboxed attempt
  `phase6-20260813-031632` could not launch the repository Python and stopped before command `01`.
  The same consolidated validator was then executed with the required process permission as
  `phase6-20260813-031655`.
- That run passed Python/dependency identity, whitespace diagnostics, all three Git-ignore checks,
  active-version verification, compilation, and Ruff diagnostics. It stopped at `08-format`, which
  reported exactly ten Phase 6 files requiring mechanical formatting. No tests, GUI renders, or
  builds ran after that first failure.
- Only the ten reported files are mechanically formatted before one fresh consolidated rerun. The
  retained failure records no Docker, CUDA campaign, policy, protected-case, publication, release,
  or inferred human-acceptance work.

### 2026-08-13 - Phase 6 GUI run exposes unresolved-device label

- Fresh run `phase6-20260813-031800` passed the preliminary gates, the corrected format gate, and
  all 21 Phase 6 unit contracts. The Phase 6 GUI suite passed 5/6 and stopped at one truthful-status
  assertion: the default compatibility value `runtime_compute_device = cpu` was displayed as an
  actual assignment even though `runtime_device_resolution` was empty and no resolver had run.
- Status, context, and activity surfaces now present `not assigned` unless a runtime resolution
  record exists. Changing configured compute intent also clears the previous resolution metadata so
  a stale actual assignment cannot be displayed under new intent. The scientific resolver,
  execution modes, fallback rules, and runtime binding semantics are unchanged.
- The retained failed bundle proves stable nonignored source state and records no Docker, CUDA
  campaign, policy, protected-case, publication, release, or inferred human-acceptance work. A
  fresh consolidated run is required.

### 2026-08-13 - Phase 6 GUI/native/packaging validation passes

- Consolidated run `phase6-20260813-032036` passed all 19/19 checks against commit
  `2d7130fb63c13d35d0419dd63b1d68e2050dcf72` plus its retained dirty source identity. Its exact
  before/after nonignored source-status SHA-256 is
  `beabf2d918c4717e61e3b9c12ba449fdf2e59a38ccd549da54e2fb74cbabe9bf` and the source-stability
  gate passed.
- Passing evidence covers active-version identity, compilation, Ruff and formatting, 21 Phase 6
  unit contracts, 6 Phase 6 GUI contracts, 21 affected GUI regressions, 9 empty-policy/training
  navigation integration tests, light/dark/constrained offscreen renders, fresh wheel and sdist
  creation, existing distribution safety, and Phase 6 native/GUI distribution membership.
- The bundle records policy training, evaluation, qualification, registration, activation, and
  deletion; protected-case work; Docker; CUDA campaign; publication; and release execution as
  false. Automated evidence does not infer human screen-reader, usability, or scientist acceptance.
- Phase 6 development and its automated GUI/native/packaging validation gate are complete. This is
  not a release-policy decision, policy authorization, release candidate, publication
  authorization, release-readiness claim, or final release.

### 2026-08-13 - Phase 6 professional visual refinement completed

- Workspace navigation now has a complete ribbon palette covering all 16 stable workspaces. The
  left dock contains contextual inputs only. A later focused follow-up makes that dock permanently
  visible and makes the ribbon permanently expanded.
- `Train policy` is one ribbon-owned stateful action: it focuses missing inputs, runs the explicit
  readiness check when inputs are complete, changes to `Start training` only for the unchanged
  checked input fingerprint, and then uses the existing confirmation and lifecycle safeguards. It
  no longer opens a redundant training document. No training was executed during refinement.
- The inactive ribbon-page visibility bug was removed, eliminating cross-tab bleed-through. The
  default Activity dock is compact, workspace labels remain legible at 1120x720, input forms stack
  in narrow panes, and common page subtitles plus obsolete CALO training/qualification prose are
  absent from normal presentation.
- Targeted ribbon/context/training GUI contracts passed 7/7. Offscreen shell rendering passed in
  light, dark, and constrained layouts, and `phase6-panel-sweep-20260813-041700` rendered all 16
  workspace panels plus four shell states for direct visual inspection. Publication overlap and
  remaining oversized prose blocks were corrected from that sweep. Policy, scientific,
  protected-case, Docker, publication, and release execution all remained false.

### 2026-08-13 - Phase 6 training inputs restored to the active workflow

- The visual-refinement pass had hidden the retired embedded trainer correctly, but its replacement
  exposed only four prerequisite paths. That made the active training action look incomplete.
- The single left Inputs pane now loads the selected frozen TSH-CALO plan and exposes its actual
  campaign, case, member, population, evaluation, compute, PPO, and model inputs. Edits change the
  readiness fingerprint and are materialized as a separate hash-addressed plan for the same
  readiness/start command; the source plan is never overwritten.
- The legacy center trainer remains hidden and disconnected. The existing ribbon action still
  performs load, readiness, and explicit confirmed start in sequence, and no policy workflow was
  run during development. The focused Phase 6 validator awaits owner execution.
- The Inputs dock is now permanently visible, left-docked, non-floating, and non-closable; the
  ribbon is permanently expanded. Their hide/compact commands were removed. The central scientific
  preview is hosted on a roomy scrollable canvas, and its taller document bar carries a dedicated
  CALO-RPD brand mark. Saved shell layout moved to version 3 so older hidden/compact state is not
  restored.

### 2026-08-13 - Automatic eligible training-case selection and native shell cleanup

- Free-text case entry is replaced by a bundled-case checklist with `All eligible bundled cases`.
  `case30` and `case57` are selected automatically. `case118` and `case300` remain visible as
  disabled protected holdouts and cannot enter a generated or edited training plan. Eligible custom
  development cases in a previously frozen plan remain representable as plan-bound checkboxes.
- The application and main window now receive a multi-resolution CALO-RPD icon. Ribbon, document,
  and Activity tab bars suppress their native base line, and both themes explicitly style main-
  window separators and use one controlled dock/activity boundary instead of the native gray lines.
- Source and focused-contract updates are complete. No GUI, validator, Docker, policy, protected-
  case, or scientific workflow was executed; the focused owner validator remains pending.

### 2026-08-13 - Training-input information controls added

- All 19 visible training inputs now have a compact circular `i` control beside their label. Hover,
  click, keyboard focus, and accessibility metadata expose concise explanations without adding
  descriptive paragraphs to the normal pane.
- Help covers identity/governance behavior and the directional effects of increasing, decreasing,
  enabling, disabling, or changing campaign, compute, PPO, and architecture values. Protected-case,
  exact-budget, readiness-invalidation, and unqualified-output boundaries remain explicit.
- Focused GUI/render/active-status validator source is updated but not executed. No policy,
  scientific, protected-case, Docker, qualification, activation, or release work occurred.

### 2026-08-13 - Ribbon heading paint remnant corrected

- The short colored fragment between the CALO-RPD product heading and Home category was an
  inactive ribbon page painting beyond the category-page viewport, not hidden text or a workspace
  control.
- Ribbon page visibility is now synchronized whenever the category changes and whenever an already
  selected category is requested again. Exactly the current category page remains visible, which
  prevents stale command-group edges from appearing in the identity-to-category gap.
- A focused regression contract and the ignored Phase 6 validator are updated but not executed.
  No GUI, policy, scientific, protected-case, Docker, qualification, activation, or release
  workflow was run.

### 2026-08-13 - Ribbon group captions contained

- Native `QGroupBox` titles had been repositioned onto the lower frame edge. On Windows they could
  paint partly outside the card and clip captions such as `Project` and `Navigate`.
- Every ribbon group now uses an untitled frame with a centered footer label inside its layout.
  Caption height, padding, colors, and accessibility names are explicit in both themes, so the
  correction applies consistently to every category and group.
- The focused GUI contract and ignored Phase 6 validator source include the contained-caption
  invariant. They were not executed, and no Docker, policy, scientific, protected-case,
  qualification, activation, or release workflow occurred.

### 2026-08-13 - Numeric steppers modernized

- Integer and decimal fields no longer depend on the low-contrast native Windows spin-arrow
  glyphs. A package-safe Qt proxy style draws antialiased palette-aware chevrons for both arrow
  directions, including distinct enabled, interactive, and disabled colors.
- Both themes give the stepper a 24-pixel button column, reserved value padding, a contrasting
  divider, rounded outer corners, and explicit normal, hover, pressed, and disabled surfaces. The
  treatment applies consistently to training inputs and other numeric controls.
- The focused source contract and ignored Phase 6 validator include the new style and theme
  requirements but were not executed. No GUI, Docker, policy, scientific, protected-case,
  qualification, activation, or release workflow occurred.

### 2026-08-13 - Training tooltips gain suggested ranges

- Every current training-input information control includes a separate suggested range,
  selection, scope, format, or choice. Numeric fields show low-to-high practical starting guidance,
  their frozen default where relevant, and the wider hard GUI limit.
- The guidance keeps exact constraints visible: evaluation budgets must be at least twice and
  divisible by population, resource envelopes may lower population ceilings, seeds must be
  predeclared rather than outcome-selected, and case118/case300 remain protected and excluded.
- Each tooltip states that suggestions are not policy-quality or qualification evidence and that
  changing a checked input requires a fresh readiness check. Focused contract and ignored validator
  source are updated but not executed; no training or other policy/scientific workflow occurred.

### 2026-08-13 - Central preview uses one page scrollbar

- Long workspaces embedded in the central scientific preview now delegate vertical navigation to
  `MainPreviewScroll`, eliminating the duplicate inner page scrollbar shown in CALO Intelligence.
- The shared long-page base advertises its full content height, ignores wheel events for propagation
  to the outer canvas, and retains its own as-needed scrollbar when instantiated outside the main
  preview. Purpose-specific table, editor, plot, activity, and log scrolling is unchanged.
- A focused shell contract and the ignored Phase 6 validator are updated but not executed. No GUI,
  Docker, policy, scientific, protected-case, qualification, activation, or release workflow ran.

### 2026-08-13 - Inactive ribbon controls made noninteractive

- While Compute was selected, part of the inactive Policies page remained visible above the
  category strip and its hidden `Train policy` button could still be clicked, producing the
  readiness-input dialog. The cause was direct `setVisible` manipulation of pages already owned by
  `QTabWidget`'s internal stack.
- Qt now exclusively owns ribbon-page visibility. Every inactive page is additionally disabled,
  mouse-transparent, and removed from keyboard focus; its command buttons retain their registered
  action state and are restored when their category becomes current. Ribbon pages also paint an
  explicit opaque theme background as a visual containment boundary.
- The focused contract identifies `policies.training` by command metadata and requires it to be
  invisible, pointer-transparent, and unfocusable while Compute is selected. Validator source is
  updated but not executed; the readiness dialog observed by the owner did not start training.

### 2026-08-13 - Ribbon product heading enlarged

- The product-identity strip above the ribbon categories is now a dedicated 42-pixel row instead
  of a tight anonymous layout. CALO-RPD Studio, the exact development version, and application
  state are vertically centered with explicit spacing and accessibility names.
- Dark and light themes give the row its own surface, lower divider, larger product typography,
  readable version treatment, and a contained state badge. It contains identity and status only;
  no command can occupy or receive input in the heading strip.
- The focused GUI contract and ignored Phase 6 validator source include the heading-height and
  identity requirements. They were not executed; no GUI, Docker, policy, scientific,
  protected-case, qualification, activation, or release workflow ran.

### 2026-08-13 - Ribbon heading boundary sealed

- The taller identity row made residual category-paint fragments below the product name fully
  visible. They originated from navigation content painting above its intended region, not from
  another heading control or the native Windows title bar.
- Ribbon tabs and pages now live inside a dedicated opaque navigation frame below the heading.
  The identity frame remains the uppermost sibling, so it seals both paint and pointer input at
  the boundary while preserving the permanently expanded ribbon.
- The focused contract requires the tabs to be parented by that navigation boundary and prohibits
  tool buttons inside the identity row. Validator source is updated but not executed; no GUI,
  Docker, policy, scientific, protected-case, qualification, activation, or release workflow ran.

### 2026-08-13 - Native ribbon overlay owner removed

- Owner inspection confirmed the fragment persisted after outer paint containment, so it was not
  treated as product artwork and the earlier boundary-only correction was insufficient.
- The ribbon no longer uses `QTabWidget`, whose private page and navigation controls could retain
  the misplaced interactive overlay. It now owns a plain `QTabBar` category selector and a
  separate `QStackedWidget` command-page region with the same public category API, keyboard focus,
  action bindings, fixed expansion, and inactive-page protections.
- Focused tests and the ignored validator require the explicit selector/stack structure and reject
  a native composite tab widget or command button in the identity row. They were not executed; no
  GUI, Docker, policy, scientific, protected-case, qualification, activation, or release workflow
  ran.

### 2026-08-13 - Numeric stepper arrowheads painted directly

- The modernized numeric button column and interaction colors rendered correctly, but the small
  arrow primitive was absent under the Windows stylesheet paint path.
- The proxy style now lets the complete spin box paint first, obtains the exact upper and lower
  button rectangles, and draws compact antialiased chevrons as the final layer. Normal,
  hover/pressed, limit-disabled, dark, and light palette states remain distinct.
- The focused source contract and ignored Phase 6 validator are updated but not executed. No GUI,
  Docker, policy, scientific, protected-case, qualification, activation, or release workflow ran.

### 2026-08-13 - Redundant central workspace chrome removed

- The shared central shell no longer displays the `Scientific workspace` pseudo-tab, duplicate
  CALO-RPD badge, global guided-workflow text, or global `Continue` button while ordinary panels
  are active. Their reserved vertical space is released, so panel content begins immediately
  beneath the ribbon.
- Workflow state and ordering remain unchanged. The Dashboard's existing `Next required action`
  card is the single workflow continuation surface. Document tabs appear only when a real
  secondary document, such as the operating guide, is opened and collapse again when it closes.
- Focused contracts and the ignored Phase 6 validator are updated but not executed. No GUI,
  Docker, policy, scientific, protected-case, qualification, activation, or release workflow ran.

### 2026-08-13 - Ribbon category no longer resembles a misplaced button

- The persistent 65-pixel blue fragment directly below the heading aligned with the selected
  first ribbon category (`Home`), not with the `algorithms.configure` command or another hidden
  Configure control. It was the rounded top edge of the selected category's filled background.
- Selected categories now use transparent text with a two-pixel bottom accent instead of a filled,
  rounded blue block. Hover feedback remains subtle, and command-page content and actions are
  unchanged.
- Focused theme contracts and the ignored Phase 6 validator are updated but not executed. No GUI,
  Docker, policy, scientific, protected-case, qualification, activation, or release workflow ran.

### 2026-08-13 - Scientist-facing training governance inputs removed

- The training pane no longer asks scientists to browse for a development freeze or Phase 4
  acceptance receipt. Those records are internal source-authority details, not scientific
  parameters for a completed application.
- One read-only `Training foundation` status now represents the application-owned, immutable,
  checksum-bound A–E/F-off rules and provenance. Fresh training initializes a new policy; an
  optional prior plan may still populate visible scientific inputs but supplies no authority.
- Release packaging can include `training-foundation.json` and its two authenticated authority
  records under the dedicated package location. This development tree intentionally fabricates no
  final authority; without a legitimate packaged foundation the GUI reports unavailable and fails
  closed instead of asking the user to manufacture governance files.
- The current training CLI retains its exact clean-source checkout gate. Enabling training from a
  final installed runtime without Git is intentionally left to the later release gate; this GUI
  correction does not weaken source authentication.
- Focused tests, package-membership checks, native guidance, and the ignored Phase 6 validator are
  updated but not executed. No GUI validation, Docker, policy training/evaluation, qualification,
  protected-case, activation, publication, or release workflow ran.

### 2026-08-13 - Built-in training architecture correction

- The preceding separate `Training foundation` design is superseded. Legacy CALO already contains
  the canonical TSH-CALO architecture, so it is used automatically and is no longer displayed as
  a training input or availability status.
- Fresh GUI plans bind to the current authenticated application source; approved A–E remain
  available, optional E is off, and experimental F stays disabled by default. Development-freeze
  and Phase 4 acceptance paths remain absent from the
  scientist-facing workflow.
- Loading an existing plan imports scientific settings only, clears legacy governance fields, and
  rebinds the generated plan to the current source. It cannot grant qualification or activation
  authority.
- The first ribbon click opens the training inputs without starting a process. The same command
  then becomes `Check readiness`, and only after a successful check becomes `Start training`.
  Candidate output remains unqualified and inactive.
- Focused source, tests, documentation, and the ignored validator were updated but not executed. No
  validation, Docker, policy workflow, or protected-case operation ran.

### 2026-08-13 - User-facing base architecture choice

- The training inputs now begin with a concise `Base architecture` dropdown containing `CALO` and
  `TSH-CALO`; internal proposal-letter and phase terminology is not shown in the control.
- CALO is truthfully treated as the built-in policy-free optimizer: selecting it disables policy
  settings, readiness, resume, and start rather than silently invoking the retired CALO trainer.
  Selecting TSH-CALO enables the existing independent plan/check/explicit-start workflow.
- Architecture selection participates in the readiness fingerprint. Imported independent plans
  resolve to TSH-CALO, and a model-level guard prevents CALO from being routed through the
  TSH-CALO training command even if a GUI caller bypasses disabled controls.
- Focused source contracts and the ignored Phase 6 validator were updated but not executed. No GUI,
  validation, Docker, training, policy lifecycle, protected-case, or release operation ran.

### 2026-08-13 - Policy workflow and GUI error-path consolidation

- Policy-training records selected in Resume Center now prefill only the independent training
  model and input pane. The route cannot call the experiment manager or the removed embedded
  trainer, and it never checks readiness or starts a process automatically.
- The dormant legacy training widget tree, worker, training/qualification controls, and callbacks
  were removed from CALO Intelligence. That panel now contains one `Train policy` navigation action,
  one `Import policy` action, the policy library, and governing-policy controls.
- Resume Center `Inspect` now presents a concise task summary. Its optional technical-details area
  exposes only task identity and resumability rather than rendering the stored state as raw JSON.
- Scientist-facing failures now use shared short explanations and direct users to Activity Logs;
  exception types, messages, and tracebacks are retained in the Activity logging stream instead of
  being placed in modal dialogs or ordinary panel status text.
- Ribbon categories are exclusive styled buttons with a separately owned command-page stack. No
  native category-tab paint primitive remains, so category selection cannot render into the product
  identity row.
- Focused regression source and the ignored Phase 6 validator were updated. Only static source
  inspection and `git diff --check` were performed; no validator, test, GUI, Docker, training,
  qualification, protected-case, publication, or release workflow was executed. Current-source
  validation therefore remains pending.

### 2026-08-13 - Product-language boundary

- Ordinary GUI surfaces now describe capabilities, scientific choices, compatibility, readiness,
  and whether a policy is selected for experiments. Proposal letters, numbered phases, build-stage
  labels, development freezes, candidate terminology, feature flags, runtime ABI, checksums, and
  similar engineering vocabulary remain in internal state, Activity provenance, logs, schemas,
  and implementation records rather than normal controls or status copy.
- A separate `PRODUCT_VERSION` presents `12.0.0` in the ribbon, status bar, sidebar, Settings, and
  About dialog. The exact `12.0.0-dev.1` build identity and development stage remain unchanged for
  package/version verification and technical records.
- Raw policy-readiness reasons remain exact internally, while a centralized display adapter gives
  Dashboard, workflow navigation, the global status bar, and the policy library concise user-facing
  availability explanations. Policy integrity and compatibility enforcement are unchanged.
- Policy training no longer displays raw command lines or subprocess output in the workspace;
  those details are routed to Activity Logs. The visible page presents readiness, explicit start,
  safe resume, and the fact that a result is not selected for experiments automatically.
- Benchmark controls present method verification, held-out campaign design, and campaign records;
  the underlying verified manifest and evidence-package contracts are unchanged. Current-source
  tests and the ignored Phase 6 validator were updated but not executed. No GUI, validation,
  Docker, training, qualification, protected-case, publication, or release workflow ran.

### 2026-08-13 - Product-version status-record correction

- Owner validation run `phase6-20260813-183722` passed commands `01`, `02`, `03`, `04a`, `04b`,
  and `04c`, then failed command `05-active-version`. The returned report showed every active-version
  check passing except `active_status_identity`; no later validator command ran.
- The failure was caused by the new verifier expectation `product_version=12.0.0` not yet being
  present in `ACTIVE_DEVELOPMENT_STATUS.json`. Package version `12.0.0.dev1`, technical display
  version `12.0.0-dev.1`, release line, and development stage were already correct.
- The active status now records `product_version` independently and retains the exact failed-attempt
  history. The verifier and focused status contract require all three version identities without
  converting the internal build into a release identity.
- The validator was not rerun by the development agent. No Docker, GUI, policy, training,
  qualification, protected-case, publication, or release workflow was executed; a fresh complete
  owner run remains required.

### 2026-08-13 - Phase 6 Ruff correction after owner rerun

- Owner validation `phase6-20260813-184612` confirmed the product-version correction: command
  `05-active-version` passed every check. Command `06-compile` also passed. The run then stopped at
  `07-ruff` after eight successful validator checks.
- Ruff reported exactly seven mechanical findings: three unused `exc` bindings and four unused
  `QMessageBox` imports. The unused bindings/imports were removed without changing exception
  propagation, logging, task state, policy semantics, or scientific behavior. Exception bindings
  still used by logging and user-feedback adapters were retained.
- The pasted console visually joined two adjacent test paths, but the retained command log proves
  they were separate Ruff arguments; the validator source list required no correction.
- Command `08` and every later validator command remain unexecuted in this attempt. The development
  agent did not run Ruff, formatting, tests, GUI rendering, packaging, Docker, training,
  qualification, protected-case, publication, or release workflows. Fresh full owner validation
  is pending.

### 2026-08-13 - Phase 6 formatter correction after owner rerun

- Owner validation `phase6-20260813-185633` confirmed commands `01` through `07`, including the
  corrected active-version, compilation, and Ruff diagnostics. It then failed `08-format` after
  nine successful validator checks, listing exactly 27 files that Ruff would reformat.
- Ruff formatting was applied only to those 27 retained-log targets, in four Windows-safe command
  chunks of 8, 8, 8, and 3 files. The operation reported all 27 files reformatted. After recording
  this attempt, the two changed Python history-contract files were passed through Ruff formatting
  again and were already unchanged.
- This was a deterministic formatting write, not an executed format check or validation claim.
  Commands `09` onward did not run in the owner attempt. The development agent ran no validator,
  test, GUI render, package build, Docker, policy, training, qualification, protected-case,
  publication, or release workflow. Fresh full owner validation remains pending.

### 2026-08-13 - Phase 6 interrupted GUI validation corrected fail-closed

- Owner validation `phase6-20260813-191340` passed commands `01` through `09`, including all 59
  focused unit tests. It entered `10-phase6-gui` at 19:13:55 and retained no completion record
  before the owner pressed Ctrl+C; finalization occurred at 20:08:23. Commands `11` through `16`
  never started, so the generated PASS is rejected as incomplete evidence.
- The likely deterministic offscreen blocker was the GUI fixture sharing the real application
  session-recovery journal: an unfinished real session could open a modal recovery dialog once Qt
  events were processed. Each Phase 6 GUI test now substitutes its own temporary journal before
  constructing `MainWindow`; scheduled recovery and hardware-discovery startup probes are also
  suppressed inside this focused fixture. Production recovery and hardware behavior are unchanged.
- Command `10` now prints individual test names. An autouse two-minute per-test watchdog writes the
  stuck test identity and all Python thread stacks, then exits only the dedicated test subprocess
  with code 124. This provides bounded, diagnostic failure rather than an indefinite wait.
- The ignored validator now requires the exact full `01`-through-`17` result sequence in addition
  to PASS statuses. Missing commands after Ctrl+C or another interruption therefore force an
  incomplete FAIL in JSON, Markdown, console output, and the process exit code.
- No validator, test, GUI render, build, Docker, training, policy, qualification, protected-case,
  publication, release, or scientific workflow was executed by the development agent. One fresh
  complete owner validation is required; commands `10` onward remain unverified.

### 2026-08-13 - Phase 6 GUI test teardown timeout isolated

- Owner run `phase6-20260813-202657` confirmed the new fail-closed behavior and passed commands
  `01` through `09`, including all 60 focused unit tests. Command `10-phase6-gui` printed individual
  progress: its first 18 of 21 tests passed, then the watchdog ended the subprocess with code 124
  while pytest was tearing down the eighteenth test. The validator correctly retained FAIL.
- The eighteenth test is the only focused contract that explicitly shows the complete top-level
  window. Its assertions passed; the wait began afterward when pytest-qt closed the tracked window
  through the production `closeEvent`, which performs session recovery, policy inspection, layout
  persistence, and logging shutdown and may display safety dialogs. Those operations are outside
  this appearance/product-language contract.
- The focused fixture now supplies pytest-qt a `before_close_func` that detaches the fixture-owned
  Activity log handler and replaces `closeEvent` with direct acceptance only for teardown. The
  production close path, dialogs, persistence, active-work safeguards, and scientific behavior are
  unchanged.
- No validator, test, GUI render, build, Docker, training, policy, qualification, protected-case,
  publication, release, or scientific workflow was executed by the development agent. Commands
  `10` onward still require one fresh complete owner run.

### 2026-08-13 - Phase 6 product-language GUI contract corrected

- Owner run `phase6-20260813-205516` passed commands `01` through `09`, including all 60 focused
  unit tests. Command `10-phase6-gui` completed all 21 tests in 25.77 seconds with 20 passed and one
  failed. This confirms the prior GUI hang and teardown corrections.
- The sole failure was a stale assertion requiring `ceiling` in the memory tooltip. The current
  product-facing tooltip intentionally says `available-memory safety limit`; its admission meaning
  remains unchanged. The test now asserts that approved visible language.
- Three captured portfolio-planning ERROR entries were ordinary empty-input state during panel
  initialization, not independent failures. `refresh_plan` now presents `Select at least one
  output` and returns before planning when no output is selected. Unexpected planner exceptions
  continue to be recorded in Activity Logs, and applying an invalid plan remains fail-closed.
- No validator, test, GUI render, build, Docker, training, policy, qualification, protected-case,
  publication, release, or scientific workflow was executed by the development agent. Commands
  `10` onward require one fresh complete owner run.

### 2026-08-13 - Phase 6 offscreen training-help contract aligned

- Owner run `phase6-20260813-212634` passed commands `01` through `12`: all 61 focused unit tests,
  all 21 Phase 6 GUI tests, all 21 GUI regressions, and all 9 empty-policy integration tests passed.
  Command `13-gui-render` was the first failure.
- The offscreen renderer expected 17 training information controls and omitted `architecture`.
  The live widget and focused GUI contract both correctly require and passed all 18 controls,
  including the Base architecture help button. The renderer now includes `architecture`.
- Future renderer mismatches report exact missing and unexpected keys instead of the generic
  `Training inputs do not all have information controls` message.
- No validator, test, GUI render, build, Docker, training, policy, qualification, protected-case,
  publication, release, or scientific workflow was executed by the development agent. Commands
  `13` onward require one fresh complete owner run.

### 2026-08-13 - Phase 6 training start action made explicit

- Owner run `phase6-20260813-215626` passed the complete validator sequence: 62 unit, 21 focused
  GUI, 21 GUI-regression, and 9 empty-policy tests passed; offscreen rendering, wheel/sdist build,
  distribution safety, required Phase 6 contents, and exact source stability also passed.
- Manual GUI review then identified an interaction defect not represented by that contract: the
  contextual input pane calculated `Check readiness` and `Start training`, but rendered neither.
  It silently reassigned those stages to repeated clicks on the ribbon `Train policy` command.
- The ribbon command is now navigation-only and remains labelled `Train policy`. A persistent
  action footer below the scrollable training inputs visibly shows `Check readiness`, changes to
  `Start training` only after exact-input readiness passes, and retains the existing explicit
  confirmation before process launch. It is disabled with contextual guidance when unavailable.
- Focused GUI, static, offscreen-renderer, light-theme, and dark-theme contracts were updated. No
  validator, test, GUI render, build, Docker, training, policy, qualification, protected-case,
  publication, release, or scientific workflow was executed by the development agent. A fresh
  complete owner validator run is required for the changed source.

### 2026-08-13 - Phase 6 compatible resume made visible

- Manual start review reached the existing-output guard, which instructed the user to select a
  compatible resume option even though that canonical option still lived in the hidden controller.
- The controller-owned checkbox is now reparented into the visible Campaign inputs as `Resume
  compatible training`; it remains off by default and has its own accessible information control.
  No duplicate resume state exists. A new run cannot enable `Start training` for an existing path,
  and resume cannot enable it unless the output is an existing directory.
- Browse now matches the selected intent: fresh training selects a parent location and proposes a
  non-existing campaign-named child directory; resume selects the existing interrupted directory.
  Backend exact-plan, status-schema, campaign-state, checkpoint-identity, checkpoint-checksum,
  clean-source, readiness, and confirmation gates remain unchanged and fail closed.
- Focused GUI, static, and offscreen-renderer contracts were updated. No validator, test, GUI
  render, build, Docker, training, policy, qualification, protected-case, publication, release, or
  scientific workflow was executed by the development agent. A fresh complete owner validator run
  is required.

### 2026-08-13 - Phase 6 resumable-model library and readiness feedback corrected

- CALO remains the built-in rule-based optimizer and therefore has no policy-training lifecycle.
  TSH-CALO remains the only trainable architecture, with separate readiness, explicit start,
  unqualified saved output, and no automatic experiment selection or activation.
- Fresh TSH-CALO campaigns now default to a unique child of the operating system's per-user CALO-RPD
  application-data `training-models` directory. The campaign backend creates that directory tree
  only when explicit training starts; no model data is written into the source checkout.
- `Saved training` lists only interrupted/running campaigns discovered in the default location and
  directories explicitly registered with `Add to path`. Selecting one visibly enables the single
  canonical resume checkbox and keeps the saved plan, checkpoints, output, and source identity in
  their original directory. Added locations expand scanning only; files are not copied or moved.
- Readiness still rejects a dirty nonignored source tree. Its traceback is retained at DEBUG level
  in Activity -> Logs, while the input footer and Warnings present a short explanation that the
  application source has uncommitted changes and that training was not started.
- Focused unit, GUI, static, offscreen-renderer, status, and ignored-validator contracts were
  updated. The development agent ran no validator, tests, GUI, Docker, policy training/evaluation,
  qualification, protected-case, publication, or release workflow. A fresh complete owner Phase 6
  validator run is required.

### 2026-08-13 - Phase 6 policy training made TSH-CALO-only

- The `Base architecture` CALO/TSH-CALO selector was removed from policy training because rule-based
  CALO has no applicable training lifecycle. Leaving it there incorrectly implied that both choices
  could consume the campaign, PPO, checkpoint, and resume inputs.
- The independent training model no longer accepts a mutable architecture value, and its controller
  no longer retains hidden CALO-specific branches or no-training dialogs. Every input and action in
  this pane now applies to TSH-CALO policy training.
- Rule-based CALO remains unchanged in the ordinary algorithm and experiment-selection registry. No
  CALO search behavior, TSH-CALO policy semantics, training plan, reward, compatibility, readiness,
  resume, qualification, activation, or fallback gate changed.
- Focused GUI, unit, static, offscreen-renderer, status, traceability, and ignored-validator
  contracts were updated. The development agent ran no tests, GUI, validator, Docker, training,
  evaluation, qualification, protected-case, publication, or release workflow. A fresh owner Phase
  6 validator run is required.

### 2026-08-14 - Phase 6 training-editor startup order corrected

- Manual native launch exposed an initialization-order failure before the main window appeared:
  `TrainingPathEditor._select_new_training()` refreshed the editor before its visible status and
  primary-action widgets had been constructed.
- Initial new-training selection now occurs after those widgets and their controller connections
  exist. The first refresh can therefore publish `Ready for validation` and configure the visible
  `Check readiness` action without accessing a missing attribute.
- Focused static and offscreen startup contracts now require the safe construction order and the
  initial visible status. No GUI, tests, validator, Docker, training, evaluation, qualification,
  protected-case, publication, or release workflow was executed by the development agent. A fresh
  native launch and complete owner Phase 6 validator run are required.

### 2026-08-14 - Phase 6 readiness and Safe-80 training admission made consistent

- Manual TSH-CALO execution showed three successful readiness checks followed by immediate
  first-member failure: CUDA could not admit the estimated working set, fallback CPU admission also
  exceeded 80% of currently available RAM, and no usable policy was produced.
- `--check` had validated the plan/source boundary but returned before constructing the policy
  shape or invoking the Safe-80 device guard. Readiness now calls the same network estimator and
  device-admission routine as `IndependentTSHCALOTrainer`, records the nonexecuting admission
  summary, and immediately releases any readiness-only CUDA lease. A failed admission cannot mark
  the training fingerprint ready or expose Start training.
- Fresh GUI plans had also equated raw candidate evaluations with retained policy transitions. The
  exact session accounting consumes one initial population and then one population per policy
  transition, so rollout capacity is now `evaluations / population - 1` (bounded by 4096) and stays
  synchronized when either input changes. Evaluation totals and the 80% VRAM/RAM ceiling are not
  reduced or weakened.
- CPU/VRAM admission failures now produce a short input-pane explanation while technical details
  remain in Activity -> Logs. No tests, GUI, validator, Docker, training, evaluation,
  qualification, protected-case, publication, or release workflow was executed by the development
  agent. A fresh native owner check and complete Phase 6 validator run are required.

### 2026-08-14 - Phase 6 owner run 16 stopped at Ruff

- Owner run `phase6-20260814-003900` passed eight checks through command `06`, including all three
  ignored-path checks, then first failed command `07-ruff` with two F821 reports for the same
  undefined lowercase `root` name in the new offscreen resource-contract source reads.
- The retained command log proves the adjacent unit-test paths were separate arguments; the compact
  console rendering did not identify a PowerShell array defect. No unnecessary validator-list
  change was made.
- The offscreen contract now defines and uses an explicit repository root derived from its own
  source path. No tests, GUI, validator, Docker, training, evaluation, qualification,
  protected-case, publication, or release workflow was executed by the development agent. A fresh
  complete owner Phase 6 rerun is required.

### 2026-08-14 - Phase 6 agent-run validation 17 stopped at focused GUI contracts

- With explicit owner authorization, sandboxed launch first produced retained infrastructure-only
  run `phase6-20260814-004600`, which could not execute repository Python. The exact validator was
  then run outside the sandbox as `phase6-20260814-004621` without changing persistent PowerShell
  policy.
- The executable run passed commands `01` through `09`, including Ruff, formatting, and 73 unit
  tests. Command `10` completed with 22 of 24 GUI tests passing. One failure selected a scan path
  persisted by an earlier focused-test window; Qt application settings can retain their first
  storage location across repeated windows in one process. The fixture now clears and synchronizes
  only its isolated temporary settings before each window. Production scan-location persistence is
  unchanged.
- The second failure found visible resume help containing the internal word `checksum`. It now says
  `saved-file integrity`; exact checksum enforcement remains unchanged internally.
- No Docker, training, evaluation, qualification, protected-case, publication, or release workflow
  was executed. A fresh complete validator rerun is required.

### 2026-08-14 - Phase 6 authorized validation completed

- Authorized run `phase6-20260814-004927` passed the complete command `01` through `17` sequence:
  73 unit tests, 24 focused GUI tests, 21 affected GUI regression tests, 9 empty-policy/training-
  navigation integration tests, offscreen light/dark/constrained rendering, wheel and sdist build,
  both distribution verifiers, and nonignored-source stability.
- The validated corrections isolate temporary GUI settings per test window and use product-facing
  saved-file-integrity wording without weakening internal exact-resume verification.
- The retained evidence is under `validation/logs/phase6-20260814-004927`. The validator performed
  no Docker, policy training/evaluation/qualification/registration/activation/deletion,
  protected-case, publication, CUDA-campaign, or release operation, and it does not infer human
  screen-reader, usability, or scientist acceptance.

### 2026-08-14 - Completed training discovery and explicit policy import corrected

- The saved-training selector previously scanned only `running` and `interrupted` campaign states,
  so a valid `completed` campaign disappeared from the selector. The Policy library independently
  listed only registered records, while successful training correctly avoided automatic
  registration; together those rules made a completed output appear lost.
- `TrainingModelLibrary` now discovers completed campaigns, verifies the manifest-bound ensemble
  file by SHA-256 using a cached streaming read, scans registered roots to a bounded depth, and
  synchronizes newly added locations. Adding a location or completing a run refreshes the selector
  and Policy library; the explicitly added location selects its newest saved campaign.
- Completed candidates appear in the Policy library as `Training complete · import required`.
  The existing single Import action registers the selected candidate only after the user clicks it.
  Discovery never registers, qualifies, selects, activates, binds, resumes, or retrains a completed
  candidate.
- Focused Ruff/format checks, 18 command/native contracts, and 25 offscreen GUI contracts passed.
  No policy training or scientific lifecycle operation executed. The ignored complete Phase 6
  validator is updated for a later full follow-up run.

### 2026-08-14 - Global checkbox borders restored in both themes

- Native inspection showed checkbox state marks without a visible indicator boundary, including the
  Training cases and compatible-resume controls. Both themes had constrained every indicator to
  `15px` but supplied no application-owned boundary.
- The application-wide Qt proxy style now paints every `QCheckBox` indicator with a palette-aware
  rounded border and surface, plus vector checked and partial marks. Hover, keyboard focus, pressed,
  and disabled states remain visibly distinct in light and dark themes; radio-button rendering is
  unchanged. Both theme contracts now reserve a consistent `16px` checkbox indicator.
- Focused source contracts and the ignored Phase 6 renderer/validator now require global styling and
  retained light/dark renders of unchecked, checked, partial, focused, and disabled states with a
  minimum perimeter-contrast check. No GUI, test, validator, policy, scientific, protected-case,
  publication, or release workflow was executed by the development agent. A fresh complete Phase 6
  follow-up validator run remains pending.

### 2026-08-14 - New-training recovery and exact resume separated in the GUI

- User inspection found the exact-resume checkbox disabled while `New training` was selected and
  reasonably interpreted that as meaning a fresh campaign could not be resumed. The backend already
  saves a verified recovery point after every safely committed training window; `--resume` is only
  legal for an existing compatible interrupted campaign and would make a fresh output fail.
- The Recovery row now shows a checked, non-configurable `Automatic recovery for new training`
  status for a fresh campaign. Selecting a discovered `running` or `interrupted` campaign replaces
  that status with the interactive `Resume selected interrupted training` choice; a completed
  campaign shows that resume is not applicable. Status and help text explain the same distinction.
- No checkpoint frequency, training equation, evaluation accounting, integrity guard, or lifecycle
  authority changed. Focused source/GUI/offscreen contracts and the ignored Phase 6 validator are
  updated but not executed. No training, policy, scientific, protected-case, publication, or
  release workflow was performed; complete current-source Phase 6 follow-up validation is pending.

### 2026-08-14 - Checkbox and new-training recovery follow-up validation passed

- With explicit owner authorization, the initial sandboxed infrastructure run
  `phase6-20260814-130518` could not execute repository Python. Executable attempts then retained
  the first failure at format (`130547`), a whitespace-sensitive unit source assertion (`130631`),
  and the light unchecked-border render threshold (`130806`, `131047`, and `131435`). Ruff formatted
  exactly the five reported files; the brittle assertion now checks stable validation tokens.
- The checkbox border is fully opaque in the idle enabled state. Retained screenshot and pixel
  inspection also proved the renderer had measured a transparent child grab against transparent
  black rather than the displayed application surface. The gate now samples mapped indicator
  pixels from the fully composited host render; it still requires measurable borders for unchecked,
  checked, partial, focused, and disabled states in light and dark themes.
- Authorized run `phase6-20260814-131637` passed the complete command `01` through `17` sequence:
  74 unit tests, 25 focused GUI tests, 21 GUI regression tests, 9 empty-policy/training-navigation
  integration tests, composited light/dark checkbox renders, fresh wheel and sdist build, both
  distribution verifiers, evidence hashing, and nonignored-source stability.
- This closes the current automated Phase 6 GUI/native/packaging follow-up gate for completed-output
  discovery, global checkbox borders, and truthful fresh-training automatic recovery presentation.
  The validator executed no Docker, CUDA campaign, training, policy lifecycle, protected-case,
  publication, or release workflow and does not infer human screen-reader, usability, or scientist
  acceptance.

### 2026-08-14 - Finite training progress and repeatable checkpoint-safe pause follow-up

- The independent campaign now emits schema-bound start, committed-checkpoint, resume, member, pause,
  and completion events. `training_events.jsonl` retains the detailed event history while the CLI
  streams checkpoint events for the GUI. The global task bar exposes exact committed percentage;
  Activity Jobs/Logs retain candidate-evaluation and checkpoint identity detail.
- Pause is cooperative: the GUI writes an idempotent control request bound to the campaign ID and
  immutable plan hash. The runner acknowledges it only after the current bounded window finishes,
  the checkpoint is durably saved and hashed, and `uncommitted_cuda_window` is cleared. Only that
  receipt produces the distinct `Paused`/resumable UI state; force interruption retains the prior
  fail-closed semantics.
- There is no pause/resume-count ceiling. Every resume authenticates and continues the same finite
  plan, seeds, state, and exact candidate-evaluation budget; pausing neither expands nor resets it.
  Infinite evaluation-budget training is explicitly not implemented.
- Focused campaign, CLI, GUI, status, static, and offscreen contracts plus the ignored Phase 6
  validator are updated. Per repository instructions they were not executed by Codex. No policy
  training, scientific, protected-case, qualification, publication, or release workflow ran.
  The prior `phase6-20260814-131637` pass predates this source and fresh complete owner validation is
  pending.

### 2026-08-14 - Completed-model finite extension follow-up implemented, validation pending

- A completed campaign now records one final authenticated continuation checkpoint per ensemble
  member in its immutable manifest. Each checkpoint retains model parameters, optimizer state,
  NumPy/Torch RNG states, PPO update count, prior episode receipts, device/memory provenance,
  session/environment state, rollout collector state, and exact evaluation accounting.
- The completed-model selector exposes `Check extension readiness` and then `Extend training` only
  for manifests with the complete continuation contract. Older completed artifacts without bound
  continuation hashes remain usable as unqualified saved candidates but are not guessed into exact
  extendability.
- Every explicit extension is a new finite child segment under `extensions/segment-NNNNNN`. It
  authenticates the unchanged source/scientific/execution plan and parent manifest, continues every
  member from its final trainer state, repeats the identical case/seed curriculum and finite episode
  budget, adds unique receipt/session identities, retains cumulative exact FE accounting, and writes
  new unqualified member/ensemble candidates plus continuation checkpoints. Parent files are not
  mutated. A paused child resumes through the same checkpoint-safe protocol.
- The successful extension count has no configured ceiling, but no segment starts automatically and
  each segment remains finite and explicitly confirmed. More training is not claimed to improve,
  strengthen, qualify, register, activate, or select a model; independent evidence remains required.
- Production source, synthetic unit/GUI/static/offscreen contracts, the ignored Phase 6 validator,
  active status, traceability, and handoff are updated. Nothing was executed by Codex. Fresh complete
  owner validation remains pending and no policy/scientific/protected/release workflow ran.

### 2026-08-14 - Responsive training and completed-policy-library workflow implemented, validation pending

- The training input footer no longer repeats the percentage bar and checkpoint detail already
  presented by the persistent bottom status bar and Activity Jobs/Logs. While training owns the
  runtime, the left footer retains only a short location hint and the checkpoint-safe pause action.
  Path fields, rows, the editor host, and its stack now shrink inside the fixed-width dock; the
  horizontal scrollbar remains prohibited.
- Policy-library discovery now includes every completed campaign, including an invalid/missing
  candidate attention state. A completed campaign remains visible after import by merging its
  registry record into the same row. Import is explicit; training completion still confers no
  qualification, activation, selection, binding, or experiment authority.
- A library row can activate a policy only when the existing registry proves it independently
  qualified, compatible, post-development eligible, immutable, checksum-valid, and backed by a
  passed qualification receipt. Ordinary completed candidates visibly require import and then
  qualification. No qualification workflow was added or executed.
- `Delete model files` physically removes one exact unregistered completed campaign child directory
  only after an irreversible-path confirmation. The guard rejects scan roots, symbolic links,
  incomplete/non-discoverable campaigns, and any directory containing a registered or active
  policy. Registered records retain the evidence-backed retirement-review path.
- Applying an already ready governing policy preserves the immutable experiment binding, unlocks
  the next Power System setup step, and navigates there without loading a case, running power flow,
  starting evaluation, or performing any other scientific work. Source, synthetic unit/GUI/static/
  offscreen contracts, the ignored validator, active status, and ledgers are updated but unexecuted.
  Fresh complete owner-run Phase 6 validation remains pending.

### 2026-08-14 - Full-width dynamic Policy library layout implemented, validation pending

- The CALO Intelligence Policy library no longer receives vertical stretch or owns nested scrolling.
  Its horizontal and vertical scrollbars are disabled, Policy and Scientific status columns share
  the available width, and the table recalculates a fixed height equal to its header plus every
  current visible entry after refresh and panel resize. An empty library occupies only its header.
- The Policy library and Governing policy groups use expanding horizontal size policies. The
  governing form now grows every non-fixed field, so the selected policy, wrapped status, and apply
  action consume the available page width. The existing main preview remains the sole vertical
  scroll owner when the complete page is taller than the window.
- No policy discovery, lifecycle, activation, deletion, experiment binding, scientific, training,
  or evaluation semantics changed. GUI/static/offscreen contracts and the ignored Phase 6
  validator are updated but not executed; current-source owner validation remains pending.

### 2026-08-14 - Path visibility, explicit removal actions, and reachable policy form implemented

- The saved-training default location now uses a width-aware selectable path label. It calculates
  and reserves every wrapped line, including unbroken filesystem segments, so the Settings template
  row cannot cover or clip the path in the permanent narrow input pane. The complete path is also
  retained as its tooltip.
- Policy Library now presents two stable, unambiguous controls: `Review policy removal` for the
  evidence-backed registered-policy inventory/dry-run, and `Delete model files` for an exact
  unregistered completed campaign. Registered policy files remain protected from direct deletion;
  destructive retirement still requires the separately authorized workflow.
- Externally scrolled workspace pages now propagate their changing preferred content height through
  the workspace stack. When entry-sized policy rows push Governing policy below the viewport, the
  main preview obtains the required scroll range and can reveal the complete status and apply action
  above the Activity dock. Child table scrollbars remain disabled.
- Source, GUI/static/offscreen contracts, the ignored Phase 6 validator, active status, traceability,
  and handoff are updated but unexecuted. No model file, registry record, training, policy lifecycle,
  experiment, protected-case, publication, or release workflow was executed.

### 2026-08-14 - Usable imported-model deletion and automatic Governing policy reveal implemented

- The prior visible `Delete model files` control was disabled for an imported completed campaign;
  the latest screenshot correctly showed that it was not a usable removal action. It is now enabled
  for exactly one completed campaign whose selected registration is inactive, unqualified, has no
  qualification evidence, experiment binding, or lineage checkpoint, still matches its registered
  SHA-256, and is the only registered policy contained by that verified campaign directory.
- After explicit irreversible confirmation, the exact registration is atomically suppressed and
  removed, then the existing path-confined campaign deletion removes its checkpoints, extensions,
  logs, and model files. Active, qualified, referenced, checksum-changed, multi-registered,
  standalone, incomplete, scan-root, or symlink targets remain fail-closed. The separate reviewed
  retirement action remains available for governed cases that cannot use this narrow direct path.
- CALO Intelligence now reserves additional bottom clearance and queues `ensureWidgetVisible` for
  the complete Governing policy group after page display or policy selection. Dynamic outer-height
  synchronization remains in place, but the viewport now actively scrolls the entire group above
  Activity rather than requiring the user to discover and position the outer scrollbar manually.
- Database, registry, filesystem preflight, GUI, static, offscreen, ignored validator, status, and
  ledger contracts are updated but unexecuted. No actual model or registration was removed and no
  policy/scientific workflow ran during implementation; fresh owner validation remains pending.

### 2026-08-14 - Selection scroll, standalone deletion, and extension compatibility corrected

- Removed the automatic Governing policy reveal calls from page display and policy selection.
  Policy Library row selection now preserves the main-preview scroll position; dynamic content
  height and bottom clearance still make the full Governing policy group manually reachable.
- Removed the redundant `Review policy removal` GUI action. The existing non-GUI retirement
  inventory/dry-run tooling remains unchanged for separately authorized governed-policy work.
- `Delete model files` now evaluates the same fail-closed inactive/unqualified/unreferenced/checksum
  guard for ordinary standalone registry rows, so the first listed eligible model is deletable after
  exact-path irreversible confirmation. Completed-campaign deletion retains its directory and
  sibling-registration guards. No actual file or registration was deleted during implementation.
- Extension source admission now treats the origin software commit as provenance rather than the
  architecture identity. A clean later source revision may extend the saved exact plan only when
  authenticated trainer checkpoints still match the frozen algorithm/state/action/training-
  environment identities and the complete plan/trainer/environment parameter field schemas. Exact
  parameter values and execution-plan hashes remain unchanged. Added/removed parameters or changed
  architecture fail closed. Extension manifests record both origin and execution source commits.
- Completion-audit hardening authenticates the policy parameter layout by state-dict name, tensor
  shape, and dtype, including directly from a pre-contract continuation checkpoint. Compatibility
  comparison ignores unrelated future writer metadata; an authority field is binding when recorded,
  while older missing fields are backfilled from the authenticated plan/checkpoint instead of
  rejecting the model solely for its writer version. A completed campaign carrying legacy freeze/acceptance hashes
  can use those already authenticated embedded identities during `--extend`; the GUI no longer
  fails solely because historical authority files were not reselected.
- Production source, synthetic unit/GUI/static/offscreen contracts, active status, traceability,
  handoff, and the ignored Phase 6 validator are updated. Per repository instructions Codex did not
  run tests, validation, GUI/browser smoke, training, deletion, policy lifecycle, protected-case,
  qualification, publication, or release workflows. Fresh complete owner validation is pending.

### 2026-08-14 - Extension parameter authority narrowed to training semantics, validation pending

- A completion audit found that the first compatibility implementation described every persisted
  top-level plan path as a training parameter. That could have rejected a future plan carrying only
  new writer metadata even though its CALO architecture and training conditions were unchanged.
- The compatibility contract now emits a dedicated training-parameter-schema digest. Campaign,
  source, freeze/acceptance identity, and the reserved `writer_metadata` namespace are retained as
  provenance but excluded from parameter-schema authority. The campaign schema remains binding to
  training semantics. Every other
  plan field remains training authority by default, and trainer/session/environment field sets,
  exact values, checkpoint hashes, and policy tensor names/shapes/dtypes still fail closed.
- GUI extension readiness, the extension-chain resolver, and `train_tsh_calo --extend` now share the
  same metadata-tolerant authenticated plan parser. Fresh training remains strict and rejects the
  extension-only writer namespace. Synthetic contracts cover future writer metadata acceptance and
  retained added/removed training-field rejection.
- Source, tests, active status, gates, traceability, handoff, and the ignored Phase 6 validator were
  updated. Nothing was executed; fresh complete owner validation remains pending.

### 2026-08-14 - Latest retained Phase 6 PASS audited as superseded

- Read-only inspection found the complete retained owner bundle
  `validation/logs/phase6-20260814-132200`. All expected commands `01` through `17` passed with 74
  unit, 25 focused GUI, 21 GUI-regression, and 9 integration tests, build/distribution checks, and
  stable source status. It executed no policy/scientific/deletion/protected/release workflow.
- The bundle is not current evidence for this corrective goal. Its source-status inventory omits
  the later registry, database, and training-extension changes; its manifest records an older CALO
  Intelligence hash and an older training-campaign test hash; and its command descriptions predate
  first-row deletion, selection-scroll preservation, and architecture/parameter-based extension.
- Active status, verifier, validation instructions, gates, and handoff now identify `132200` as the
  latest complete but superseded PASS. A new complete owner-run bundle is still required.

### 2026-08-14 - Explicit Policy Library qualification and comparison workflow implemented

- Policy Library now presents the scientist-controlled sequence in place: `Import policy`, `Check
  formal plan`, `Run / resume qualification`, `Admit passed evidence`, `Compare qualified policies`,
  then `Activate for experiments`. Run completion never admits evidence; admission never activates;
  activation never binds an experiment; governing-policy Apply remains the separate Power System
  handoff.
- Formal-plan checking binds the selected imported policy SHA and the existing clean exact source
  contract. Run/resume launches the independent finite qualification command in a background
  process, retains technical output in Activity Logs, and states the exact cases, paired runs,
  optimizer cells, and per-cell evaluation budget before confirmation. It grants no registry or
  activation authority.
- Evidence admission is a new fail-closed transaction. It revalidates the formal plan and canonical
  qualification gates, exact policy/checkpoint SHA, source/design/seed/evidence/receipt/calibration
  hashes, complete zero-failure paired cells, protected-case closure, A-E evidence identities, and
  immutable receipt before setting `qualified`. Conflicting qualification IDs, active/archived
  records, changed checkpoints, screening/failing/tampered evidence, or partial results are rejected.
- Comparison uses a protocol identity that retains cases, paired seeds, exact FE/population budget,
  calibration and analysis schemas, thresholds, anytime fractions, and A-E component set. It omits
  candidate identity, run label, local device, and source-commit churn. A policy is labelled
  `Strongest comparable evidence` only if it Pareto-dominates every policy under the same design on
  conservative feasibility, objective improvement, win rate, effect size, anytime safety, and Holm
  confidence; otherwise the UI says scientist review is required. Training duration and software
  version are explicitly not quality evidence.
- Production source, synthetic temporary-registry tests, static GUI contracts, active status,
  traceability, handoff, and the ignored Phase 6 validator are updated. Codex did not execute the
  validator, tests, GUI, training, evaluation, qualification, activation, binding, deletion,
  protected-case, or release workflow. Fresh complete owner validation is pending.

### 2026-08-14 - Cumulative exact model training evaluations shown in Policy Library

- Policy Library now has a `Training evaluations` column. It reports completed, committed candidate
  evaluations used to produce that exact model; it is not a count of qualification passes or
  experiment runs.
- The authority is the model-SHA-bound TSH-CALO artifact and its authenticated per-member training
  episode receipts. An extended child retains the base receipts and appends each completed finite
  extension segment, so its cumulative count increases exactly. The base/parent artifact and its
  displayed count are never rewritten.
- New registrations retain the authenticated total as registry metadata. Older registered native
  candidates are calculated read-only from their exact checkpoint and receipts. Legacy, non-native,
  missing, or unverifiable accounting displays `Not available`; it never fabricates zero.
- Synthetic unit, GUI, and static contracts plus the ignored Phase 6 validator were updated. No
  test, validator, training, extension, qualification, experiment, activation, deletion, protected-
  case, publication, or release workflow was executed. Fresh complete owner validation is pending.

### 2026-08-14 - Policy compatibility and qualification decoupled from development phases

- Project governance and active lifecycle source now treat development phase and software revision as
  provenance, not policy-quality or qualification-admission criteria. An earlier-version policy is
  neither accepted nor rejected because of age.
- Formal qualification admission now requires the current frozen TSH-CALO ABI, an immutable exact
  checksum, a validated epistemic ensemble, authenticated independent-member training receipts, and
  protected-case isolation. The registry re-inspects the exact artifact before evidence inspection,
  activation, and experiment binding.
- Exact resume and completed-training extension remain stricter and continue to require the retained
  training compatibility contract: parameter names/shapes/dtypes, training-parameter schema,
  persisted optimizer/trainer/session/environment state, and exact evaluation accounting. Source
  revision alone is deliberately excluded from that compatibility identity.
- `Check formal plan` remains clickable for a selected imported policy. Immutable-file, ABI, or
  ensemble blockers are presented; missing historical development receipts no longer block formal
  quality evaluation. Run/admit still require a checked exact candidate-bound formal plan.
- Qualification still cannot activate or bind a policy automatically. Experiment use still requires
  passed evidence admission, explicit activation, immutable qualification receipt/calibration
  binding, and a final checksum-valid artifact inspection. Change F remains disabled.
- The prior unvalidated GUI addition of accepted-freeze and Phase 4 receipt path inputs was removed.
  No test, validator, training, extension, qualification, experiment, activation, deletion,
  protected-case, publication, or release workflow was executed. Fresh owner validation is pending.

### 2026-08-14 - Policy qualification consolidated into one frozen automatic transaction

- Policy Library now exposes one `Qualify policy` action instead of separate plan-browser,
  run/resume, and evidence-admission buttons. The click first re-inspects the exact immutable
  candidate and current TSH-CALO ABI/ensemble/training receipts/protected-case isolation. It then
  inventories every current tracked or non-ignored untracked source file, verifies the inventory is
  stable while copying, and creates a deterministic clean internal Git snapshot under per-user
  qualification storage. The development worktree is not staged, committed, reset, or modified.
  The internal snapshot commit plus retained file/SHA-256 manifest is the exact campaign source
  identity. A failed preflight changes no plan, registry, model, activation, or experiment state and
  reports the exact blocker.
- The action deterministically builds and durably freezes a candidate/source-bound A-E plan using
  case30/case57, 30 paired runs per case, population 20, and 10,000 exact evaluations per optimizer
  cell. It starts or exactly resumes 480 retained component cells. There is no resume-count ceiling;
  the plan, seeds, completed cells, and finite FE budget cannot change. Any rejected A-E component
  terminates the workflow as an immutable policy rejection.
- Only accepted exact A-E evidence can deterministically produce and freeze the formal plan. The
  formal stage uses independent fixed seeds and the same cases/runs/population/FE design for 120
  baseline/candidate optimizer cells. It starts or exactly resumes retained cells, recomputes all
  gates, and automatically admits only a complete checksum-valid pass. A completed failed formal
  result remains rejected instead of being rerun for a favorable outcome.
- Automatic admission is limited to this explicit user-invoked qualification transaction. It never
  activates, binds, initializes, or modifies an experiment. A verified admitted pass refreshes the
  Policy Library and enables the separate `Activate for experiments` button. Comparison and
  governing-policy Apply remain separate scientist actions.
- Production source, deterministic unit/GUI/static contracts, gates, traceability, handoff, and the
  Git-ignored Phase 6 validator were updated. Codex did not execute tests, validators, GUI/browser
  checks, A-E work, qualification, admission, activation, binding, experiments, protected cases,
  Docker, publication, or release work. Fresh complete owner validation is pending.

### 2026-08-14 - Qualification uses architecture/parameter authority and durable progress

- The current `Qualify policy` transaction no longer runs or requires the historical component-
  development campaign. It freezes one stage-neutral candidate contract covering the exact model
  SHA, runtime/policy state-action-training schemas, ensemble membership, feature contract,
  authenticated member training-design identities, and training-provenance digest. Product version
  and development-stage labels are absent from admission authority. Retained legacy qualification
  evidence remains readable but does not govern new plans.
- The fixed quality plan remains finite: case30/case57, 30 paired runs per case, population 20, and
  10,000 exact evaluations in each of 120 baseline/candidate cells. Completed cell files drive the
  persistent bottom progress bar and percentage. They remain available for unlimited exact resume
  attempts without changing the finite budget, seeds, or plan.
- Every foreground run now locks the ribbon, input dock, and document workspace. Activity and the
  global status bar remain enabled so Jobs/Logs/Warnings/Device/Provenance and durable progress can
  be inspected throughout execution. Existing status-bar safe pause/cancel authority is preserved.
- Source, synthetic unit/GUI/static contracts, current ledgers, and the ignored validator were
  updated. Per repository instructions no test, validator, GUI, training, qualification, protected-
  case, Docker, publication, or release workflow was executed. Fresh owner validation is pending.

### 2026-08-14 — qualification micro-observability and safe partial-cell continuation

- The qualification campaign now emits schema-bound structured events at each cell boundary and
  every 500 formal-cell evaluations. Events are fsync-retained in `qualification_events.jsonl` and
  streamed to Activity with case/run/side, live FE count, feasible objective, violation,
  first-feasible FE, throughput, and cell ETA.
- The global bar combines live evaluation progress with the authoritative retained-cell count and
  explicitly labels current-cell work as uncommitted. Activity and the bottom bar stay enabled while
  other task-changing surfaces remain locked.
- **Pause safely** writes a frozen-plan-bound request. The optimizer latches it only after a complete
  population transition, saves its population, archives/memory, histories, FE counter, and RNG state
  through the existing authenticated exact-run envelope, and exits with code 75 only after the
  campaign records a matching acknowledged receipt. Resume restores that partial cell; completed
  records remain immutable, and the finite plan/budget cannot change across unlimited pause cycles.
- Production source, synthetic unit/GUI/static contracts, docs, and the ignored Phase 6 validator
  were updated. No test, validator, GUI, training, qualification, protected-case, Docker,
  publication, or release workflow was executed; fresh owner validation is pending.

### 2026-08-15 - New training plans generated from visible inputs, validation pending

- The ordinary **New training** path now treats the campaign, eligible cases, ensemble/member seed,
  population, finite evaluation budget, compute/fallback, and PPO/model controls as the complete
  scientist-facing settings surface. **Check readiness** generates the full internal plan from
  those values plus code-owned safe resource/schema/provenance/resume defaults. No external JSON
  template is required and plan construction still starts no training or policy lifecycle action.
- `Add to path`, the displayed managed default location, `Settings template`, and `Import settings`
  were removed from the training pane. The managed per-user library and output directory remain
  internal/application-owned; `Refresh` and the explicit training-directory chooser remain.
- Selecting a retained campaign still loads its exact plan internally for authenticated resume or
  compatible finite extension. A library refresh that loses the selected campaign now resets both
  the selector and backing model to **New training**, clears the stale plan identity/error, creates a
  new campaign/output identity, and permits input-generated plan construction.
- Synthetic GUI/static/offscreen contracts and the Git-ignored Phase 6 validator were updated for
  automatic plan generation, absence of the four obsolete controls, and the disappearing-campaign
  regression. Per repository instructions Codex ran no tests, validator, GUI, training, extension,
  qualification, protected-case, deletion, Docker, publication, or release workflow. Fresh complete
  owner Phase 6 validation remains pending.

### 2026-08-15 - Fresh-plan error classification and native launch provenance corrected

- User-returned GUI evidence showed `Saved training plan could not be loaded` after requesting
  readiness for the simplified flow. Inspection found that the status branch labeled every
  `plan_error` as a retained saved-plan failure, including errors raised while generating a new plan.
  New-plan generation failures now have their own truthful status, display the exact corrective
  reason, and retain it in the status/action tooltips; saved campaign failures retain their separate
  load wording. Invalid scientific inputs are not silently rounded or replaced.
- Fresh plan source identity previously depended on the process working directory. A native command
  or shortcut started outside the checkout could therefore resolve `unavailable` even though the
  imported application source was a Git checkout. Resolution now anchors to the imported package
  root and still falls back to an immutable build declaration outside a checkout.
- Synthetic unit, GUI, and offscreen contracts plus the Git-ignored validator instructions were
  updated. Per repository instructions Codex ran no tests, validator, GUI, training, qualification,
  protected-case, Docker, publication, or release workflow. Fresh complete owner Phase 6 validation
  remains pending.

### 2026-08-15 - Transactional qualification evidence correction, validation pending

- The source-bound formal campaign under `architecture-v2-e266bd7598befa54-b47484ec0b9c`
  stopped after its first case30 baseline cell because completion telemetry supplied `total_cells`
  twice. The success record had already committed; the broad exception boundary then retained a
  failure artifact for the same logical cell. That directory remains byte-for-byte untouched,
  carries no completion authority or receipt, and is now classified read-only as an
  infrastructure-aborted evidence-integrity conflict. It is not resumable or admissible.
- New campaigns commit exactly one canonical cell identity to either `committed_success` or
  `committed_scientific_failure`, maintain a checksum-bound unique-cell index, and keep event/status
  faults in a separate infrastructure-incident channel. Telemetry can no longer transform a
  committed scientific result into a failure. Any conflicting terminal identity or infrastructure
  incident fails closed and requires a fresh source-bound campaign.
- A final `qualification_completion.json` is written last and binds the plan, seeds, unique-cell
  index, event log, status, evidence, and optional receipt. Current-schema admission requires that
  authority plus 120 unique successful cells and zero infrastructure incidents; legacy completed
  evidence remains readable through its explicit older schema.
- The next `Qualify policy` action skips the damaged campaign, prepares a new corrected-source
  snapshot/run identity, and proves that every operative frozen design field matches the retained
  plan. Candidate checksum, case30/case57, 30 paired runs, seeds, population 20, 10,000 exact FE per
  cell, analysis, OOD, thresholds, protected-case closure, and lifecycle boundaries are unchanged.
  Source, tests, ledgers, and an ignored validator are implemented but unexecuted; do not launch the
  new qualification until the owner validator passes and its complete log directory is reviewed.

### 2026-08-15 - Transactional qualification import correction, validation still pending

- The owner attempted to launch the application before running the transactional-evidence validator.
  Python stopped during module import with an `IndentationError` in `_initialize_cell_evidence`:
  `_scan_terminal_cells()` was not indented beneath its `try` statement. Because import failed before
  application construction, no campaign discovery, resume, scientific execution, or evidence write
  occurred.
- The single indentation defect is corrected. Textual inspection confirms the surrounding
  resume-integrity exception boundary now encloses the terminal-cell scan as designed. No test,
  validator, GUI, training, qualification, protected-case, activation, Docker, publication, or
  release workflow was executed by Codex.
- All prior validation evidence is stale with respect to this correction. The owner must run
  `validation/Validate-Qualification-Evidence-Transactions.ps1`, return its complete newly timestamped
  log directory, and wait for review before launching the corrected-source qualification.
