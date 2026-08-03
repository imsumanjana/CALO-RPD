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
