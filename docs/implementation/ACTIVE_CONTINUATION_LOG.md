# Active continuation checkpoint

**Updated:** 2026-08-21  
**Checkout:** `C:\Users\User\Downloads\calo-rpd-studio-v1.0.0\calo-rpd-studio`  
**Branch:** `agent/ai-repository-intelligence`  
**Pre-cleanup Git checkpoint:** `ba597eb`  
**Product:** `12.0.0.dev1` / `12.0.0-dev.1`  
**Stage:** development only

This file contains only the current continuation boundary. The superseded append-only history is recoverable from Git checkpoint `ba597eb` and earlier commits.

## Current source state

- Current source implements the Phase 6 scientist-facing shell, explicit policy lifecycle, guarded TSH-CALO training/qualification surfaces, Individual-versus-Workspace execution ownership, Portfolio-to-Study flow, algorithm staging, safe pause/resume, and exact evaluation-accounting contracts.
- Later production-source changes exist after the newest retained full Phase 6 validation source `4560b2fba6ecc5c3271da7dfd680a0985ca501f3`.
- Therefore current-source Phase 6 engineering validation is **pending**. Do not describe the current tree as validated from the retained older bundle.
- Repository cleanup changes documentation, status routing, active-version verification, and removes obsolete historical integrity tests. These changes also require the next current-source validation run.

## Last retained engineering checkpoint

- Bundle: `validation/logs/phase6-20260817-235629`
- Result at its exact source: PASS, 17/17 commands.
- Source commit: `4560b2fba6ecc5c3271da7dfd680a0985ca501f3`.
- Scope: Python/dependency identity, diff/ignore contracts, active version, compile, Ruff, format, unit/GUI/integration checks, offscreen rendering, package build, distribution verification, and source stability.
- Excluded: policy training/evaluation/qualification/activation/deletion, protected cases, Docker/CUDA campaigns, publication, release, and human acceptance.

## Protected state retained during cleanup

- Current SQLite database and schema-migration backups.
- Runtime-owned SQLite `-shm`/`-wal` sidecars; ignored but not deleted.
- Checked-in trained-model records and local TSH-CALO training/qualification artifacts, because policy deletion requires exact reference and lifecycle review.
- The last runtime-required historical freeze `calo_v690_freeze.json` and its referenced historical training snapshot.
- Repository intelligence under `.ai/` and all protected `AGENTS.md` policy blocks.

## Next required action

After recreating `.venv`, the project owner may explicitly run:

```powershell
& .\validation\Validate-Phase6.ps1 -PythonExecutable ".\.venv\Scripts\python.exe"
```

Return the complete new `validation/logs/phase6-*` directory for read-only review. The run must not be interpreted as scientific qualification, policy authorization, protected-case evidence, release approval, or human acceptance.
