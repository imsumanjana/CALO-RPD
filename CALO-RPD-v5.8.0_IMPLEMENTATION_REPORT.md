# CALO-RPD Studio v5.8.0 — Implementation Report

**Release:** 5.8.0 — *Transactional Competitive Training and Scientific Closure*  
**Date:** 22 July 2026

## Scope

v5.8 primarily implements the unresolved NEW v5.6 competitive-training findings and the new/reopened v5.7 findings identified by the v5.7 deep re-audit. It does not introduce a new optimizer family or silently change the frozen CALO scientific comparison protocol.

## Implemented corrections

### Transactional competitive exact resume

- Exact-resume branch workers operate only on session-staged checkpoints.
- Successful sessions publish immutable branch generations under generation-specific directories.
- The root `.branches.json` file is the authoritative final commit point.
- Previous coherent generations remain immutable and authoritative until a complete new generation has been validated and committed.
- Convenience `Bxx.resume.pt` aliases are explicitly non-authoritative and refresh only after the authoritative manifest commit.

### Safe Stop, crash recovery, and bounded infinite operation

- Session-start exact state is materialized before expensive champion validation.
- Rolling common safe checkpoints use a fixed 10-epoch cadence.
- Safe Stop has explicit typed outcomes: `SAFE_STOPPED` and `SAFE_STOPPED_DEGRADED`.
- A grace deadline prevents indefinite waiting on stuck branch processes.
- Interrupted sessions retain a durable recovery index and staged states for explicit Recover/Discard.
- Resume history, coordinator messages, and champion-decision history are bounded.
- Full epoch telemetry is append-only JSONL outside resume-critical payloads.

### Champion/Base scientific selection

- Feasibility/validity eligibility is mandatory before promotion.
- Hardware inference latency is removed from scientific quality ranking.
- Validation evidence is fingerprinted by comparator schema and validation bundle.
- Previous Base and final branch candidates are re-evaluated under one common final bundle.
- Global Base selection is deterministic and order-independent.
- No eligible candidate means no false Base promotion; only an explicitly provisional terminal artifact may be retained.

### Resource orchestration

- Competitive branches receive explicit device assignments.
- Per-accelerator concurrency is capped; excess branches fall back to CPU.
- Heterogeneous branches cannot silently instantiate sibling accelerator lanes outside their admitted assignment.
- Resource assignments are recorded in session provenance.

### Formal qualification

- Superiority PASS/promotion requires complete paired evidence, predeclared favorable direction, minimum win/effect gates, minimum paired count, and Holm-adjusted significance.
- Non-inferiority is a distinct mode with an explicit relative margin.
- Paired comparisons use case-relative differences to avoid pooling incompatible raw objective scales.

### Publication evidence

- Publication-grade portfolio export requires the full expected selected-horizon evidence set.
- When independent validation is required, every expected paired row must be verified.
- Partial verified subsets are blocked from publication-ready export rather than silently treated as complete evidence.

### Scenario exact-resume fingerprint

Canonical scientific identity now handles:
- plain functions and code hashes,
- defaults and keyword defaults,
- closure cells,
- `functools.partial` underlying function/args/keywords,
- bound methods,
- callable objects and stable instance state,
- sets/frozensets, paths and byte digests.

### Trusted exact-resume migration

- Added an explicit one-time migration utility for legacy locally trusted resumes.
- Legacy SHA/integrity is verified before pickle-capable loading.
- Migration requires an explicit user trust assertion.
- Original checkpoint remains unchanged.
- Migrated checkpoint is re-saved with current machine-local HMAC trust and source SHA/provenance.

### Sparse Newton solver

- Sparse SciPy Ybus is retained as sparse.
- Polar Jacobian blocks use sparse complex-voltage derivatives.
- Full dense N×N trigonometric/Jacobian construction is avoided on the primary SciPy path.
- A deterministic dense fallback remains only for minimal environments without SciPy sparse support.

## Test additions

v5.8 adds/updates regressions for:
- hardware-neutral champion comparison,
- deterministic order-independent final ranking,
- validation-bundle fingerprint changes,
- curriculum independence from hidden duration epochs,
- `functools.partial` scientific fingerprint differentiation,
- sparse Newton Jacobian construction,
- fail-closed formal superiority qualification,
- distinct Safe Stop session status,
- immutable prior branch generations across exact resume,
- immediate Safe Stop initial exact-state persistence.

## Residual work

The following remain explicit limitations rather than being relabeled as solved:
- complete device-resident CALO cognitive/control execution while preserving seeded parity,
- target-hardware full GUI/PYPOWER/CUDA/XPU validation in this build environment,
- structural decomposition of several large legacy orchestration modules,
- broader exception-policy cleanup outside the scientific-state critical paths,
- long-duration physical accelerator soak/fault-injection evidence.

No CALO or neural-policy superiority claim is made by this software release itself.
