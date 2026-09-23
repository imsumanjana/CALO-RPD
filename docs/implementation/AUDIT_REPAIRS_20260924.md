# Sequential audit repairs â€” 2026-09-24

These repairs do not qualify or activate policies or authorize a release.

## 01 â€” Verified checkpoint snapshots

Portable loads, legacy authenticated resumes, and explicit legacy migrations now hash and load the same private snapshot. V2 envelopes also release the replaceable pathname before decoding. Regression tests use temporary tensor-only fixtures and never production trust keys.

Validation: `python -m pytest -q tests/unit/test_audit_checkpoint_snapshot.py` â€” 6 passed on Windows/Python 3.11.

## 02 â€” make trust-key initialization single-winner

Publish a fully fsynced private candidate using an atomic no-clobber hard link, then return the persisted winner. Existing keys are never rotated. Unsupported filesystems fail closed.

Validation: 11 cumulative checkpoint/key tests passed; five new key-publication tests.

## 03 â€” release the global device mutex before waiting

Hold the global mutex only for nonblocking acquisition and reference bookkeeping; wait and cancellation run outside it. Failed constructors own no reference and close is synchronized/idempotent.

Validation: 15 cumulative audit regressions passed; four new lease tests, no GPU workload.

## 04 â€” preserve primary errors when provenance collection fails

Catch secondary provenance errors, attach a diagnostic note, preserve the original exception object, and deny resumability when provenance is unavailable. Existing successful-provenance accounting conditions are retained.

Validation: 19 cumulative audit tests passed; four new primary-error regressions.

## 05 â€” check platform-specific locking on Linux and Windows

Use sys.platform guards recognized by mypy. CI runs the unchanged 23-file typed safety boundary for both linux and win32; no type-ignore or gate suppression was added.

Validation: 21 cumulative audit tests passed; both 23-file mypy targets passed.

## 06 â€” unify CUDA leases by runtime physical UUID

All production CUDA acquisitions use a runtime-UUID factory and one physical namespace. Missing UUIDs and mismatching UUID claims fail closed. Competing processes and containers must share the lease directory.

Validation: 53 selected tests passed, including 34 new audit regressions; both 23-file mypy targets passed.
