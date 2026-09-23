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
