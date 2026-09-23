# Sequential audit repairs â€” 2026-09-24

These repairs do not qualify or activate policies or authorize a release.

## 01 â€” Verified checkpoint snapshots

Portable loads, legacy authenticated resumes, and explicit legacy migrations now hash and load the same private snapshot. V2 envelopes also release the replaceable pathname before decoding. Regression tests use temporary tensor-only fixtures and never production trust keys.

Validation: `python -m pytest -q tests/unit/test_audit_checkpoint_snapshot.py` â€” 6 passed on Windows/Python 3.11.
