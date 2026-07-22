# CALO-RPD v5.8 API and compatibility boundaries

The application release is **5.8.0**. Policy runtime/state/action schema strings retain their existing `calo-v4.1` ABI generation where semantics have not changed. Application release and policy ABI are distinct namespaces.

## Exact resume

v5.8 trusted-local exact resume uses authenticated machine-local HMAC sidecars. Legacy pre-HMAC resumes are not automatically trusted. The explicit `migrate_legacy_resume` utility verifies the legacy integrity sidecar and requires a deliberate trust assertion before converting a local checkpoint.

Competitive exact-resume persistence uses immutable branch generations and an authoritative root branch manifest. Convenience `Bxx.resume.pt` aliases are non-authoritative.

## Training result status

Competitive training returns a typed result with `COMPLETED`, `SAFE_STOPPED`, `SAFE_STOPPED_DEGRADED`, or failure semantics. The result remains two-value iterable for legacy `path, history = ...` callers.

## Policy evidence

Champion metrics carry comparator/validation-bundle identity. Stored metrics from a different evidence bundle are not scientifically comparable without re-evaluation. Formal qualification remains separate from in-training champion selection.
