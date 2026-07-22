# CALO-RPD Studio v5.4.1 — Release Validation

**Release date:** 2026-07-22  
**Release purpose:** policy-gated runtime safety + release-identity/provenance correction.

This correction release resolves two audited release blockers together:

1. the active source/release identity and release evidence were stale at 5.0.0 and the old current manifests no longer matched the supplied v5.4 tree;
2. policy-assisted CALO could encounter a missing default policy path and construct an untrained neural fallback instead of failing closed.

## Policy-gated runtime correction

- CALO-RPD intentionally ships with **no automatically active/default neural policy artifact**.
- `AIController` no longer creates a random/untrained fallback network when a checkpoint is absent.
- Policy-assisted CALO requires an explicit checkpoint and strict immutable policy binding; missing-policy and non-bound execution are blocked.
- Missing checkpoint paths raise a hard error instead of falling back.
- Rule/No-AI control remains available only as an explicit research/qualification mode; it does not instantiate a fake neural model and is never an automatic fallback.
- CALO Intelligence is policy-gated:
  - with no registered usable policy artifact, only policy provisioning/training/import remains enabled;
  - with policies present but no compatible active policy, the Policy Center can be used for qualification/activation while runtime intelligence controls remain locked;
  - runtime controls unlock only when a compatible policy is explicitly active.
- Training and import do not auto-activate a policy.
- Runtime activation rejects incompatible legacy policy schemas.
- Training is blocked from overwriting an already registered policy artifact.
- The normal CALO default configuration keeps policy assistance enabled but now also defaults to strict policy binding, forcing explicit provisioning before policy-assisted evaluation.

## Release identity and provenance correction

The active release identity is now **5.4.1** and is consistent across:

- `pyproject.toml`;
- `calo_rpd_studio/version.py`;
- bootstrap fallback version;
- application About/Settings/sidebar displays;
- README and citation metadata;
- benchmark campaign defaults;
- current freeze manifest selection;
- CI workflow label;
- root release metadata and validation evidence.

A new current software freeze was generated:

`calo_rpd_studio/data/frozen/calo_v541_freeze.json`

The freeze deliberately does **not** contain a neural policy checkpoint. The software freeze certifies the CALO/software implementation scope; each policy-assisted experiment separately binds the exact user-selected policy SHA-256. This prevents the software release from implying that any bundled/default policy is scientifically preferred.

Current freeze verification:

- status: **PASS**
- frozen files checked: **98**
- default/missing neural policy in freeze: **none by design**
- untrained fallback permitted: **no**

Current full-repository release manifest:

- `MANIFEST.sha256`: **PASS**
- release-controlled files hashed: **362**
- caches/bytecode/build directories: excluded from the release manifest by design

Historical freeze manifests remain in the repository for provenance but are not the current release gate.

## Validation executed in this build environment

### Targeted policy/release tests

`25 passed`

Covered:

- no-checkpoint fail-closed controller behavior;
- missing checkpoint hard failure;
- explicit No-AI controller contains no neural network;
- native policy registration/qualification/activation;
- incompatible legacy policy activation rejection;
- policy-network cache reuse using a generated valid native checkpoint;
- release version consistency and no bundled `.pt` default policy;
- strict CALO policy-binding default.

### Dependency-light unit partition

`173 passed`

One additional PYPOWER-dependent unit test was deliberately deselected because **PYPOWER is not installed in this audit/build runtime**; it requires `case118` and is unrelated to the v5.4.1 policy/release corrections.

One Qt workflow-restoration unit module could not be collected because **PyQt6 is not installed in this build runtime**.

### Compilation

`python -m compileall -q calo_rpd_studio calo_bootstrap tests` — **PASS**

### Ruff / full GUI / PYPOWER scientific suite

Not executed in this container because the required tooling/dependencies are not installed. No PASS claim is made for those checks in this correction artifact.

## Scientific claim discipline

v5.4.1 is a release-integrity and fail-closed policy-safety correction. It does **not** claim that any CALO policy is best, that policy training is fully corrected, or that the remaining v5.4 audit findings are resolved. A user policy must still be scientifically evaluated/qualified according to the configured study protocol before publication-grade use.
