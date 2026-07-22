# CALO-RPD v5.7 API and compatibility boundaries

## Release version versus policy ABI

The application release is **5.7.0**. The policy runtime/state/action schema strings intentionally retain the `calo-v4.1` / `calo-state-v4.1-32` generation where the binary/semantic ABI has not changed. An application release number is not a policy ABI number; compatible checkpoints must not be invalidated merely to make the labels cosmetically match.

## Deployable policy versus exact-resume state

Two serialization trust classes are deliberately separate:

1. **Deployable/importable policy artifacts** use the portable checkpoint path and safe `weights_only=True` loading. They may be imported from external sources and must pass schema/checksum inspection before activation.
2. **Exact-resume training state** can contain optimizer/RNG/Python state that requires pickle-capable loading. It is therefore accepted only as a **trusted-local** artifact after SHA-256 plus machine-local HMAC authentication. A downloaded pickle and self-supplied hash are not trusted exact-resume evidence. External policies must use deployable import followed by a Base-Guided Fork when a fresh training trajectory is desired.

`CheckpointManager.write_deployable_model()` is the explicit deployable-model API. `write_torch()` remains a backward-compatible alias for deployable checkpoint callers and must not be used as an exact-resume API.

## Continuation semantics

- `exact_continue`: supported only when the optimizer has a complete exact checkpoint; currently CALO is the supported optimizer-state continuation path.
- `recompute_from_seed`: publication-safe from-start rerun at the larger FE horizon for paired multi-algorithm comparisons.
- These semantics are recorded separately and must never be presented as identical trajectories.

## Compatibility policy

Removed/renamed orchestration helpers are internal unless documented as public. New external integrations should use the stable experiment configuration, ResultDatabase, ExperimentEvolutionService, PolicyRegistry, CheckpointManager deployable API, and ResumeService contracts rather than importing GUI implementation details.
