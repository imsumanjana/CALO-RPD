# CALO-RPD v5.0 — Scientific Continuation Rules

## Policy training

1. An **exact resume** means the full saved training state is restored: model, optimizer, curriculum/history, RNG states and compatible scientific hyperparameters.
2. Changing learning rate, reward semantics, architecture or other scientific/training parameters is a **new fine-tuning phase/fork**, not exact resume.
3. Every immutable deployable checkpoint may be evaluated independently.
4. `latest` and `best-qualified` are separate roles. Newer is not automatically better.
5. Existing experiments keep their original policy checkpoint SHA even when the lineage later trains further.
6. Indefinite training must use Safe Stop/checkpoints; interruption does not justify silently restarting optimizer state.

## Adding independent runs

1. Increasing an experiment from N to M runs preserves runs `1..N` and schedules only new paired run indices.
2. Seed generation remains deterministic by run index and paired across algorithms.
3. Existing result rows must not be regenerated merely because the target run count increased.
4. Statistical snapshots at the earlier run count remain reproducible historical evidence.

## Extending FE horizon

### Exact segmented continuation

- Allowed only when a complete optimizer-state checkpoint exists.
- CALO v5 supports this.
- The source horizon must be explicit and preserved.
- The saved scientific problem fingerprint must match the resumed case/formulation/scenario definition.
- Only evaluations after the source horizon are newly consumed.
- `5k -> 10k` segmented continuation is not equivalent to a 10k-from-start run if algorithm schedules used the old horizon in the first segment.

### Recompute from original paired seed

- Starts at FE=0 with the larger horizon.
- Applies consistently across compared algorithms.
- Preferred for publication-safe from-start comparisons at a new FE horizon.
- Must never be labeled exact continuation.

## Extension protocols

### All paired

Extend/recompute every relevant algorithm/run pair under the same new horizon. Eligible for primary publication evidence when complete and independently validated.

### Predeclared deterministic subset

A subset is fixed before inspecting outcomes and recorded in the revision protocol. It may support a valid reduced-cost study when scientifically justified.

### Manual/post-hoc exploratory

Runs selected after seeing results are exploratory. They may be useful for algorithm diagnosis but must not be mixed into unbiased primary comparative statistics.

## Evidence boundaries

- Store and analyze each FE horizon separately.
- Store validation status against the exact horizon validated.
- Never let validation of an old snapshot overwrite the current run-head validation status.
- Never silently mix incomplete 5k and 10k rows in one primary statistical table.
- Preserve prior exports; revised horizons/run counts must have distinct evidence identities/directories.

## Policy qualification

A policy checkpoint is scientifically preferred only after recorded Candidate vs Reference vs No-AI CALO qualification under paired equal-FE conditions. Training loss, training return, epoch count or recency alone are insufficient.
