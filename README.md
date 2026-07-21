# CALO-RPD Studio 5.0.0

**CALO-RPD Studio 5.0.0** is a Python/PyQt6 research platform for deterministic and robust optimal reactive power dispatch (ORPD), reproducible comparison of twenty optimizers, Cognitive Adaptive Learning Optimizer (CALO) research, policy qualification, independent validation, statistics, and publication evidence.

## v5.0 — Continuation without destroying provenance

Version 5.0 adds long-lived research continuation while preserving the v4.1 scientific evaluator, policy qualification rules, immutable experiment-policy binding, workspace restoration, and equal-function-evaluation fairness.

### Policy evolution

- Policy training supports **cumulative target**, **additional epochs**, and **indefinite** modes across multiple application sessions.
- Exact training resume preserves network, optimizer, PPO/curriculum state and Python/NumPy/Torch/CUDA RNG state at trusted SHA-verified checkpoints.
- Periodic model-only checkpoints are immediately usable for policy evaluation or CALO runtime while the same lineage continues training.
- Policy lineages track **latest** and **best-qualified** checkpoints separately. More epochs never automatically mean a better policy.
- A selected checkpoint may be continued/fine-tuned as a new documented phase or forked into a child lineage; old policy artifacts and experiments remain bound to their original SHA-256.
- Safe-stop preserves exact resume state. Atomic checkpoint writes and SHA sidecars protect long training campaigns.

### Experiment evolution

- An existing experiment can increase its independent paired-run target without rerunning or overwriting completed runs.
- Deterministic seed schedules are extended by run index, preserving original paired seeds.
- Experiment revisions record run target, FE horizon, extension mode, protocol, publication eligibility, status and provenance.
- CALO runs write exact full optimizer-state checkpoints and can continue from an old FE horizon to a larger one without hidden fresh-population evaluations.
- The previous run result is snapshotted before an extended result replaces the current run head; run segments and horizon snapshots preserve the complete evidence history.
- Post-hoc selected run extensions are explicitly marked **exploratory** and excluded from unbiased primary statistics. Publication-eligible protocols require paired/predeclared extension.
- **Exact segmented continuation** is available only where a scientifically complete optimizer-state checkpoint exists; CALO v5 supports it. For multi-algorithm publication comparisons, unsupported baselines use the distinct **paired recompute-from-original-seed** strategy at the larger FE horizon. The software never labels recomputation as exact continuation.
- Exact continuation records an explicit **source FE horizon** and writes revision-scoped checkpoints, so a later branch cannot silently resume from the wrong exploratory trajectory or overwrite an earlier checkpoint.
- A segmented `5k → 10k` continuation is not claimed to be identical to a run planned for `10k` from FE=0, because adaptive schedules experienced the original horizon during the first segment. Publication-safe higher-horizon comparison therefore defaults to paired recompute-from-seed when from-start horizon semantics are required.

### Experiment restoration

The v4.1 workspace restoration remains in place: opening/resuming an experiment restores configuration, CALO intelligence and immutable policy binding, workflow access, historical convergence data, selected run/plot state, and stored results where available.

## Scientific rules

CALO never receives hidden extra objective evaluations. A continuation segment starts from an authenticated optimizer checkpoint and all newly requested evaluations count normally. Increasing the number of runs preserves paired seed semantics. Historical or selectively extended evidence is never silently mixed into primary publication statistics.

The bundled/legacy policy is not claimed to be the best ORPD policy. Policy promotion must be based on recorded Candidate vs Reference vs No-AI CALO qualification under paired equal-FE runs. IEEE 118/300 remain protected holdout systems unless a study explicitly documents otherwise.

## Run

```bash
python bootstrap.py
```

Windows launcher and dependency/bootstrap helpers remain included in the repository.


## Important v5.0 limitations and open research work

- CALO cognitive/control state is **not yet fully Torch/CUDA/XPU resident**; the common ORPD numerical evaluator is accelerator-capable, while portions of CALO control remain compact NumPy/Python host logic. No full-device-control claim is made.
- The lightweight PPO training environment uses the native 32-feature policy schema and CALO cognition semantics, but it is **not a bit-identical implementation of the complete runtime transition loop**. Candidate policies still require real-optimizer Policy Qualification before scientific promotion.
- Exact optimizer-state horizon continuation is currently implemented for CALO. Other algorithms can participate fairly at a larger horizon through paired recomputation from their original seeds; they are not falsely labeled exact continuations.
- `qualification_interval_epochs` is currently an advisory scheduling field. Automatic asynchronous qualification/promotion during indefinite training is not yet implemented.
- No bundled policy or CALO configuration is claimed universally superior. Final claims require frozen, paired, feasible, independently validated evidence.
