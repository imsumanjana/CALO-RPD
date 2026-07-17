# Portfolio planning and resume model

## Planning

Portfolio Manager converts requested evidence into a dependency graph. It checks minimum repeated runs, selected algorithms, benchmark blocks, robust scenarios, CALO diagnostics, accelerator records, independent validation, and required stored fields. The derived plan is authoritative for repeated-run count.

## Scientific fingerprints

A run fingerprint includes the physical case/formulation, objective, constraints, scenario set, robust aggregation, evaluator, algorithm/version/parameters, policy checksum, evaluation budget, and run seeds. GUI layout, output directory, worker count, device scheduling, portfolio selections, and checkpoint settings are excluded because they do not change the mathematical experiment.

## Resume levels

- **Campaign:** continue only unfinished algorithm/run jobs.
- **Safe pause:** stop new admissions and wait for active jobs to commit.
- **Emergency stop:** completed jobs remain; interrupted jobs restart from original seeds.
- **Policy training:** restart from the last completed PPO epoch; partial on-policy rollouts are discarded.
- **Validation:** continue the remaining unverified IDs.
- **Portfolio export:** reuse each completed artifact listed in the atomic manifest.

## Data integrity

SQLite uses WAL mode and transactional completion. Numeric traces are immutable and may be shared by exact-reuse records; a trace is deleted only after its last database reference is removed. Startup converts stale running/pausing records into interrupted, resumable records.
