# Reproducibility

## Seed policy

A `SeedSequence` derives one tuple per repeated-run index:

- algorithm seed;
- scenario seed;
- CALO AI-inference seed.

Every primary algorithm receives the same run-index scenario seed and algorithm seed tuple. CALO policy sampling uses the separately recorded AI-inference seed.

## Fair cost policies

- **Equal objective evaluations** is the default publication comparison.
- **Equal wall-clock time** is available and must retain machine provenance.
- **Algorithm-native limits** are available for method-specific studies and are flagged by the fairness audit.

Population-based and single-solution methods should be compared primarily by objective-function evaluation count unless a different predefined protocol is explicitly justified.

## Experiment record

The database records the experiment configuration and machine/software provenance. Each run records:

- algorithm and parameters;
- seed tuple;
- case checksum;
- objective and scenario configuration;
- objective components;
- total and component-wise constraint violation where available;
- feasibility;
- evaluation and iteration counts;
- runtime and termination reason;
- best-feasible objective and best-constraint-violation histories;
- decoded physical controls;
- reconstructed scenario-wise physical state;
- validation status.

CALO Core v2 additionally records diagnostic histories, regime history, operator use/success information, dual-archive sizes, policy checksum, and evaluations to first exact feasibility.

## CALO policy

The v12 development line does not package or designate a final CALO policy. An empty policy store is
a valid and required safe state. Existing policy files, if still present before the separately
authorized post-freeze cleanup, are development-only, unqualified, inactive, non-final, and excluded
from final-candidate selection, initialization, release evidence, and policy-assisted experiments.

Phase 4 records and verifies empty-policy behavior but does not train, evaluate, qualify, register,
activate, or delete a policy. After the development freeze, an exact inventory and explicitly
authorized cleanup must precede empty-store verification. If the later release path uses a policy,
it must be a completely new A-E/F-off artifact trained without old-policy weights. Its adjacent
metadata must record the training seed and configuration, policy schemas and architecture,
curriculum, training history, protected-asset exclusions, and artifact SHA-256. Qualification,
registration, activation, and immutable experiment binding remain distinct gates. A policy-free
release-validation path is also permitted.

The Phase 4 development-freeze candidate binds the exact Git identity and clean/dirty state; a
sorted SHA-256 manifest of every Git-tracked and non-ignored untracked source file; the ignored
manual validator/instructions; dependency locks; container declarations; exclusion rules;
experiment schema; and the A-E/F-off interfaces. It records zero release-scope policies and whether
it can authorize later new-policy training. Training eligibility is true only for a clean durable
source identity after policy files, external references, and lifecycle rows are empty. A future
campaign must bind both the exact freeze commit and retained freeze payload SHA-256; the training
command validates the complete retained report before it can start or resume. A report with pending
development-only policy files/rows remains valid engineering evidence but is not training-eligible.
It is not a release manifest, policy receipt, RC, scientific qualification, or final freeze.

The manual validator does not accept its own output. A separate post-review acceptance receipt must
verify every retained log hash, the fully passing set of 32 exact result IDs from 30 numbered
stages, the candidate/summary source
identity, and an explicit decision ID. That receipt binds a production-source content digest while
excluding only recorded development-policy artifacts so the later authorized cleanup does not
change the accepted code identity. Future new-policy provenance must record both this Phase 4
acceptance-receipt SHA-256 and the clean, empty-policy post-transition freeze SHA-256.

## Future policy-training reproducibility contract

This section defines the required record for a future, separately authorized post-development
training workflow; it is not a Phase 4 execution instruction. The trainer must record all fields of
`TrainingConfig`, including:

- epochs and episodes;
- horizon;
- random seed;
- learning rate;
- discount factor;
- GAE lambda;
- PPO clip ratio;
- entropy and value weights;
- PPO update epochs;
- minibatch size;
- hidden dimension;
- training population size.

The Python, NumPy, and PyTorch random seeds are set from the declared training seed. The training
environment records the current versioned policy state, action, training, reward, and transition
schemas. It must also prove that the new run did not load or initialize from an old policy.

Executable modes are CUDA-preferred and CPU-only; Intel XPU is not executable. CUDA and CPU
admission records must use no more than 80% of currently free VRAM or currently available RAM,
respectively, and must distinguish requested allocation from measured utilization. Actor
trajectories must bind to the exact current policy snapshot before entering an update buffer.

Training produces a new, unqualified candidate without overwriting or silently selecting a
registered artifact. The candidate must pass the separately authorized real-runtime qualification,
registration, explicit activation, and immutable experiment-binding gates before any
policy-assisted TEST execution. F remains independently flagged and disabled by default.

## Raw data

Important numerical data are stored as JSON/SQLite records and compressed NPZ arrays. Figures are regenerated from raw data; they are not the sole record of a result. Preview visibility and export styling never alter stored scientific data.

## v5.7 telemetry durability boundary

Live GUI telemetry is a convenience preview, not scientific evidence. Publication/statistical reconstruction uses committed optimizer histories, run/revision records, exact checkpoints where supported, stored numerical results, and independent-validation records. A hard process/power failure may lose UI-only transient/downsampled points created after the last committed scientific state; those points are deliberately excluded from publication evidence and are never used to claim exact trajectory reconstruction.


## v5.8 competitive training durability and bounded telemetry

Competitive branch exact-resume state is committed by immutable generation plus one authoritative root manifest. Worker branches write staged states only. Resume-critical history is bounded; full per-epoch training telemetry is append-only external JSONL and is not required to reconstruct optimizer/RNG scientific state. Safe Stop is a typed resumable session outcome, not normal completion.
