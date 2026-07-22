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

The packaged CALO Core v2 checkpoint has adjacent JSON metadata containing:

- training seed;
- complete training configuration;
- policy architecture dimensions;
- curriculum stages;
- training history;
- explicit statement that final publication benchmark cases were not silently used for training.

The checkpoint SHA-256 hash is stored in CALO result metadata. Final benchmark studies should freeze the selected checkpoint before execution. Changing the checkpoint creates a different experiment configuration.

Legacy/incompatible CALO policy checkpoints may remain visible for provenance, but they cannot become the active runtime policy. Policy-assisted execution requires a current compatible policy schema, explicit activation, and immutable experiment SHA-256 binding.

## PPO training reproducibility

The policy trainer records all fields of `TrainingConfig`, including:

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

The Python, NumPy, and PyTorch random seeds are set from the declared training seed. The training environment records the current versioned policy state/action/training schemas. Candidate policies must still pass real-runtime qualification before scientific promotion.

In weighted heterogeneous mode, each PPO epoch records the requested CUDA/XPU/CPU shares, the effective integer episode allocation, the actor devices, and the policy-snapshot SHA-256 used by every lane. All actor trajectories must match the current snapshot before entering the PPO buffer. The update begins only after the synchronous CUDA, XPU, and CPU actors have completed. Configured shares refer to rollout episodes/transitions rather than measured hardware utilization.

Training produces candidate checkpoints without overwriting any registered policy artifact. CALO-RPD does not fabricate or silently choose a default neural policy; a candidate must be registered, qualified as required, explicitly activated, and immutably bound to an experiment before policy-assisted TEST execution.

## Raw data

Important numerical data are stored as JSON/SQLite records and compressed NPZ arrays. Figures are regenerated from raw data; they are not the sole record of a result. Preview visibility and export styling never alter stored scientific data.

## v5.7 telemetry durability boundary

Live GUI telemetry is a convenience preview, not scientific evidence. Publication/statistical reconstruction uses committed optimizer histories, run/revision records, exact checkpoints where supported, stored numerical results, and independent-validation records. A hard process/power failure may lose UI-only transient/downsampled points created after the last committed scientific state; those points are deliberately excluded from publication evidence and are never used to claim exact trajectory reconstruction.


## v5.8 competitive training durability and bounded telemetry

Competitive branch exact-resume state is committed by immutable generation plus one authoritative root manifest. Worker branches write staged states only. Resume-critical history is bounded; full per-epoch training telemetry is append-only external JSONL and is not required to reconstruct optimizer/RNG scientific state. Safe Stop is a typed resumable session outcome, not normal completion.
