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

Legacy CALO policy checkpoints are rejected by the Core v2 runtime because their input/output architecture is not compatible with the 24-value constraint-aware state and hierarchical regime/operator/Beta-parameter controller.

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

The Python, NumPy, and PyTorch random seeds are set from the declared training seed. The training environment reuses the runtime CALO Core v2 operator and selection modules.

## Raw data

Important numerical data are stored as JSON/SQLite records and compressed NPZ arrays. Figures are regenerated from raw data; they are not the sole record of a result. Preview visibility and export styling never alter stored scientific data.
