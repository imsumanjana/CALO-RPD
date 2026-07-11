# Reproducibility

## Seed policy

A `SeedSequence` derives one tuple per run index:

- algorithm seed;
- scenario seed;
- CALO AI-inference seed.

Every primary algorithm receives the same run-index scenario seed and algorithm seed tuple. CALO's
policy sampling uses the separately recorded AI-inference seed.

## Fair cost policies

- **Equal objective evaluations** is the default publication comparison.
- **Equal wall-clock time** is available and must retain machine provenance.
- **Algorithm-native limits** are available for method-specific studies and are flagged by the fairness audit.

## Experiment record

The database records the experiment configuration and machine/software provenance. Each run records
algorithm parameters, seeds, objective values, raw objective components, constraint violation,
feasibility, evaluation count, iteration count, runtime, termination reason, convergence history, decoded
controls, policy checksum where applicable, and reconstructed physical system state.

## CALO policy

The packaged checkpoint has adjacent JSON metadata containing its training seed, configuration,
training problem identifiers, and explicit final-test leakage flag. The checkpoint SHA-256 hash is
reported in CALO result metadata.

## Raw data

Important numerical data are stored as JSON/SQLite records and compressed NPZ arrays. Figures are
regenerated from data; they are not the sole record of a result.
