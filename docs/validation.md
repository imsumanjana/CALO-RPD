# Scientific Validation

## Case validation

Case loading checks unique bus numbers, exactly one reference bus, referenced bus existence, voltage
bounds, generator P/Q limit consistency, and structural data integrity.

## Base power flow

The internal Newton-Raphson solver reports convergence, iteration count, mismatch history, reactive-limit
switching, branch complex power, loading, and total active loss. Where PYPOWER is installed, the final
electrical formulation is independently solved and compared by voltage magnitude, voltage angle, and
loss tolerance.

When internal PV-to-PQ switching has occurred, the independent solver receives the resulting final bus
types and reactive limit states and solves that formulation with its own Newton engine. This avoids
relying on compatibility-sensitive external Q-limit loop code while still cross-checking the converged
network state.

## Best-solution validation

A saved decision vector is reloaded from persistent storage, decoded again, evaluated across the original
seeded scenario set, and compared with its stored objective. The audit explicitly checks:

- objective agreement;
- decision-vector validity;
- power-flow convergence;
- bus-voltage limits;
- aggregate generator P and Q limits;
- branch thermal limits;
- scenario-wise constraint violation.

Only runs marked `verified` are included in publication export.

## Statistical claims

The application provides descriptive statistics, confidence intervals, Wilcoxon testing, Friedman testing,
Holm correction, Cliff's delta, and average ranks. A lowest single-run objective is not, by itself, treated
as statistical superiority.
