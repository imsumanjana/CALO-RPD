# CALO-RPD Studio 1.0.9

This release corrects two execution issues found during long benchmark campaigns.

## CPU utilization and parallel execution

The GUI previously stored a `Parallel workers` value but the Qt experiment worker executed every optimizer job sequentially. Version 1.0.9 now runs independent algorithm/run jobs in a spawn-safe CPU process pool when the worker count is greater than one. Numerical results are still written to SQLite only by the parent process.

Parallel throughput improves completion time for large campaigns. Because simultaneous jobs contend for CPU resources, use one worker when wall-clock runtime itself is a publication metric.

GPU and disk activity are expected to remain low: AC power flow and the 19 baseline metaheuristics are CPU-oriented numerical workloads, and CALO's policy network is deliberately small. The performance target is therefore multi-core CPU throughput rather than artificial GPU utilization.

## Primary comparison versus CALO ablation

A total of 35 jobs with five repeated runs means `7 CALO ablation variants × 5 runs`, not a 20-algorithm comparison. The interface now states the exact plan before execution:

- Primary comparison: selected primary algorithms × repeated runs.
- CALO ablation: seven fixed variants × repeated runs.

The CALO ablation button now explicitly warns that it ignores the primary algorithm checkbox selection.
