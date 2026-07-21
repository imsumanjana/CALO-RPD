"""Run a reproducible command-line optimizer comparison."""

from __future__ import annotations
import argparse
from pathlib import Path
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.experiment_runner import run_sequential_resilient
from calo_rpd_studio.experiments.parallel_runner import run_parallel_resilient
from calo_rpd_studio.experiments.provenance import collect_provenance
from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.results.result_store import ResultStore


def parser():
    p = argparse.ArgumentParser(description="Run a CALO-RPD comparative experiment.")
    p.add_argument("--config")
    p.add_argument("--case", default="case30", choices=["case30", "case57", "case118", "case300"])
    p.add_argument("--algorithms", default="CALO,TLBO,PSO")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--budget", type=int, default=5000)
    p.add_argument("--population", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--database", default="calo_rpd_results.sqlite")
    p.add_argument("--output", default="results_data")
    return p


def main():
    a = parser().parse_args()
    if a.config:
        config = ExperimentConfig.load(a.config)
    else:
        config = ExperimentConfig(
            case_name=a.case,
            algorithms=[x.strip() for x in a.algorithms.split(",") if x.strip()],
            runs=a.runs,
            master_seed=a.seed,
            population_size=a.population,
            output_directory=a.output,
            parallel_workers=a.workers,
        )
        config.budget.max_evaluations = a.budget
    config.validate()
    db = ResultDatabase(a.database)
    eid = db.create_experiment(config, collect_provenance())
    store = ResultStore(config.output_directory)
    done, failed = (
        run_parallel_resilient(config)
        if config.parallel_workers > 1
        else run_sequential_resilient(config)
    )
    for run in done:
        path = store.save_arrays(run.result)
        db.add_run(eid, run, str(path))
        print(
            f"{run.algorithm:>8s} run={run.run_index:02d} objective={run.result.best_objective:.10g} feasible={run.result.feasible} evals={run.result.evaluations}"
        )
    for failure in failed:
        fid = db.add_failure(eid, failure)
        print(
            f"FAILED {failure.algorithm:>8s} run={failure.run_index:02d} type={failure.failure_type} record={fid}"
        )
    print(f"Experiment ID: {eid}")
    print(f"Database: {Path(a.database).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
