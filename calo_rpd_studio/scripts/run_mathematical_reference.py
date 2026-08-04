"""Run a source-bound mathematical ORPD reference and write immutable JSON evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

from calo_rpd_studio.compute.source_identity import resolve_source_identity
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.experiment_runner import build_scenarios
from calo_rpd_studio.orpd.mathematical_reference import (
    MATHEMATICAL_REFERENCE_SCHEMA,
    SLSQPReferenceOptions,
    solve_exhaustive_finite_lattice_reference,
    solve_slsqp_continuous_reference,
)
from calo_rpd_studio.orpd.problem import ORPDProblem, ORPDProblemConfig
from calo_rpd_studio.power_system.case_identity import protected_holdout_matches
from calo_rpd_studio.power_system.case_loader import CaseLoader


REFERENCE_EVIDENCE_ENVELOPE_SCHEMA = "calo-rpd-mathematical-reference-evidence-v1"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_mapping(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    if path.suffix.lower() in {".yaml", ".yml"}:
        payload = yaml.safe_load(raw.decode("utf-8"))
    else:
        payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("Reference configuration must contain one JSON/YAML object")
    return payload, _sha256(raw)


def load_reference_problem(
    config_path: str | Path,
    *,
    scenario_seed: int,
) -> tuple[ORPDProblem, ExperimentConfig, str]:
    """Load only the scientific task definition on the trusted CPU reference path."""

    source = Path(config_path)
    payload, config_sha256 = _read_mapping(source)
    config = ExperimentConfig.from_dict(payload)
    config.validate_policy_development()
    protected = protected_holdout_matches((config.case_name,))
    if protected:
        raise ValueError(
            "Mathematical-reference development command refuses protected holdout cases: "
            + ", ".join(protected)
        )
    case = CaseLoader.load(config.case_name)
    scenarios = build_scenarios(config, int(scenario_seed), case)
    problem = ORPDProblem(
        case,
        ORPDProblemConfig(
            objective=config.objective,
            variables=config.variables,
            robust=config.robust_objective,
            power_flow=config.power_flow,
            constraint_tolerances=config.constraint_tolerances,
        ),
        scenarios,
    )
    return problem, config, config_sha256


def load_start_vector(path: str | Path, *, dimension: int) -> tuple[np.ndarray, str]:
    source = Path(path)
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list):
        raise TypeError("SLSQP start file must contain one JSON array")
    vector = np.asarray(payload, dtype=float)
    if vector.shape != (int(dimension),):
        raise ValueError(f"Expected start vector shape ({dimension},), got {vector.shape}")
    return vector, _sha256(raw)


def write_reference_evidence(path: str | Path, payload: dict) -> Path:
    """Write one evidence envelope durably and refuse replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    try:
        with destination.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite reference evidence: {destination}") from exc
    return destination


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", required=True, help="Frozen ExperimentConfig JSON/YAML")
    command.add_argument("--output", required=True, help="New JSON evidence path")
    command.add_argument("--scenario-seed", required=True, type=int)
    command.add_argument(
        "--mode",
        required=True,
        choices=("slsqp-relaxation", "finite-lattice-exhaustive"),
    )
    command.add_argument("--start", help="JSON normalized start vector; required for SLSQP")
    command.add_argument("--max-iterations", type=int, default=200)
    command.add_argument("--function-tolerance", type=float, default=1e-9)
    command.add_argument("--finite-difference", choices=("2-point", "3-point"), default="2-point")
    command.add_argument("--maximum-lattice-candidates", type=int, default=10_000)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "slsqp-relaxation" and not args.start:
        raise SystemExit("--start is required for slsqp-relaxation")
    if args.mode == "finite-lattice-exhaustive" and args.start:
        raise SystemExit("--start is not accepted for finite-lattice-exhaustive")

    source_identity = resolve_source_identity(cwd=Path.cwd(), require_durable=True)
    problem, config, config_sha256 = load_reference_problem(
        args.config,
        scenario_seed=args.scenario_seed,
    )
    start_sha256 = ""
    if args.mode == "slsqp-relaxation":
        start, start_sha256 = load_start_vector(args.start, dimension=problem.dimension)
        report = solve_slsqp_continuous_reference(
            problem,
            start,
            options=SLSQPReferenceOptions(
                max_iterations=args.max_iterations,
                function_tolerance=args.function_tolerance,
                finite_difference_scheme=args.finite_difference,
            ),
            run_independent_validation=True,
        )
    else:
        report = solve_exhaustive_finite_lattice_reference(
            problem,
            maximum_candidates=args.maximum_lattice_candidates,
            run_independent_validation=True,
        )
    envelope = {
        "schema_version": REFERENCE_EVIDENCE_ENVELOPE_SCHEMA,
        "reference_schema_version": MATHEMATICAL_REFERENCE_SCHEMA,
        "source_identity": source_identity.to_dict(),
        "input": {
            "config_sha256": config_sha256,
            "case_name": str(config.case_name),
            "case_checksum": str(problem.case.checksum()),
            "scenario_seed": int(args.scenario_seed),
            "mode": str(args.mode),
            "start_sha256": start_sha256,
        },
        "report_sha256": report.sha256(),
        "report": report.to_dict(),
        "claim_boundary": (
            "No continuous-relaxation lower bound or optimality gap is claimed unless a future "
            "adapter supplies separate certification. Protected holdouts are refused."
        ),
    }
    destination = write_reference_evidence(args.output, envelope)
    print(destination)
    print(f"report_sha256={report.sha256()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
