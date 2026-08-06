"""Publication-grade paired CALO policy qualification.

Formal qualification is deliberately stronger than training-time champion selection.  Candidate,
reference and explicit No-AI CALO are evaluated under identical paired seeds/FE budgets, case-wise
rather than by pooling incomparable raw objective scales, and every candidate result must pass the
independent PYPOWER cross-validation gate before it can be promoted.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import uuid

import numpy as np

from calo_rpd_studio.experiments.seed_manager import SeedManager
from calo_rpd_studio.experiments.experiment_runner import run_single, build_problem
from calo_rpd_studio.compute.source_identity import resolve_source_identity
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.statistics.paired import (
    DEFAULT_OBJECTIVE_SCALE_FLOOR,
    PAIRED_ANALYSIS_SCHEMA_VERSION,
    RELATIVE_IMPROVEMENT_VERSION,
    exact_keyed_pairs,
    matched_pairs_rank_biserial,
    pair_manifest,
    relative_objective_improvement,
    wilcoxon_signed_rank_evidence,
)
from calo_rpd_studio.statistics.posthoc import holm_correction

_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

HOLDOUT_CASES = {"case118", "case300"}
CONVERGENCE_METRIC_VERSION = "calo-convergence-v2-time-to-feasible-plus-post-feasible-auc"


@dataclass(slots=True)
class PolicyQualificationConfig:
    cases: tuple[str, ...] = ("case30", "case57")
    runs: int = 30
    max_evaluations: int = 1000
    population_size: int = 40
    master_seed: int = 20260410
    allow_holdout_cases: bool = False
    objective_regression_tolerance: float = 0.01
    minimum_feasible_probability: float = 0.90
    minimum_promotion_runs: int = 30
    require_independent_validation: bool = True
    qualification_mode: str = "superiority"  # superiority | non_inferiority
    statistical_alpha: float = 0.05
    minimum_win_rate: float = 0.50
    minimum_rank_biserial: float = 0.0
    non_inferiority_margin: float = 0.01
    objective_scale_floor: float = DEFAULT_OBJECTIVE_SCALE_FLOOR

    def validate(self) -> None:
        if not self.cases:
            raise ValueError("Policy qualification requires at least one development case")
        if int(self.runs) < 2:
            raise ValueError("Policy qualification requires at least two paired runs")
        if int(self.max_evaluations) <= 0 or int(self.population_size) <= 1:
            raise ValueError("Qualification FE budget and population must be positive")
        if int(self.max_evaluations) % int(self.population_size) != 0:
            raise ValueError(
                "Policy qualification FE budget must be divisible by population size for exact "
                "FE parity"
            )
        if int(self.minimum_promotion_runs) < 5:
            raise ValueError("minimum_promotion_runs must be at least 5")
        if str(self.qualification_mode) not in {"superiority", "non_inferiority"}:
            raise ValueError("qualification_mode must be superiority or non_inferiority")
        if not 0.0 < float(self.statistical_alpha) < 1.0:
            raise ValueError("statistical_alpha must be between 0 and 1")
        if (
            not math.isfinite(float(self.objective_regression_tolerance))
            or float(self.objective_regression_tolerance) < 0.0
        ):
            raise ValueError("objective_regression_tolerance must be finite and non-negative")
        if (
            not math.isfinite(float(self.minimum_feasible_probability))
            or not 0.0 <= float(self.minimum_feasible_probability) <= 1.0
        ):
            raise ValueError("minimum_feasible_probability must be finite and between 0 and 1")
        if not 0.0 <= float(self.minimum_win_rate) <= 1.0:
            raise ValueError("minimum_win_rate must be between 0 and 1")
        if (
            not math.isfinite(float(self.minimum_rank_biserial))
            or not -1.0 <= float(self.minimum_rank_biserial) <= 1.0
        ):
            raise ValueError("minimum_rank_biserial must be finite and between -1 and 1")
        if (
            not math.isfinite(float(self.non_inferiority_margin))
            or float(self.non_inferiority_margin) < 0.0
        ):
            raise ValueError("non_inferiority_margin must be finite and non-negative")
        if (
            not math.isfinite(float(self.objective_scale_floor))
            or float(self.objective_scale_floor) <= 0.0
        ):
            raise ValueError("objective_scale_floor must be finite and positive")
        leaked = HOLDOUT_CASES & {str(name).lower() for name in self.cases}
        if leaked and not self.allow_holdout_cases:
            raise ValueError(
                "IEEE 118/300 are protected holdout cases for policy qualification by default: "
                + ", ".join(sorted(leaked))
            )


def _finite_objective(result) -> float:
    value = float(result.best_objective)
    return value if bool(result.feasible) and math.isfinite(value) else float("nan")


def _convergence_auc(result) -> float:
    """Return the post-feasibility incumbent-objective AUC.

    Time to first feasibility is reported separately by :func:`_eval_to_feasible`.  Separating the
    two quantities avoids a run-dependent synthetic penalty and keeps objective AUC values
    comparable without folding feasibility delay into an arbitrary objective scale.
    """
    md = dict(getattr(result, "metadata", {}) or {})
    xs = np.asarray(md.get("convergence_evaluations", []), dtype=float)
    ys = np.asarray(md.get("best_feasible_objective_history", []), dtype=float)
    if xs.size == 0 or xs.size != ys.size:
        return float("inf")
    if not np.all(np.isfinite(xs)) or np.any(np.diff(xs) < 0.0):
        return float("inf")
    feasible_indices = np.flatnonzero(np.isfinite(ys))
    if feasible_indices.size == 0:
        return float("inf")
    start = int(feasible_indices[0])
    xs = xs[start:]
    ys = ys[start:].copy()
    incumbent = float(ys[0])
    for index in range(ys.size):
        if math.isfinite(float(ys[index])):
            incumbent = float(ys[index])
        else:
            ys[index] = incumbent
    horizon = max(float(getattr(result, "evaluations", 0) or 0), float(xs[-1]), 1.0)
    if xs.size == 1 or horizon <= float(xs[0]):
        return float(ys[0])
    if xs[-1] < horizon:
        xs = np.r_[xs, horizon]
        ys = np.r_[ys, ys[-1]]
    duration = max(float(horizon - xs[0]), 1.0)
    return float(_trapezoid(ys, xs) / duration)


def _eval_to_feasible(result) -> float:
    value = (getattr(result, "metadata", {}) or {}).get("first_feasible_evaluation")
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _aggregate(records: list[dict]) -> dict:
    objectives = np.asarray([row["objective"] for row in records], float)
    finite = objectives[np.isfinite(objectives)]
    aucs = np.asarray([row["auc"] for row in records], float)
    auc_finite = aucs[np.isfinite(aucs)]
    etf = np.asarray([row["eval_to_feasible"] for row in records], float)
    etf_finite = etf[np.isfinite(etf)]
    runtimes = np.asarray([row["runtime"] for row in records], float)
    independent = [bool(row.get("independent_validation_passed", False)) for row in records]
    return {
        "n": len(records),
        "feasible_probability": float(np.mean([row["feasible"] for row in records]))
        if records
        else 0.0,
        "independent_validation_probability": float(np.mean(independent)) if records else 0.0,
        "median_objective": float(np.median(finite)) if finite.size else float("nan"),
        "mean_objective": float(np.mean(finite)) if finite.size else float("nan"),
        "std_objective": float(np.std(finite, ddof=1))
        if finite.size > 1
        else (0.0 if finite.size == 1 else float("nan")),
        "iqr_objective": float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))
        if finite.size
        else float("nan"),
        "median_auc": float(np.median(auc_finite)) if auc_finite.size else float("inf"),
        "median_eval_to_feasible": float(np.median(etf_finite))
        if etf_finite.size
        else float("nan"),
        "mean_runtime_seconds": float(np.mean(runtimes)) if runtimes.size else float("nan"),
    }


def _case_summaries(records: list[dict]) -> dict[str, dict]:
    cases = sorted({str(row["case"]) for row in records})
    return {
        case: _aggregate([row for row in records if str(row["case"]) == case]) for case in cases
    }


def _stable_relative_difference(candidate: float, comparator: float) -> float:
    """Compatibility alias for the v12 positive-is-better relative improvement."""
    return relative_objective_improvement(candidate, comparator)


def _paired_evidence(
    candidate_rows: list[dict],
    comparator_rows: list[dict],
    *,
    objective_scale_floor: float = DEFAULT_OBJECTIVE_SCALE_FLOOR,
) -> dict:
    keyed = exact_keyed_pairs(
        candidate_rows,
        comparator_rows,
        key_fields=("case", "run_index"),
    )
    finite_pairs = []
    for pair in keyed:
        candidate = float(pair.candidate["objective"])
        comparator = float(pair.comparator["objective"])
        if math.isfinite(candidate) and math.isfinite(comparator):
            finite_pairs.append((pair.key, candidate, comparator))
    if not finite_pairs:
        statistical_test = wilcoxon_signed_rank_evidence((), alternative="two-sided")
        return {
            "analysis_schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
            "relative_improvement_version": RELATIVE_IMPROVEMENT_VERSION,
            "objective_scale_floor": float(objective_scale_floor),
            "expected_pairs": int(len(keyed)),
            "n_pairs": 0,
            "median_improvement": float("nan"),
            "median_difference": float("nan"),  # compatibility alias; positive is favorable in v12
            "win_rate": float("nan"),
            "wilcoxon_p_two_sided": float("nan"),
            "holm_p": float("nan"),
            "noninferiority_p_one_sided": float("nan"),
            "holm_noninferiority_p": float("nan"),
            "rank_biserial": float("nan"),
            "paired_relative_improvements": [],
            "paired_relative_differences": [],
            "statistical_test": statistical_test,
            "pair_manifest": pair_manifest(keyed, ("case", "run_index")),
        }
    improvements = np.asarray(
        [
            relative_objective_improvement(
                candidate,
                comparator,
                scale_floor=float(objective_scale_floor),
            )
            for _, candidate, comparator in finite_pairs
        ],
        dtype=float,
    )
    statistical_test = wilcoxon_signed_rank_evidence(improvements, alternative="two-sided")
    pvalue = statistical_test["p_value"]
    median = float(np.median(improvements))
    return {
        "analysis_schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
        "relative_improvement_version": RELATIVE_IMPROVEMENT_VERSION,
        "objective_scale_floor": float(objective_scale_floor),
        "expected_pairs": int(len(keyed)),
        "n_pairs": int(len(finite_pairs)),
        "median_improvement": median,
        "median_difference": median,  # compatibility alias; positive is favorable in v12
        "median_relative_improvement": median,
        "median_relative_difference": median,
        "win_rate": float(np.mean(improvements > 0.0)),
        "wilcoxon_p_two_sided": float(pvalue) if pvalue is not None else float("nan"),
        "holm_p": float("nan"),
        "noninferiority_p_one_sided": float("nan"),
        "holm_noninferiority_p": float("nan"),
        "paired_relative_improvements": improvements.tolist(),
        "paired_relative_differences": improvements.tolist(),
        "rank_biserial": matched_pairs_rank_biserial(improvements),
        "statistical_test": statistical_test,
        "pair_manifest": pair_manifest(keyed, ("case", "run_index")),
    }


def _apply_holm(paired: dict, *, non_inferiority_margin: float = 0.0) -> dict:
    keys, pvalues = [], []
    ni_keys, ni_pvalues = [], []
    for key, item in paired.items():
        p = float(item.get("wilcoxon_p_two_sided", float("nan")))
        if math.isfinite(p):
            keys.append(key)
            pvalues.append(p)
        improvements = np.asarray(item.get("paired_relative_improvements") or [], dtype=float)
        shifted = improvements[np.isfinite(improvements)] + float(non_inferiority_margin)
        ni_p = float("nan")
        ni_test = wilcoxon_signed_rank_evidence(shifted, alternative="greater")
        if ni_test["p_value"] is not None:
            ni_p = float(ni_test["p_value"])
        item["noninferiority_p_one_sided"] = ni_p
        item["noninferiority_statistical_test"] = ni_test
        if math.isfinite(ni_p):
            ni_keys.append(key)
            ni_pvalues.append(ni_p)
    corrected = holm_correction(pvalues) if pvalues else []
    for key, p in zip(keys, corrected, strict=True):
        paired[key]["holm_p"] = float(p)
    ni_corrected = holm_correction(ni_pvalues) if ni_pvalues else []
    for key, p in zip(ni_keys, ni_corrected, strict=True):
        paired[key]["holm_noninferiority_p"] = float(p)
    return paired


def _grade(candidate, reference, no_ai, config, paired, case_summaries, case_paired=None):
    reasons: list[str] = []
    if int(config.runs) < int(config.minimum_promotion_runs):
        reasons.append(
            "formal promotion requires at least "
            f"{config.minimum_promotion_runs} paired runs per case; this run is screening-only"
        )
    feasible = float(candidate["feasible_probability"])
    if feasible < float(config.minimum_feasible_probability):
        reasons.append(
            f"feasible probability {feasible:.3f} is below {config.minimum_feasible_probability:.3f}"
        )
    if config.require_independent_validation:
        if float(candidate.get("independent_validation_probability", 0.0)) < 1.0:
            reasons.append(
                "not every candidate run passed mandatory independent PYPOWER validation"
            )
        if float(no_ai.get("independent_validation_probability", 0.0)) < 1.0:
            reasons.append(
                "not every No-AI comparator run passed mandatory independent PYPOWER validation"
            )
        if (
            reference is not None
            and float(reference.get("independent_validation_probability", 0.0)) < 1.0
        ):
            reasons.append(
                "not every reference-policy run passed mandatory independent PYPOWER validation"
            )

    cand_cases = case_summaries["candidate"]
    comparator_labels = ["no_ai"] + (["reference"] if reference is not None else [])
    relative_case_improvements = []
    for case, cand_case in cand_cases.items():
        cand_med = float(cand_case.get("median_objective", float("nan")))
        if not math.isfinite(cand_med):
            reasons.append(f"{case}: candidate has no finite feasible median objective")
            continue
        for label in comparator_labels:
            comp_case = case_summaries.get(label, {}).get(case, {})
            comp_med = float(comp_case.get("median_objective", float("nan")))
            if not math.isfinite(comp_med):
                continue
            tolerance = abs(comp_med) * float(config.objective_regression_tolerance)
            if cand_med > comp_med + tolerance:
                reasons.append(f"{case}: candidate materially regresses versus {label}")
            relative_case_improvements.append(
                relative_objective_improvement(
                    cand_med,
                    comp_med,
                    scale_floor=float(config.objective_scale_floor),
                )
            )

    evidence_source = case_paired if case_paired else paired
    evidence_rows = [item for item in evidence_source.values()]
    favorable = bool(evidence_rows) and all(
        int(item.get("n_pairs", 0)) >= int(config.minimum_promotion_runs)
        and math.isfinite(float(item.get("median_difference", float("nan"))))
        and float(item["median_difference"]) >= 0.0
        and float(item.get("win_rate", 0.0)) >= float(config.minimum_win_rate)
        and float(item.get("rank_biserial", -1.0)) >= float(config.minimum_rank_biserial)
        and math.isfinite(float(item.get("holm_p", float("nan"))))
        and float(item["holm_p"]) <= float(config.statistical_alpha)
        for item in evidence_rows
    )
    if str(config.qualification_mode) == "superiority":
        if not favorable:
            reasons.append(
                "formal superiority promotion requires complete paired favorable-direction evidence, "
                "the predeclared minimum effect/win gates, and Holm-adjusted statistical significance"
            )
    else:
        # Non-inferiority is a separate protocol: every comparator must remain within the declared
        # relative margin. It is never mislabeled as statistical superiority.
        noninferior = bool(evidence_rows) and all(
            int(item.get("n_pairs", 0)) >= int(config.minimum_promotion_runs)
            and math.isfinite(float(item.get("median_difference", float("nan"))))
            and float(item["median_difference"]) >= -float(config.non_inferiority_margin)
            and math.isfinite(float(item.get("holm_noninferiority_p", float("nan"))))
            and float(item["holm_noninferiority_p"]) <= float(config.statistical_alpha)
            for item in evidence_rows
        )
        if not noninferior:
            reasons.append(
                "formal non-inferiority qualification requires the paired relative margin plus a "
                "one-sided Wilcoxon signed-rank criterion with Holm-adjusted significance"
            )
    passed = not reasons
    if not passed:
        return False, "U", 0.0, reasons

    median_rel = float(np.median(relative_case_improvements)) if relative_case_improvements else 0.0
    auc_nonregression = True
    for case, cand_case in cand_cases.items():
        cand_auc = float(cand_case.get("median_auc", float("inf")))
        for label in comparator_labels:
            comp_auc = float(
                case_summaries.get(label, {}).get(case, {}).get("median_auc", float("inf"))
            )
            if math.isfinite(comp_auc):
                scale = max(abs(comp_auc), float(config.objective_scale_floor))
                if not math.isfinite(cand_auc) or cand_auc > comp_auc + 0.01 * scale:
                    auc_nonregression = False
    if median_rel > 0.01 and feasible >= 0.99 and favorable and auc_nonregression:
        grade = "A+"
    elif median_rel >= -0.002 and feasible >= 0.99:
        grade = "A"
    elif median_rel >= -float(config.objective_regression_tolerance):
        grade = "B+"
    else:
        grade = "B"
    score = {"B": 1.0, "B+": 2.0, "A": 3.0, "A+": 4.0}[grade]
    return True, grade, score, reasons


def _independent_validate_result(cfg, seeds, result) -> dict:
    problem = build_problem(cfg, seeds.scenario_seed)
    controlled, _ = problem.decoder.decode(np.asarray(result.best_vector, dtype=float))
    checks = []
    for scenario in problem.scenarios:
        formulation_case = scenario.apply(controlled)
        internal = run_ac_power_flow(formulation_case, cfg.power_flow)
        try:
            from calo_rpd_studio.power_system.independent_validator import validate_against_pypower
        except ModuleNotFoundError as exc:
            return {
                "available": False,
                "passed": False,
                "reason": f"independent_validator_unavailable:{exc}",
            }
        cross = validate_against_pypower(
            formulation_case, internal, power_flow_options=cfg.power_flow
        )
        checks.append(
            {
                "scenario": scenario.name,
                "available": bool(cross.available),
                "passed": bool(cross.passed),
                "message": str(cross.message),
                "max_vm_difference": float(cross.max_vm_difference),
                "max_va_difference_deg": float(cross.max_va_difference_deg),
                "loss_difference_mw": float(cross.loss_difference_mw),
            }
        )
    return {
        "available": bool(checks) and all(item["available"] for item in checks),
        "passed": bool(checks) and all(item["available"] and item["passed"] for item in checks),
        "scenarios": checks,
    }


class PolicyQualifier:
    def __init__(self, base_config, registry) -> None:
        self.base_config = base_config
        self.registry = registry

    def run(
        self,
        candidate_policy_id: str,
        *,
        reference_policy_id: str = "",
        config=None,
        progress_callback=None,
        cancel_callback=None,
    ) -> dict:
        qconfig = config or PolicyQualificationConfig()
        qconfig.validate()
        source_identity = resolve_source_identity()
        candidate = self.registry.get(candidate_policy_id)
        reference = self.registry.get(reference_policy_id) if reference_policy_id else None
        candidate_inspection = self.registry.inspect_checkpoint(candidate.checkpoint_path)
        if candidate_inspection["sha256"] != candidate.sha256:
            raise RuntimeError(
                "Candidate policy checksum does not match the registered immutable artifact"
            )
        if reference is not None:
            ref_inspection = self.registry.inspect_checkpoint(reference.checkpoint_path)
            if ref_inspection["sha256"] != reference.sha256:
                raise RuntimeError(
                    "Reference policy checksum does not match the registered immutable artifact"
                )
        participants = [("candidate", candidate), ("no_ai", None)]
        if reference is not None and reference.id != candidate.id:
            participants.insert(1, ("reference", reference))
        total = len(qconfig.cases) * qconfig.runs * len(participants)
        done = 0
        records = {name: [] for name, _ in participants}
        paired_seeds = SeedManager(qconfig.master_seed).generate(qconfig.runs)
        for case_name in qconfig.cases:
            for run_index in range(qconfig.runs):
                for label, policy in participants:
                    if cancel_callback and cancel_callback():
                        raise RuntimeError("Policy qualification cancelled")
                    cfg = deepcopy(self.base_config)
                    cfg.case_name = str(case_name)
                    cfg.algorithms = ["CALO"]
                    cfg.runs = 1
                    cfg.population_size = int(qconfig.population_size)
                    cfg.budget.max_evaluations = int(qconfig.max_evaluations)
                    cfg.max_iterations = max(int(cfg.max_iterations), int(qconfig.max_evaluations))
                    params = dict(cfg.algorithm_parameters.get("CALO", {}))
                    params.update(
                        {
                            "strict_benchmark_mode": True,
                            "use_historical_parameter_priors": False,
                            "use_cross_algorithm_warm_start": False,
                        }
                    )
                    if policy is None:
                        params["use_ai"] = False
                        params.pop("policy_checkpoint", None)
                        params.pop("policy_sha256", None)
                        params["strict_policy_binding"] = False
                    else:
                        params.update(
                            {
                                "use_ai": True,
                                "policy_id": policy.id,
                                "policy_checkpoint": policy.checkpoint_path,
                                "policy_sha256": policy.sha256,
                                "policy_state_schema_version": policy.state_schema_version,
                                "policy_action_schema_version": policy.action_schema_version,
                                "policy_architecture_version": policy.architecture_version,
                                "policy_training_environment_version": policy.training_environment_version,
                                "strict_policy_binding": True,
                                "allow_unqualified_policy": True,
                                "deterministic_policy": True,
                            }
                        )
                    cfg.algorithm_parameters["CALO"] = params
                    seeds = paired_seeds[run_index]
                    completed = run_single(cfg, "CALO", run_index, seeds)
                    result = completed.result
                    independent = _independent_validate_result(cfg, seeds, result)
                    records[label].append(
                        {
                            "case": str(case_name),
                            "run_index": int(run_index),
                            "objective": _finite_objective(result),
                            "feasible": bool(result.feasible),
                            "auc": _convergence_auc(result),
                            "eval_to_feasible": _eval_to_feasible(result),
                            "runtime": float(result.runtime_seconds),
                            "evaluations": int(result.evaluations),
                            "independent_validation_available": bool(independent["available"]),
                            "independent_validation_passed": bool(independent["passed"]),
                            "independent_validation": independent,
                        }
                    )
                    done += 1
                    if progress_callback:
                        progress_callback(
                            int(100 * done / max(total, 1)),
                            f"{done}/{total} · {case_name} · run {run_index + 1} · {label}",
                        )
        summaries = {name: _aggregate(rows) for name, rows in records.items()}
        case_summaries = {name: _case_summaries(rows) for name, rows in records.items()}
        # Aggregate paired evidence is retained for concise UI display, while formal promotion is
        # gated case-by-case.  This prevents a strong/easy case from statistically masking a weak
        # case even after objective-scale normalization. Holm correction is applied across every
        # required comparator x case hypothesis in the formal gate.
        paired = {
            "vs_no_ai": _paired_evidence(
                records["candidate"],
                records["no_ai"],
                objective_scale_floor=float(qconfig.objective_scale_floor),
            )
        }
        if "reference" in records:
            paired["vs_reference"] = _paired_evidence(
                records["candidate"],
                records["reference"],
                objective_scale_floor=float(qconfig.objective_scale_floor),
            )
        paired = _apply_holm(paired, non_inferiority_margin=float(qconfig.non_inferiority_margin))
        case_paired = {}
        for case_name in qconfig.cases:
            candidate_case = [
                row for row in records["candidate"] if str(row["case"]) == str(case_name)
            ]
            no_ai_case = [row for row in records["no_ai"] if str(row["case"]) == str(case_name)]
            case_paired[f"vs_no_ai::{case_name}"] = _paired_evidence(
                candidate_case,
                no_ai_case,
                objective_scale_floor=float(qconfig.objective_scale_floor),
            )
            if "reference" in records:
                reference_case = [
                    row for row in records["reference"] if str(row["case"]) == str(case_name)
                ]
                case_paired[f"vs_reference::{case_name}"] = _paired_evidence(
                    candidate_case,
                    reference_case,
                    objective_scale_floor=float(qconfig.objective_scale_floor),
                )
        case_paired = _apply_holm(
            case_paired, non_inferiority_margin=float(qconfig.non_inferiority_margin)
        )
        passed, grade, score, reasons = _grade(
            summaries["candidate"],
            summaries.get("reference"),
            summaries["no_ai"],
            qconfig,
            paired,
            case_summaries,
            case_paired,
        )
        if passed and not source_identity.durable_evidence_eligible:
            passed, grade, score = False, "U", 0.0
            reasons.append(
                "formal promotion requires a full clean source identity; this evidence is "
                "development-only"
            )
        schema = candidate_inspection["schema"]
        return {
            "qualification_id": str(uuid.uuid4()),
            "candidate_policy_id": candidate.id,
            "reference_policy_id": reference.id if reference else "",
            "candidate_policy_sha256": candidate.sha256,
            "candidate_policy_schema": schema,
            "native_v59": bool(schema.get("native_v59", False)),
            "native_v41": bool(schema.get("native_v59", False)),
            "reference_policy_sha256": reference.sha256 if reference else "",
            "config": asdict(qconfig),
            "participants": summaries,
            "case_summaries": case_summaries,
            "records": records,
            "paired_evidence": paired,
            "case_paired_evidence": case_paired,
            "passed": bool(passed),
            "grade": grade,
            "score": score,
            "reasons": reasons,
            "analysis_schema_version": PAIRED_ANALYSIS_SCHEMA_VERSION,
            "relative_improvement_version": RELATIVE_IMPROVEMENT_VERSION,
            "convergence_metric_version": CONVERGENCE_METRIC_VERSION,
            "source_identity": source_identity.to_dict(),
            "qualification_basis": (
                "case-wise exact-keyed paired feasible-objective evidence with rank-based effect "
                "sizes and Holm correction across comparator-by-case hypotheses + separate "
                "time-to-feasibility and post-feasibility AUC + configured independent AC-PF "
                "cross-validation + predeclared superiority/non-inferiority promotion gate"
            ),
        }
