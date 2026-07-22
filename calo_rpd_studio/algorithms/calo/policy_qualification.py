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
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))

from calo_rpd_studio.experiments.seed_manager import SeedManager
from calo_rpd_studio.experiments.experiment_runner import run_single, build_problem
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.statistics.posthoc import holm_correction

HOLDOUT_CASES = {"case118", "case300"}


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

    def validate(self) -> None:
        if not self.cases:
            raise ValueError("Policy qualification requires at least one development case")
        if int(self.runs) < 2:
            raise ValueError("Policy qualification requires at least two paired runs")
        if int(self.max_evaluations) <= 0 or int(self.population_size) <= 1:
            raise ValueError("Qualification FE budget and population must be positive")
        if int(self.max_evaluations) % int(self.population_size) != 0:
            raise ValueError(
                "Policy qualification FE budget must be divisible by population size for exact FE parity"
            )
        if int(self.minimum_promotion_runs) < 5:
            raise ValueError("minimum_promotion_runs must be at least 5")
        if str(self.qualification_mode) not in {"superiority", "non_inferiority"}:
            raise ValueError("qualification_mode must be superiority or non_inferiority")
        if not 0.0 < float(self.statistical_alpha) < 1.0:
            raise ValueError("statistical_alpha must be between 0 and 1")
        if not 0.0 <= float(self.minimum_win_rate) <= 1.0:
            raise ValueError("minimum_win_rate must be between 0 and 1")
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
    """Feasibility-first AUC over the full FE horizon, including pre-feasible delay.

    Only ``best_feasible_objective_history`` is used.  Pre-feasible points receive a deterministic
    penalty derived from the observed feasible scale; raw infeasible objective values are never
    integrated.  A run that never becomes feasible has infinite AUC.
    """
    md = dict(getattr(result, "metadata", {}) or {})
    xs = np.asarray(md.get("convergence_evaluations", []), dtype=float)
    ys = np.asarray(md.get("best_feasible_objective_history", []), dtype=float)
    n = min(xs.size, ys.size)
    if n == 0:
        return float("inf")
    xs, ys = xs[:n], ys[:n]
    mask_x = np.isfinite(xs)
    xs, ys = xs[mask_x], ys[mask_x]
    if xs.size == 0:
        return float("inf")
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    finite = ys[np.isfinite(ys)]
    if finite.size == 0:
        return float("inf")
    scale = max(float(np.max(np.abs(finite))), 1.0)
    penalty = float(np.max(finite) + 0.10 * scale)
    filled = np.where(np.isfinite(ys), ys, penalty)
    horizon = max(float(getattr(result, "evaluations", 0) or 0), float(xs[-1]), 1.0)
    if xs[0] > 0.0:
        xs = np.r_[0.0, xs]
        filled = np.r_[penalty, filled]
    if xs[-1] < horizon:
        xs = np.r_[xs, horizon]
        filled = np.r_[filled, filled[-1]]
    return float(_trapezoid(filled, xs) / horizon)


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
        "feasible_probability": float(np.mean([row["feasible"] for row in records])) if records else 0.0,
        "independent_validation_probability": float(np.mean(independent)) if records else 0.0,
        "median_objective": float(np.median(finite)) if finite.size else float("nan"),
        "mean_objective": float(np.mean(finite)) if finite.size else float("nan"),
        "std_objective": float(np.std(finite, ddof=1)) if finite.size > 1 else (0.0 if finite.size == 1 else float("nan")),
        "iqr_objective": float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25)) if finite.size else float("nan"),
        "median_auc": float(np.median(auc_finite)) if auc_finite.size else float("inf"),
        "median_eval_to_feasible": float(np.median(etf_finite)) if etf_finite.size else float("nan"),
        "mean_runtime_seconds": float(np.mean(runtimes)) if runtimes.size else float("nan"),
    }


def _case_summaries(records: list[dict]) -> dict[str, dict]:
    cases = sorted({str(row["case"]) for row in records})
    return {case: _aggregate([row for row in records if str(row["case"]) == case]) for case in cases}


def _paired_evidence(candidate_rows: list[dict], comparator_rows: list[dict]) -> dict:
    comp_map = {(row["case"], int(row["run_index"])): row for row in comparator_rows}
    pairs = []
    for row in candidate_rows:
        other = comp_map.get((row["case"], int(row["run_index"])))
        if other is None:
            continue
        a, b = float(row["objective"]), float(other["objective"])
        if math.isfinite(a) and math.isfinite(b):
            pairs.append((a, b))
    if not pairs:
        return {"n_pairs": 0, "median_difference": float("nan"), "win_rate": float("nan"), "wilcoxon_p_two_sided": float("nan"), "holm_p": float("nan"), "rank_biserial": float("nan")}
    a = np.asarray([p[0] for p in pairs], float)
    b = np.asarray([p[1] for p in pairs], float)
    # Case-wise relative paired differences avoid pooling incomparable raw objective scales.
    d = (a - b) / np.maximum(np.abs(b), 1e-12)
    nonzero = d[np.abs(d) > 1e-15]
    pvalue = float("nan")
    if len(nonzero) >= 2:
        try:
            from scipy.stats import wilcoxon
            pvalue = float(wilcoxon(nonzero, alternative="two-sided", zero_method="wilcox").pvalue)
        except Exception:
            import math as _math
            k = min(int(np.sum(nonzero < 0)), int(np.sum(nonzero > 0)))
            n = len(nonzero)
            tail = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
            pvalue = float(min(1.0, 2.0 * tail))
    return {
        "n_pairs": int(len(pairs)),
        "median_difference": float(np.median(d)),
        "median_relative_difference": float(np.median(d)),
        "win_rate": float(np.mean(d < 0.0)),
        "wilcoxon_p_two_sided": pvalue,
        "holm_p": float("nan"),
        "rank_biserial": float((np.sum(nonzero < 0) - np.sum(nonzero > 0)) / max(len(nonzero), 1)),
    }


def _apply_holm(paired: dict) -> dict:
    keys, pvalues = [], []
    for key, item in paired.items():
        p = float(item.get("wilcoxon_p_two_sided", float("nan")))
        if math.isfinite(p):
            keys.append(key); pvalues.append(p)
    corrected = holm_correction(pvalues) if pvalues else []
    for key, p in zip(keys, corrected, strict=True):
        paired[key]["holm_p"] = float(p)
    return paired


def _grade(candidate, reference, no_ai, config, paired, case_summaries):
    reasons: list[str] = []
    if int(config.runs) < int(config.minimum_promotion_runs):
        reasons.append(
            f"formal promotion requires at least {config.minimum_promotion_runs} paired runs per case; this run is screening-only"
        )
    feasible = float(candidate["feasible_probability"])
    if feasible < float(config.minimum_feasible_probability):
        reasons.append(f"feasible probability {feasible:.3f} is below {config.minimum_feasible_probability:.3f}")
    if config.require_independent_validation:
        if float(candidate.get("independent_validation_probability", 0.0)) < 1.0:
            reasons.append("not every candidate run passed mandatory independent PYPOWER validation")
        if float(no_ai.get("independent_validation_probability", 0.0)) < 1.0:
            reasons.append("not every No-AI comparator run passed mandatory independent PYPOWER validation")
        if reference is not None and float(reference.get("independent_validation_probability", 0.0)) < 1.0:
            reasons.append("not every reference-policy run passed mandatory independent PYPOWER validation")

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
            relative_case_improvements.append((comp_med - cand_med) / max(abs(comp_med), 1e-12))

    evidence_rows = [item for item in paired.values()]
    favorable = bool(evidence_rows) and all(
        int(item.get("n_pairs", 0)) >= int(config.minimum_promotion_runs)
        and math.isfinite(float(item.get("median_difference", float("nan"))))
        and float(item["median_difference"]) <= 0.0
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
            and float(item["median_difference"]) <= float(config.non_inferiority_margin)
            for item in evidence_rows
        )
        if not noninferior:
            reasons.append("formal non-inferiority qualification failed the declared paired relative margin")
    passed = not reasons
    if not passed:
        return False, "U", 0.0, reasons

    median_rel = float(np.median(relative_case_improvements)) if relative_case_improvements else 0.0
    cand_auc = float(candidate.get("median_auc", float("inf")))
    comparator_aucs = [float(no_ai.get("median_auc", float("inf")))]
    if reference is not None:
        comparator_aucs.append(float(reference.get("median_auc", float("inf"))))
    finite_aucs = [v for v in comparator_aucs if math.isfinite(v)]
    auc_nonregression = (
        (not finite_aucs and math.isfinite(cand_auc))
        or (bool(finite_aucs) and math.isfinite(cand_auc) and cand_auc <= min(finite_aucs) * 1.01)
    )
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
            return {"available": False, "passed": False, "reason": f"independent_validator_unavailable:{exc}"}
        cross = validate_against_pypower(formulation_case, internal)
        checks.append({
            "scenario": scenario.name,
            "available": bool(cross.available),
            "passed": bool(cross.passed),
            "message": str(cross.message),
            "max_vm_difference": float(cross.max_vm_difference),
            "max_va_difference_deg": float(cross.max_va_difference_deg),
            "loss_difference_mw": float(cross.loss_difference_mw),
        })
    return {
        "available": bool(checks) and all(item["available"] for item in checks),
        "passed": bool(checks) and all(item["available"] and item["passed"] for item in checks),
        "scenarios": checks,
    }


class PolicyQualifier:
    def __init__(self, base_config, registry) -> None:
        self.base_config = base_config
        self.registry = registry

    def run(self, candidate_policy_id: str, *, reference_policy_id: str = "", config=None, progress_callback=None, cancel_callback=None) -> dict:
        qconfig = config or PolicyQualificationConfig()
        qconfig.validate()
        candidate = self.registry.get(candidate_policy_id)
        reference = self.registry.get(reference_policy_id) if reference_policy_id else None
        candidate_inspection = self.registry.inspect_checkpoint(candidate.checkpoint_path)
        if candidate_inspection["sha256"] != candidate.sha256:
            raise RuntimeError("Candidate policy checksum does not match the registered immutable artifact")
        if reference is not None:
            ref_inspection = self.registry.inspect_checkpoint(reference.checkpoint_path)
            if ref_inspection["sha256"] != reference.sha256:
                raise RuntimeError("Reference policy checksum does not match the registered immutable artifact")
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
                    cfg.case_name = str(case_name); cfg.algorithms = ["CALO"]; cfg.runs = 1
                    cfg.population_size = int(qconfig.population_size)
                    cfg.budget.max_evaluations = int(qconfig.max_evaluations)
                    cfg.max_iterations = max(int(cfg.max_iterations), int(qconfig.max_evaluations))
                    params = dict(cfg.algorithm_parameters.get("CALO", {}))
                    params.update({"strict_benchmark_mode": True, "use_historical_parameter_priors": False, "use_cross_algorithm_warm_start": False})
                    if policy is None:
                        params["use_ai"] = False; params.pop("policy_checkpoint", None); params.pop("policy_sha256", None); params["strict_policy_binding"] = False
                    else:
                        params.update({"use_ai": True, "policy_id": policy.id, "policy_checkpoint": policy.checkpoint_path, "policy_sha256": policy.sha256, "policy_state_schema_version": policy.state_schema_version, "policy_action_schema_version": policy.action_schema_version, "policy_architecture_version": policy.architecture_version, "policy_training_environment_version": policy.training_environment_version, "strict_policy_binding": True, "allow_unqualified_policy": True, "deterministic_policy": True})
                    cfg.algorithm_parameters["CALO"] = params
                    seeds = paired_seeds[run_index]
                    completed = run_single(cfg, "CALO", run_index, seeds)
                    result = completed.result
                    independent = _independent_validate_result(cfg, seeds, result)
                    records[label].append({
                        "case": str(case_name), "run_index": int(run_index),
                        "objective": _finite_objective(result), "feasible": bool(result.feasible),
                        "auc": _convergence_auc(result), "eval_to_feasible": _eval_to_feasible(result),
                        "runtime": float(result.runtime_seconds), "evaluations": int(result.evaluations),
                        "independent_validation_available": bool(independent["available"]),
                        "independent_validation_passed": bool(independent["passed"]),
                        "independent_validation": independent,
                    })
                    done += 1
                    if progress_callback:
                        progress_callback(int(100 * done / max(total, 1)), f"{done}/{total} · {case_name} · run {run_index + 1} · {label}")
        summaries = {name: _aggregate(rows) for name, rows in records.items()}
        case_summaries = {name: _case_summaries(rows) for name, rows in records.items()}
        paired = {"vs_no_ai": _paired_evidence(records["candidate"], records["no_ai"])}
        if "reference" in records:
            paired["vs_reference"] = _paired_evidence(records["candidate"], records["reference"])
        paired = _apply_holm(paired)
        passed, grade, score, reasons = _grade(summaries["candidate"], summaries.get("reference"), summaries["no_ai"], qconfig, paired, case_summaries)
        schema = candidate_inspection["schema"]
        return {
            "qualification_id": str(uuid.uuid4()), "candidate_policy_id": candidate.id,
            "reference_policy_id": reference.id if reference else "", "candidate_policy_sha256": candidate.sha256,
            "candidate_policy_schema": schema, "native_v41": bool(schema.get("native_v41", False)),
            "reference_policy_sha256": reference.sha256 if reference else "", "config": asdict(qconfig),
            "participants": summaries, "case_summaries": case_summaries, "records": records,
            "paired_evidence": paired, "passed": bool(passed), "grade": grade, "score": score, "reasons": reasons,
            "qualification_basis": "case-normalized paired feasible objective + feasibility-first AUC + mandatory independent PF validation + mandatory predeclared superiority/non-inferiority promotion gate with Holm correction",
        }
