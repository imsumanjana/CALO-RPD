"""Paired CALO policy qualification against a reference policy and no-AI CALO.

Qualification is intentionally separated from policy training.  A checkpoint is promoted only on
optimization outcomes under identical seeds and FE budgets; PPO loss/return alone is never used as
proof that a policy is superior.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import uuid

import numpy as np

from calo_rpd_studio.experiments.seed_manager import SeedManager
from calo_rpd_studio.experiments.experiment_runner import run_single


HOLDOUT_CASES = {"case118", "case300"}


@dataclass(slots=True)
class PolicyQualificationConfig:
    cases: tuple[str, ...] = ("case30", "case57")
    runs: int = 5
    max_evaluations: int = 1000
    population_size: int = 40
    master_seed: int = 20260410
    allow_holdout_cases: bool = False
    objective_regression_tolerance: float = 0.01
    minimum_feasible_probability: float = 0.90

    def validate(self) -> None:
        if not self.cases:
            raise ValueError("Policy qualification requires at least one development case")
        if int(self.runs) < 2:
            raise ValueError("Policy qualification requires at least two paired runs")
        if int(self.max_evaluations) <= 0 or int(self.population_size) <= 1:
            raise ValueError("Qualification FE budget and population must be positive")
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
    history = list(result.convergence_history or [])
    xs, ys = [], []
    for point in history:
        if isinstance(point, dict):
            x = point.get("evaluations", point.get("evaluation", point.get("fe", None)))
            y = point.get(
                "best_feasible_objective", point.get("best_objective", point.get("objective", None))
            )
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        try:
            x = float(x)
            y = float(y)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            xs.append(x)
            ys.append(y)
    if len(xs) < 2:
        return float("nan")
    order = np.argsort(xs)
    x = np.asarray(xs, float)[order]
    y = np.asarray(ys, float)[order]
    span = max(float(x[-1] - x[0]), 1.0)
    return float(np.trapezoid(y, x) / span)


def _eval_to_feasible(result) -> float:
    history = list(result.convergence_history or [])
    for point in history:
        if isinstance(point, dict):
            value = point.get("best_feasible_objective", None)
            evaluations = point.get("evaluations", point.get("evaluation", point.get("fe", None)))
            try:
                if value is not None and math.isfinite(float(value)):
                    return float(evaluations)
            except (TypeError, ValueError):
                pass
    return 0.0 if result.feasible else float("nan")


def _aggregate(records: list[dict]) -> dict:
    objectives = np.asarray([row["objective"] for row in records], float)
    finite = objectives[np.isfinite(objectives)]
    aucs = np.asarray([row["auc"] for row in records], float)
    auc_finite = aucs[np.isfinite(aucs)]
    etf = np.asarray([row["eval_to_feasible"] for row in records], float)
    etf_finite = etf[np.isfinite(etf)]
    runtimes = np.asarray([row["runtime"] for row in records], float)
    return {
        "n": len(records),
        "feasible_probability": float(np.mean([row["feasible"] for row in records]))
        if records
        else 0.0,
        "median_objective": float(np.median(finite)) if finite.size else float("nan"),
        "mean_objective": float(np.mean(finite)) if finite.size else float("nan"),
        "std_objective": float(np.std(finite, ddof=1))
        if finite.size > 1
        else 0.0
        if finite.size == 1
        else float("nan"),
        "iqr_objective": float(np.quantile(finite, 0.75) - np.quantile(finite, 0.25))
        if finite.size
        else float("nan"),
        "median_auc": float(np.median(auc_finite)) if auc_finite.size else float("nan"),
        "median_eval_to_feasible": float(np.median(etf_finite))
        if etf_finite.size
        else float("nan"),
        "mean_runtime_seconds": float(np.mean(runtimes)) if runtimes.size else float("nan"),
    }


def _paired_evidence(candidate_rows: list[dict], comparator_rows: list[dict]) -> dict:
    """Paired feasible-objective evidence without inventing values for infeasible runs."""
    comp_map = {(row["case"], int(row["run_index"])): row for row in comparator_rows}
    pairs: list[tuple[float, float]] = []
    for row in candidate_rows:
        other = comp_map.get((row["case"], int(row["run_index"])))
        if other is None:
            continue
        a = float(row["objective"])
        b = float(other["objective"])
        if math.isfinite(a) and math.isfinite(b):
            pairs.append((a, b))
    if not pairs:
        return {
            "n_pairs": 0,
            "median_difference": float("nan"),
            "win_rate": float("nan"),
            "wilcoxon_p_two_sided": float("nan"),
            "rank_biserial": float("nan"),
        }
    a = np.asarray([item[0] for item in pairs], float)
    b = np.asarray([item[1] for item in pairs], float)
    d = a - b  # lower objective is better, so negative favors candidate
    wins = float(np.mean(d < 0.0))
    nonzero = d[np.abs(d) > 1e-15]
    rank_biserial = float((np.sum(nonzero < 0) - np.sum(nonzero > 0)) / max(len(nonzero), 1))
    pvalue = float("nan")
    if len(nonzero) >= 2:
        try:
            from scipy.stats import wilcoxon

            pvalue = float(wilcoxon(a, b, alternative="two-sided", zero_method="wilcox").pvalue)
        except Exception:
            # Exact sign-test fallback is conservative and dependency-free.
            import math as _math

            k = min(int(np.sum(nonzero < 0)), int(np.sum(nonzero > 0)))
            n = len(nonzero)
            tail = sum(_math.comb(n, i) for i in range(k + 1)) / (2**n)
            pvalue = float(min(1.0, 2.0 * tail))
    return {
        "n_pairs": int(len(pairs)),
        "median_difference": float(np.median(d)),
        "win_rate": wins,
        "wilcoxon_p_two_sided": pvalue,
        "rank_biserial": rank_biserial,
    }


def _grade(
    candidate: dict,
    reference: dict | None,
    no_ai: dict,
    config: PolicyQualificationConfig,
    paired: dict | None = None,
) -> tuple[bool, str, float, list[str]]:
    reasons: list[str] = []
    feasible = float(candidate["feasible_probability"])
    if feasible < float(config.minimum_feasible_probability):
        reasons.append(
            f"feasible probability {feasible:.3f} is below {config.minimum_feasible_probability:.3f}"
        )
    cand = float(candidate["median_objective"])
    baseline = float(no_ai["median_objective"])
    if not math.isfinite(cand):
        reasons.append("candidate has no finite feasible median objective")
    if math.isfinite(cand) and math.isfinite(baseline):
        allowed = (
            baseline * (1.0 + float(config.objective_regression_tolerance))
            if baseline >= 0
            else baseline + abs(baseline) * float(config.objective_regression_tolerance)
        )
        if cand > allowed:
            reasons.append("candidate materially regresses versus No-AI CALO")
    passed = not reasons
    if not passed:
        return False, "U", 0.0, reasons

    comparators = [no_ai]
    if reference is not None and math.isfinite(
        float(reference.get("median_objective", float("nan")))
    ):
        comparators.append(reference)
    best_comp = min(
        float(item["median_objective"])
        for item in comparators
        if math.isfinite(float(item["median_objective"]))
    )
    rel = (best_comp - cand) / max(abs(best_comp), 1e-12) if math.isfinite(cand) else -1.0
    cand_auc = float(candidate.get("median_auc", float("nan")))
    comp_aucs = [float(item.get("median_auc", float("nan"))) for item in comparators]
    finite_aucs = [value for value in comp_aucs if math.isfinite(value)]
    auc_gain = (
        ((min(finite_aucs) - cand_auc) / max(abs(min(finite_aucs)), 1e-12))
        if finite_aucs and math.isfinite(cand_auc)
        else 0.0
    )
    paired = dict(paired or {})
    evidence_rows = [dict(paired.get("vs_no_ai") or {})] if "vs_no_ai" in paired else [paired]
    if reference is not None and "vs_reference" in paired:
        evidence_rows.append(dict(paired.get("vs_reference") or {}))
    statistically_supported = bool(evidence_rows) and all(
        int(item.get("n_pairs", 0) or 0) >= 5
        and math.isfinite(float(item.get("wilcoxon_p_two_sided", float("nan"))))
        and float(item.get("wilcoxon_p_two_sided")) <= 0.05
        for item in evidence_rows
    )
    if rel > 0.01 and auc_gain >= -0.01 and feasible >= 0.99 and statistically_supported:
        grade = "A+"
    elif rel >= -0.002 and feasible >= 0.99:
        grade = "A"
    elif rel >= -float(config.objective_regression_tolerance):
        grade = "B+"
    else:
        grade = "B"
    # Ordinal display index only: it never trades objective quality against runtime/AUC and is
    # not used to decide scientific superiority. Actual metrics remain visible side-by-side.
    score = {"B": 1.0, "B+": 2.0, "A": 3.0, "A+": 4.0}[grade]
    return True, grade, score, reasons


class PolicyQualifier:
    def __init__(self, base_config, registry) -> None:
        self.base_config = base_config
        self.registry = registry

    def run(
        self,
        candidate_policy_id: str,
        *,
        reference_policy_id: str = "",
        config: PolicyQualificationConfig | None = None,
        progress_callback=None,
        cancel_callback=None,
    ) -> dict:
        qconfig = config or PolicyQualificationConfig()
        qconfig.validate()
        candidate = self.registry.get(candidate_policy_id)
        reference = self.registry.get(reference_policy_id) if reference_policy_id else None
        candidate_inspection = self.registry.inspect_checkpoint(candidate.checkpoint_path)
        if candidate_inspection["sha256"] != candidate.sha256:
            raise RuntimeError(
                "Candidate policy checksum does not match the registered immutable artifact"
            )
        reference_inspection = None
        if reference is not None:
            reference_inspection = self.registry.inspect_checkpoint(reference.checkpoint_path)
            if reference_inspection["sha256"] != reference.sha256:
                raise RuntimeError(
                    "Reference policy checksum does not match the registered immutable artifact"
                )
        participants = [("candidate", candidate), ("no_ai", None)]
        if reference is not None and reference.id != candidate.id:
            participants.insert(1, ("reference", reference))
        total = len(qconfig.cases) * qconfig.runs * len(participants)
        done = 0
        records: dict[str, list[dict]] = {name: [] for name, _ in participants}
        seed_manager = SeedManager(qconfig.master_seed)
        paired = seed_manager.generate(qconfig.runs)
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
                    params["strict_benchmark_mode"] = True
                    params["use_historical_parameter_priors"] = False
                    params["use_cross_algorithm_warm_start"] = False
                    if policy is None:
                        params["use_ai"] = False
                        params.pop("policy_checkpoint", None)
                        params.pop("policy_sha256", None)
                        params["strict_policy_binding"] = False
                    else:
                        params["use_ai"] = True
                        params.update(
                            {
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
                    completed = run_single(cfg, "CALO", run_index, paired[run_index])
                    result = completed.result
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
                        }
                    )
                    done += 1
                    if progress_callback:
                        progress_callback(
                            int(100 * done / max(total, 1)),
                            f"{done}/{total} · {case_name} · run {run_index + 1} · {label}",
                        )
        summaries = {name: _aggregate(rows) for name, rows in records.items()}
        paired = {"vs_no_ai": _paired_evidence(records["candidate"], records["no_ai"])}
        if "reference" in records:
            paired["vs_reference"] = _paired_evidence(records["candidate"], records["reference"])
        passed, grade, score, reasons = _grade(
            summaries["candidate"], summaries.get("reference"), summaries["no_ai"], qconfig, paired
        )
        schema = candidate_inspection["schema"]
        return {
            "qualification_id": str(uuid.uuid4()),
            "candidate_policy_id": candidate.id,
            "reference_policy_id": reference.id if reference else "",
            "candidate_policy_sha256": candidate.sha256,
            "candidate_policy_schema": schema,
            "native_v41": bool(schema.get("native_v41", False)),
            "reference_policy_sha256": reference.sha256 if reference else "",
            "config": asdict(qconfig),
            "participants": summaries,
            "records": records,
            "paired_evidence": paired,
            "passed": bool(passed),
            "grade": grade,
            "score": score,
            "reasons": reasons,
        }
