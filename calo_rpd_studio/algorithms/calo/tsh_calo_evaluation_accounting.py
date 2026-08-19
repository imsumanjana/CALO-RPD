"""Canonical exact-evaluation accounting for finite TSH-CALO training segments.

Legacy ``candidate_evaluations`` fields describe policy-training episode work only.  The helpers in
this module retain those fields for compatibility while adding an explicit split for policy
training, development-only generalization-guard work, and the total counted evaluator work.  The
same formulas are used by campaign status, extension lineage, manifests, CLI output, and the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping, Protocol


class _EpisodeCollection(Protocol):
    episodes: tuple


class _GuardConfig(Protocol):
    enabled: bool

    def validation_evaluations_per_case(self, population_size: int) -> int: ...


class _TrainingPlan(Protocol):
    development_cases: tuple[str, ...]
    members: tuple[_EpisodeCollection, ...]
    population_size: int
    max_evaluations: int
    generalization_guard: _GuardConfig | None


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingEvaluationAccounting:
    """Exact evaluator-work totals for one fresh or extension training segment."""

    training_candidate_evaluations: int
    generalization_guard_candidate_evaluations: int
    total_counted_candidate_evaluations: int

    def validate(self) -> None:
        for label, value in (
            ("training", self.training_candidate_evaluations),
            ("generalization guard", self.generalization_guard_candidate_evaluations),
            ("total counted", self.total_counted_candidate_evaluations),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"TSH-CALO {label} candidate-evaluation count is invalid")
        if self.total_counted_candidate_evaluations != (
            self.training_candidate_evaluations
            + self.generalization_guard_candidate_evaluations
        ):
            raise ValueError("TSH-CALO total counted candidate evaluations are inconsistent")

    def to_dict(self) -> dict:
        self.validate()
        return {
            "training_candidate_evaluations": self.training_candidate_evaluations,
            "generalization_guard_candidate_evaluations": (
                self.generalization_guard_candidate_evaluations
            ),
            "total_counted_candidate_evaluations": self.total_counted_candidate_evaluations,
        }


def plan_training_evaluation_accounting(
    plan: _TrainingPlan,
) -> TSHCALOTrainingEvaluationAccounting:
    """Return the exact planned split for one fresh or extension segment.

    A guarded member evaluates two frozen baselines, one monitor after every training episode, and
    one final audit.  Each bundle covers every declared development case for the configured number
    of validation batches and population members.
    """

    training = int(
        sum(len(member.episodes) for member in plan.members) * int(plan.max_evaluations)
    )
    guard = plan.generalization_guard
    guard_evaluations = 0
    if guard is not None and bool(guard.enabled):
        per_bundle = int(
            len(plan.development_cases)
            * guard.validation_evaluations_per_case(int(plan.population_size))
        )
        guard_evaluations = int(
            sum((len(member.episodes) + 3) * per_bundle for member in plan.members)
        )
    result = TSHCALOTrainingEvaluationAccounting(
        training_candidate_evaluations=training,
        generalization_guard_candidate_evaluations=guard_evaluations,
        total_counted_candidate_evaluations=training + guard_evaluations,
    )
    result.validate()
    return result


def _count_from_evidence(value, *, label: str) -> int:
    if value is None:
        return 0
    if not isinstance(value, Mapping):
        raise ValueError(f"TSH-CALO {label} evidence is invalid")
    count = value.get("candidate_evaluations", 0)
    if not isinstance(count, int) or isinstance(count, bool) or count < 0:
        raise ValueError(f"TSH-CALO {label} candidate-evaluation count is invalid")
    return int(count)


def committed_generalization_guard_candidate_evaluations(status: Mapping) -> int:
    """Count only guard evidence already retained in durable campaign status."""

    root = status.get("generalization_guard")
    if root is None:
        return 0
    if not isinstance(root, Mapping):
        raise ValueError("TSH-CALO generalization-guard status is invalid")
    members = root.get("members", {})
    if not isinstance(members, Mapping):
        raise ValueError("TSH-CALO generalization-guard member status is invalid")
    total = 0
    for member_key, slot in members.items():
        if not isinstance(slot, Mapping):
            raise ValueError(
                f"TSH-CALO generalization-guard status for member {member_key!s} is invalid"
            )
        total += _count_from_evidence(
            slot.get("baseline_monitor_evidence"), label="baseline monitor"
        )
        total += _count_from_evidence(
            slot.get("baseline_final_evidence"), label="baseline final audit"
        )
        monitors = slot.get("monitor_evidence", [])
        if not isinstance(monitors, list):
            raise ValueError("TSH-CALO generalization monitor status is invalid")
        total += sum(
            _count_from_evidence(item, label="monitor") for item in monitors
        )
        result = slot.get("result")
        if result is not None:
            if not isinstance(result, Mapping):
                raise ValueError("TSH-CALO generalization result status is invalid")
            total += _count_from_evidence(
                result.get("final_evidence"), label="final audit"
            )
    return int(total)


def synchronize_training_progress(plan: _TrainingPlan, status: MutableMapping) -> dict:
    """Mutate one status record so every progress surface uses the canonical split."""

    accounting = plan_training_evaluation_accounting(plan)
    progress = status.get("progress")
    if progress is None:
        progress = {}
        status["progress"] = progress
    if not isinstance(progress, MutableMapping):
        raise ValueError("TSH-CALO campaign progress status is invalid")

    committed_training = progress.get(
        "committed_training_candidate_evaluations",
        progress.get("committed_candidate_evaluations", 0),
    )
    if (
        not isinstance(committed_training, int)
        or isinstance(committed_training, bool)
        or not 0 <= committed_training <= accounting.training_candidate_evaluations
    ):
        raise ValueError("TSH-CALO committed training evaluation accounting is invalid")
    committed_guard = committed_generalization_guard_candidate_evaluations(status)
    if committed_guard > accounting.generalization_guard_candidate_evaluations:
        raise ValueError("TSH-CALO committed generalization evaluation accounting exceeds plan")
    if (
        str(status.get("state", "")) == "completed"
        and committed_guard != accounting.generalization_guard_candidate_evaluations
    ):
        raise ValueError("Completed TSH-CALO guard evaluation accounting is incomplete")

    committed_total = int(committed_training + committed_guard)
    if accounting.total_counted_candidate_evaluations:
        percent = min(
            99,
            int(
                committed_total
                * 100
                / accounting.total_counted_candidate_evaluations
            ),
        )
    else:
        percent = 0
    if str(status.get("state", "")) == "completed":
        percent = 100

    # Compatibility fields retain their original policy-training-only meaning.
    progress["committed_candidate_evaluations"] = int(committed_training)
    progress["total_candidate_evaluations"] = accounting.training_candidate_evaluations
    progress.update(
        {
            "committed_training_candidate_evaluations": int(committed_training),
            "total_training_candidate_evaluations": (
                accounting.training_candidate_evaluations
            ),
            "committed_generalization_guard_candidate_evaluations": committed_guard,
            "total_generalization_guard_candidate_evaluations": (
                accounting.generalization_guard_candidate_evaluations
            ),
            "committed_total_candidate_evaluations": committed_total,
            "total_counted_candidate_evaluations": (
                accounting.total_counted_candidate_evaluations
            ),
            "progress_percent": percent,
        }
    )

    extension = status.get("extension")
    if isinstance(extension, Mapping):
        segment_number = int(extension.get("segment_number", 0))
        if segment_number < 1:
            raise ValueError("TSH-CALO extension segment number is invalid")
        prior_training = int(
            extension.get(
                "prior_cumulative_training_candidate_evaluations",
                extension.get("prior_cumulative_candidate_evaluations", 0),
            )
        )
        expected_prior_guard = int(
            segment_number * accounting.generalization_guard_candidate_evaluations
        )
        prior_guard = int(
            extension.get(
                "prior_cumulative_generalization_guard_candidate_evaluations",
                expected_prior_guard,
            )
        )
        if prior_guard != expected_prior_guard:
            raise ValueError("TSH-CALO extension prior guard accounting changed")
        prior_total = int(
            extension.get(
                "prior_cumulative_total_counted_candidate_evaluations",
                prior_training + prior_guard,
            )
        )
        if prior_total != prior_training + prior_guard:
            raise ValueError("TSH-CALO extension prior total accounting is inconsistent")
        progress.update(
            {
                "cumulative_candidate_evaluations": prior_training + int(committed_training),
                "cumulative_training_candidate_evaluations": (
                    prior_training + int(committed_training)
                ),
                "cumulative_generalization_guard_candidate_evaluations": (
                    prior_guard + committed_guard
                ),
                "cumulative_total_counted_candidate_evaluations": (
                    prior_total + committed_total
                ),
            }
        )
    return dict(progress)


def augment_root_manifest(plan: _TrainingPlan, payload: MutableMapping) -> None:
    accounting = plan_training_evaluation_accounting(plan)
    payload["evaluation_accounting"] = accounting.to_dict()
    contract = dict(payload.get("extension_contract", {}) or {})
    contract.update(
        {
            "segment_evaluations": accounting.training_candidate_evaluations,
            "segment_training_candidate_evaluations": (
                accounting.training_candidate_evaluations
            ),
            "segment_generalization_guard_candidate_evaluations": (
                accounting.generalization_guard_candidate_evaluations
            ),
            "segment_total_counted_candidate_evaluations": (
                accounting.total_counted_candidate_evaluations
            ),
            "legacy_candidate_evaluation_fields_are_training_only": True,
        }
    )
    payload["extension_contract"] = contract


def augment_extension_plan(plan: _TrainingPlan, payload: MutableMapping) -> None:
    accounting = plan_training_evaluation_accounting(plan)
    segment_number = int(payload.get("segment_number", 0))
    if segment_number < 1:
        raise ValueError("TSH-CALO extension segment number is invalid")
    prior_training = int(payload.get("prior_cumulative_candidate_evaluations", 0))
    prior_guard = int(
        segment_number * accounting.generalization_guard_candidate_evaluations
    )
    prior_total = prior_training + prior_guard
    payload.update(
        {
            "segment_candidate_evaluations": accounting.training_candidate_evaluations,
            "segment_training_candidate_evaluations": (
                accounting.training_candidate_evaluations
            ),
            "segment_generalization_guard_candidate_evaluations": (
                accounting.generalization_guard_candidate_evaluations
            ),
            "segment_total_counted_candidate_evaluations": (
                accounting.total_counted_candidate_evaluations
            ),
            "prior_cumulative_training_candidate_evaluations": prior_training,
            "prior_cumulative_generalization_guard_candidate_evaluations": prior_guard,
            "prior_cumulative_total_counted_candidate_evaluations": prior_total,
            "next_cumulative_training_candidate_evaluations": (
                prior_training + accounting.training_candidate_evaluations
            ),
            "next_cumulative_generalization_guard_candidate_evaluations": (
                prior_guard + accounting.generalization_guard_candidate_evaluations
            ),
            "next_cumulative_total_counted_candidate_evaluations": (
                prior_total + accounting.total_counted_candidate_evaluations
            ),
            "legacy_candidate_evaluation_fields_are_training_only": True,
        }
    )


def augment_extension_manifest(plan: _TrainingPlan, payload: MutableMapping) -> None:
    accounting = plan_training_evaluation_accounting(plan)
    completed_extensions = int(payload.get("completed_extension_count", 0))
    if completed_extensions < 1:
        raise ValueError("TSH-CALO completed extension count is invalid")
    segment_count = completed_extensions + 1  # root campaign plus completed extensions
    cumulative_training = int(payload.get("cumulative_candidate_evaluations", 0))
    cumulative_guard = int(
        segment_count * accounting.generalization_guard_candidate_evaluations
    )
    cumulative_total = cumulative_training + cumulative_guard
    payload.update(
        {
            "segment_candidate_evaluations": accounting.training_candidate_evaluations,
            "segment_training_candidate_evaluations": (
                accounting.training_candidate_evaluations
            ),
            "segment_generalization_guard_candidate_evaluations": (
                accounting.generalization_guard_candidate_evaluations
            ),
            "segment_total_counted_candidate_evaluations": (
                accounting.total_counted_candidate_evaluations
            ),
            "cumulative_training_candidate_evaluations": cumulative_training,
            "cumulative_generalization_guard_candidate_evaluations": cumulative_guard,
            "cumulative_total_counted_candidate_evaluations": cumulative_total,
            "evaluation_accounting": {
                "segment": accounting.to_dict(),
                "cumulative": {
                    "training_candidate_evaluations": cumulative_training,
                    "generalization_guard_candidate_evaluations": cumulative_guard,
                    "total_counted_candidate_evaluations": cumulative_total,
                },
            },
            "legacy_candidate_evaluation_fields_are_training_only": True,
        }
    )
    contract = dict(payload.get("extension_contract", {}) or {})
    contract.update(
        {
            "segment_training_candidate_evaluations": (
                accounting.training_candidate_evaluations
            ),
            "segment_generalization_guard_candidate_evaluations": (
                accounting.generalization_guard_candidate_evaluations
            ),
            "segment_total_counted_candidate_evaluations": (
                accounting.total_counted_candidate_evaluations
            ),
            "legacy_candidate_evaluation_fields_are_training_only": True,
        }
    )
    payload["extension_contract"] = contract


def augment_extension_summary(plan: _TrainingPlan, payload: MutableMapping) -> None:
    augment_extension_plan(plan, payload)
