"""Public finite-extension API with the canonical counted-evaluation split."""

from __future__ import annotations

from . import _tsh_calo_training_extension_core as _core
from .tsh_calo_evaluation_accounting import augment_extension_summary


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

_core_extension_plan_summary = _core.extension_plan_summary


class IndependentTSHCALOTrainingExtension(_core.IndependentTSHCALOTrainingExtension):
    """Public extension runner; its campaign runner supplies synchronized counted-work status."""


def extension_plan_summary(plan, campaign_directory) -> dict:
    """Return a readiness summary that discloses training, guard, and total counted work."""

    payload = _core_extension_plan_summary(plan, campaign_directory)
    segment_number = int(payload["next_segment_number"])
    payload["segment_number"] = segment_number
    augment_extension_summary(plan, payload)
    payload.pop("segment_number", None)
    payload["summary_sha256"] = _core._canonical_sha256(
        {
            "plan": plan.execution_plan_sha256(),
            "parent": payload["parent_manifest_sha256"],
            "segment_training": payload["segment_training_candidate_evaluations"],
            "segment_guard": payload[
                "segment_generalization_guard_candidate_evaluations"
            ],
            "segment_total": payload["segment_total_counted_candidate_evaluations"],
            "prior_total": payload[
                "prior_cumulative_total_counted_candidate_evaluations"
            ],
        }
    )
    return payload


__all__ = tuple(
    sorted(
        name
        for name in globals()
        if not name.startswith("_") and name not in {"annotations"}
    )
)
