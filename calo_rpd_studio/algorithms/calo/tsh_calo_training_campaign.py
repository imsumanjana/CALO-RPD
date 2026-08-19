"""Public independent TSH-CALO campaign API with canonical counted-work accounting.

The scientific campaign implementation remains in ``_tsh_calo_training_campaign_core``.  This
boundary preserves its public API while synchronizing durable status/events and augmenting manifests
with the exact policy-training/generalization-guard split.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from . import _tsh_calo_training_campaign_core as _core
from .tsh_calo_evaluation_accounting import (
    augment_extension_manifest,
    augment_extension_plan,
    augment_root_manifest,
    plan_training_evaluation_accounting,
    synchronize_training_progress,
)


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

_core_write_json = _core._write_json


def _load_adjacent_training_plan(path: Path):
    plan_path = path.parent / _core.IndependentTSHCALOTrainingCampaign.PLAN_FILE
    if path == plan_path or not plan_path.is_file():
        return None
    return _core.parse_tsh_calo_extension_plan(_core._read_json(plan_path))


def _write_json(path: str | Path, payload: dict) -> str:
    """Write JSON after adding canonical accounting to immutable output contracts."""

    target = Path(path)
    if not isinstance(payload, dict):
        raise TypeError("TSH-CALO JSON payload must be an object")
    plan = _load_adjacent_training_plan(target)
    if target.name == "extension_plan.json":
        if plan is None:
            raise ValueError("TSH-CALO extension plan lacks its immutable training plan")
        augment_extension_plan(plan, payload)
    elif target.name == _core.IndependentTSHCALOTrainingCampaign.MANIFEST_FILE:
        if plan is None:
            raise ValueError("TSH-CALO training manifest lacks its immutable training plan")
        if (
            payload.get("schema_version") == "tsh-calo-training-extension-manifest-v1"
            or payload.get("state") == "completed_unqualified_extension"
        ):
            augment_extension_manifest(plan, payload)
        else:
            augment_root_manifest(plan, payload)
    return _core_write_json(target, payload)


# Campaign-core methods resolve this module global at call time.  Use the accounting-aware writer
# without changing the scientific/execution plan or its hashes.
_core._write_json = _write_json

evaluate_generalization_bundle = _core.evaluate_generalization_bundle


class IndependentTSHCALOTrainingCampaign(_core.IndependentTSHCALOTrainingCampaign):
    """Campaign runner whose status, events, pause records, and manifests share one ledger."""

    _ACCOUNTING_EVENT_FIELDS = (
        "committed_training_candidate_evaluations",
        "total_training_candidate_evaluations",
        "committed_generalization_guard_candidate_evaluations",
        "total_generalization_guard_candidate_evaluations",
        "committed_total_candidate_evaluations",
        "total_counted_candidate_evaluations",
        "cumulative_training_candidate_evaluations",
        "cumulative_generalization_guard_candidate_evaluations",
        "cumulative_total_counted_candidate_evaluations",
        "progress_percent",
    )

    def _evaluate_generalization(
        self,
        trainer,
        training,
        *,
        final: bool,
        observation_index: int,
    ) -> dict:
        # Resolve the public module global so tests and controlled development fixtures can replace
        # the evaluator exactly as they did before the API boundary was introduced.
        config = self._generalization_guard_config()
        if config is None:
            raise RuntimeError("TSH-CALO generalization guard is not enabled for this plan")
        return evaluate_generalization_bundle(
            trainer,
            training,
            config,
            development_cases=self.plan.development_cases,
            population_size=self.plan.population_size,
            environment_template=asdict(self.plan.environment),
            problem_factory=lambda identity: self._build_problem(
                identity, device_hint=str(trainer.device)
            ),
            final=final,
            observation_index=observation_index,
            evaluation_backend="campaign_problem_factory",
        )

    def _write_status(self, status: dict) -> None:
        synchronize_training_progress(self.plan, status)
        _core.IndependentTSHCALOTrainingCampaign._write_status(self, status)

    def _record_event(
        self,
        status: dict,
        event: str,
        *,
        details: dict | None = None,
        notify_transition: bool = False,
    ) -> dict:
        progress = synchronize_training_progress(self.plan, status)
        enriched = dict(details or {})
        for key in self._ACCOUNTING_EVENT_FIELDS:
            if key in progress:
                enriched[key] = progress[key]
        extension = status.get("extension")
        if isinstance(extension, dict):
            for key in (
                "segment_training_candidate_evaluations",
                "segment_generalization_guard_candidate_evaluations",
                "segment_total_counted_candidate_evaluations",
                "prior_cumulative_training_candidate_evaluations",
                "prior_cumulative_generalization_guard_candidate_evaluations",
                "prior_cumulative_total_counted_candidate_evaluations",
            ):
                if key in extension:
                    enriched[key] = extension[key]
        enriched["legacy_candidate_evaluation_fields_are_training_only"] = True
        return _core.IndependentTSHCALOTrainingCampaign._record_event(
            self,
            status,
            event,
            details=enriched,
            notify_transition=notify_transition,
        )

    def _honor_pause_after_checkpoint(self, status: dict, progress: dict) -> None:
        del progress
        synchronized = synchronize_training_progress(self.plan, status)
        _core.IndependentTSHCALOTrainingCampaign._honor_pause_after_checkpoint(
            self,
            status,
            synchronized,
        )


__all__ = tuple(
    sorted(
        name
        for name in globals()
        if not name.startswith("_")
        and name not in {"Path", "annotations", "asdict"}
    )
)
