"""Scientist-facing policy-training UI with integrated learning-health accounting.

The established rendered workflow remains in ``_independent_training_panel_core``.  This public
boundary applies the latest TSH-CALO learning-health guard to new GUI plans, reads the canonical
counted-work ledger, and presents one consistent total in readiness, live progress, completion, and
saved-campaign views.  Exact legacy resume/extension plans keep their stored scientific identity.
"""

from __future__ import annotations

import json
from pathlib import Path

from calo_rpd_studio.algorithms.calo.tsh_calo_evaluation_accounting import (
    plan_training_evaluation_accounting,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_generalization_guard import (
    TSHCALOGeneralizationGuardConfig,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_campaign import (
    TSHCALOTrainingCampaignPlan,
)

from . import _independent_training_panel_core as _core


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def _guarded_new_plan_payload(payload: dict) -> dict:
    """Return a validated new-plan payload with a visible default guard when none is declared."""

    values = dict(payload or {})
    configured = values.get("generalization_guard")
    if not isinstance(configured, dict) or not configured:
        values["generalization_guard"] = TSHCALOGeneralizationGuardConfig().to_dict()
    plan = TSHCALOTrainingCampaignPlan.from_dict(values)
    return plan.to_dict()


def _plan_accounting_text(payload: dict | None) -> str:
    """Return ordinary-language readiness text from the canonical plan ledger."""

    if not isinstance(payload, dict):
        return "Learning-health check: pending plan validation."
    try:
        plan = TSHCALOTrainingCampaignPlan.from_dict(payload)
        accounting = plan_training_evaluation_accounting(plan)
    except (KeyError, TypeError, ValueError):
        return "Learning-health check: review the selected training inputs."
    guard = plan.generalization_guard
    enabled = guard is not None and bool(guard.enabled)
    if enabled:
        return (
            "Learning-health check: enabled · counted candidate evaluations: "
            f"{accounting.training_candidate_evaluations} training + "
            f"{accounting.generalization_guard_candidate_evaluations} learning-health = "
            f"{accounting.total_counted_candidate_evaluations} total."
        )
    return (
        "Learning-health check: not included in this retained plan · counted candidate "
        f"evaluations: {accounting.total_counted_candidate_evaluations} total."
    )


def _display_accounting_event(event: dict) -> dict:
    """Map compatibility fields to the explicit counted totals used by rendered progress text."""

    displayed = dict(event or {})
    accounting = displayed.get("evaluation_accounting")
    if not isinstance(accounting, dict):
        accounting = {}
    total = displayed.get(
        "total_counted_candidate_evaluations",
        accounting.get("total_counted_candidate_evaluations"),
    )
    committed = displayed.get("committed_total_candidate_evaluations")
    cumulative = displayed.get("cumulative_total_counted_candidate_evaluations")
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        displayed["total_candidate_evaluations"] = int(total)
    if isinstance(committed, int) and not isinstance(committed, bool) and committed >= 0:
        displayed["committed_candidate_evaluations"] = int(committed)
    if isinstance(cumulative, int) and not isinstance(cumulative, bool) and cumulative >= 0:
        displayed["cumulative_candidate_evaluations"] = int(cumulative)
    return displayed


def _latest_manifest_accounting(campaign_directory: str | Path) -> dict:
    """Read the newest completed manifest ledger without accepting an incomplete child segment."""

    root = Path(campaign_directory).expanduser()
    candidates = [root]
    candidates.extend(sorted((root / "extensions").glob("segment-*")))
    selected: dict = {}
    for directory in candidates:
        status_path = directory / _core.TrainingModelLibrary.STATUS_FILE
        manifest_path = directory / _core.TrainingModelLibrary.MANIFEST_FILE
        if not status_path.is_file() or not manifest_path.is_file():
            continue
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(status, dict) or status.get("state") != "completed":
            continue
        if not isinstance(manifest, dict):
            continue
        accounting = manifest.get("evaluation_accounting")
        if not isinstance(accounting, dict):
            continue
        cumulative = accounting.get("cumulative")
        selected = dict(cumulative) if isinstance(cumulative, dict) else dict(accounting)
    return selected


def _nonnegative_int(value):
    return (
        int(value)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        else None
    )


class TrainingLaunchModel(_core.TrainingLaunchModel):
    """Build new scientist plans with the guard while preserving exact saved-plan identity."""

    def _apply_new_plan_guard(self) -> None:
        if self.plan_payload is None:
            return
        try:
            self.plan_payload = _guarded_new_plan_payload(self.plan_payload)
        except (KeyError, TypeError, ValueError) as exc:
            self.plan_payload = None
            self.plan_error = str(exc)

    def load_plan(self, *, preserve_identity: bool = False) -> None:
        previous = self.blockSignals(True)
        try:
            super().load_plan(preserve_identity=preserve_identity)
        finally:
            self.blockSignals(previous)
        if not preserve_identity and self.plan_payload is not None:
            self._apply_new_plan_guard()
        self.changed.emit(dict(self.values))

    def create_plan(
        self,
        *,
        campaign_id: str,
        development_cases: list[str],
        member_count: int,
        master_seed: int,
        population_size: int,
        max_evaluations: int,
        requested_device: str,
        allow_cpu_fallback: bool,
        training: dict,
    ) -> None:
        previous = self.blockSignals(True)
        try:
            super().create_plan(
                campaign_id=campaign_id,
                development_cases=development_cases,
                member_count=member_count,
                master_seed=master_seed,
                population_size=population_size,
                max_evaluations=max_evaluations,
                requested_device=requested_device,
                allow_cpu_fallback=allow_cpu_fallback,
                training=training,
            )
        finally:
            self.blockSignals(previous)
        self._apply_new_plan_guard()
        self.changed.emit(dict(self.values))


class TrainingModelLibrary(_core.TrainingModelLibrary):
    """Expose training-only, learning-health, and total counted work for every saved campaign."""

    def saved_campaigns(self) -> tuple[dict, ...]:
        rows = []
        for source in super().saved_campaigns():
            row = dict(source)
            progress = dict(row.get("progress", {}) or {})
            accounting = (
                _latest_manifest_accounting(row["directory"])
                if row.get("state") == "completed"
                else {}
            )
            training = _nonnegative_int(
                accounting.get(
                    "training_candidate_evaluations",
                    progress.get(
                        "cumulative_training_candidate_evaluations",
                        progress.get("total_training_candidate_evaluations"),
                    ),
                )
            )
            guard = _nonnegative_int(
                accounting.get(
                    "generalization_guard_candidate_evaluations",
                    progress.get(
                        "cumulative_generalization_guard_candidate_evaluations",
                        progress.get("total_generalization_guard_candidate_evaluations"),
                    ),
                )
            )
            total = _nonnegative_int(
                accounting.get(
                    "total_counted_candidate_evaluations",
                    progress.get(
                        "cumulative_total_counted_candidate_evaluations",
                        progress.get("total_counted_candidate_evaluations"),
                    ),
                )
            )
            if training is not None and row.get("training_evaluations") is None:
                row["training_evaluations"] = training
            row["generalization_guard_evaluations"] = guard
            row["total_counted_evaluations"] = total
            rows.append(row)
        return tuple(rows)


class IndependentTrainingPanel(_core.IndependentTrainingPanel):
    """Render the same canonical accounting used by the campaign and extension contracts."""

    def _configuration_changed(self, values: dict) -> None:
        super()._configuration_changed(values)
        self.plan_summary.setText(
            self.plan_summary.text() + "\n" + _plan_accounting_text(self.model.plan_payload)
        )

    def _refresh_preview(self) -> None:
        super()._refresh_preview()
        self.command_preview.setPlainText(
            self.command_preview.toPlainText()
            + "\n\n"
            + _plan_accounting_text(self.model.plan_payload)
        )

    def _apply_training_progress(self, event: dict) -> None:
        super()._apply_training_progress(_display_accounting_event(event))


# Core helpers and any classes retained below the public boundary resolve these names dynamically.
_core.TrainingLaunchModel = TrainingLaunchModel
_core.TrainingModelLibrary = TrainingModelLibrary
_core.IndependentTrainingPanel = IndependentTrainingPanel


__all__ = tuple(
    sorted(
        name
        for name in globals()
        if not name.startswith("_")
        and name
        not in {
            "Path",
            "TSHCALOGeneralizationGuardConfig",
            "TSHCALOTrainingCampaignPlan",
            "annotations",
            "json",
        }
    )
)
