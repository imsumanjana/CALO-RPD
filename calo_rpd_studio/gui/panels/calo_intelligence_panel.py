"""Policy-library UI boundary for cumulative counted training work.

The complete lifecycle interface remains in ``_calo_intelligence_panel_core``.  This public
boundary changes only the evaluation-accounting presentation: the policy table consumes the shared
validated campaign/registry ledger, labels it accurately, persists exact manifest totals when a
completed campaign is imported, and refuses to display a partial extension guard history as a
cumulative total.
"""

from __future__ import annotations

from pathlib import Path

from . import _calo_intelligence_panel_core as _core


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def _valid_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _row_counted_evaluations(row, registry) -> int | None:
    if isinstance(row, dict):
        retained = row.get("total_counted_evaluations")
        if _valid_count(retained):
            return int(retained)
        policy = row.get("registered_policy")
    else:
        policy = row
    policy_id = str(getattr(policy, "id", "") or "")
    counter = getattr(registry, "counted_evaluation_count", None)
    if not policy_id or not callable(counter):
        return None
    try:
        value = counter(policy_id)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        return None
    return int(value) if _valid_count(value) else None


def _campaign_accounting(row: dict) -> dict | None:
    values = {
        "training_candidate_evaluations": row.get("training_evaluations"),
        "generalization_guard_candidate_evaluations": row.get(
            "generalization_guard_evaluations"
        ),
        "total_counted_candidate_evaluations": row.get("total_counted_evaluations"),
    }
    if not all(_valid_count(value) for value in values.values()):
        return None
    if values["total_counted_candidate_evaluations"] != (
        values["training_candidate_evaluations"]
        + values["generalization_guard_candidate_evaluations"]
    ):
        return None
    return {key: int(value) for key, value in values.items()}


class CALOIntelligencePanel(_core.CALOIntelligencePanel):
    """Render one exact cumulative total for policy training and learning-health checks."""

    def refresh_policy_library(self) -> None:
        super().refresh_policy_library()
        header = self.policy_table.horizontalHeaderItem(2)
        if header is not None:
            header.setText("Counted evaluations")
        registry = self.state.policy_registry
        tooltip = (
            "Cumulative exact candidate evaluations retained for policy training and "
            "development-only learning-health checks. Qualification and experiment evaluations "
            "are excluded. Not available means an imported extended artifact lacks its complete "
            "predecessor-manifest ledger."
        )
        for row_index, row in enumerate(self._policy_rows):
            item = self.policy_table.item(row_index, 2)
            if item is None:
                continue
            total = _row_counted_evaluations(row, registry)
            item.setText(self._training_evaluation_text(total))
            item.setToolTip(tooltip)

    def import_policy(self) -> None:
        completed = self._selected_completed_training()
        retained = dict(completed) if isinstance(completed, dict) else None
        super().import_policy()
        if retained is None:
            return
        accounting = _campaign_accounting(retained)
        candidate_text = str(retained.get("policy_candidate", "") or "").strip()
        binder = getattr(
            self.state.policy_registry,
            "bind_training_evaluation_accounting",
            None,
        )
        if accounting is None or not candidate_text or not callable(binder):
            return
        try:
            candidate = Path(candidate_text).expanduser().resolve()
            policy = next(
                item
                for item in self.state.policy_registry.list(include_archived=False)
                if Path(item.checkpoint_path).expanduser().resolve() == candidate
            )
            binder(policy.id, accounting)
        except (KeyError, OSError, RuntimeError, StopIteration, TypeError, ValueError) as exc:
            self.activity_message.emit(
                "WARNING",
                "The policy was imported, but its cumulative learning-health evaluation total "
                f"could not be retained: {str(exc) or type(exc).__name__}",
            )
            return
        self.refresh_policy_library()


_core.CALOIntelligencePanel = CALOIntelligencePanel


__all__ = tuple(
    sorted(
        name
        for name in globals()
        if not name.startswith("_") and name not in {"Path", "annotations"}
    )
)
