"""Hide obsolete registered policy artifacts from the ordinary Policy library view."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import QTableWidgetItem

from calo_rpd_studio.ai.model_io import checkpoint_sha256
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import TSH_CALO_ALGORITHM_ID

from .calo_intelligence_obsolete_models import ObsoleteAwareCALOIntelligencePanel


class CompleteObsoleteAwareCALOIntelligencePanel(ObsoleteAwareCALOIntelligencePanel):
    """Classify both managed training folders and registered unusable models as obsolete."""

    def __init__(self, *args, **kwargs) -> None:
        self._registered_obsolete_cache: dict[tuple, tuple[str, ...]] = {}
        super().__init__(*args, **kwargs)

    def _registered_obsolete_issue(self, policy) -> tuple[str, str, str] | None:
        if not policy.compatible_with(TSH_CALO_ALGORITHM_ID):
            return (
                "Not compatible",
                "Not compatible",
                "This registered model does not match the current frozen TSH-CALO policy contract.",
            )
        source = Path(policy.checkpoint_path).expanduser()
        if source.is_symlink():
            return (
                "Unverified model target",
                "Not usable",
                "The registered model path is a symbolic link and is not accepted as a current executable artifact.",
            )
        if not source.is_file():
            return (
                "Model file unavailable",
                "Not usable",
                "The registered model file is no longer available at its recorded path.",
            )
        try:
            resolved = source.resolve(strict=True)
            stat = resolved.stat()
        except OSError as exc:
            return (
                "Model file unavailable",
                "Not usable",
                f"The registered model file could not be inspected ({type(exc).__name__}: {exc}).",
            )
        key = (
            policy.id,
            str(resolved).casefold(),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            str(policy.sha256).lower(),
        )
        cached = self._registered_obsolete_cache.get(key)
        if cached is not None:
            return cached if len(cached) == 3 else None
        try:
            observed = checkpoint_sha256(resolved).lower()
        except OSError as exc:
            issue = (
                "Saved model unreadable",
                "Not usable",
                f"The registered model file could not be read ({type(exc).__name__}: {exc}).",
            )
            self._registered_obsolete_cache[key] = issue
            return issue
        if observed != str(policy.sha256).lower():
            issue = (
                "Model integrity failed",
                "Not usable",
                "The model file no longer matches the SHA-256 recorded when it was registered.",
            )
            self._registered_obsolete_cache[key] = issue
            return issue
        self._registered_obsolete_cache[key] = ()
        return None

    @staticmethod
    def _entry_policy(entry):
        return entry.get("registered_policy") if isinstance(entry, dict) else entry

    def refresh_policy_library(self) -> None:
        selected_key = self._row_key(self._selected_row())
        super().refresh_policy_library()
        if not hasattr(self, "show_obsolete_models"):
            return

        issues: dict[str, tuple[str, str, str]] = {}
        for entry in self._policy_rows:
            if isinstance(entry, dict) and entry.get("row_kind") == "obsolete_training":
                continue
            policy = self._entry_policy(entry)
            if policy is None or getattr(policy, "archived", False):
                continue
            issue = self._registered_obsolete_issue(policy)
            if issue is not None:
                issues[policy.id] = issue
        if not issues:
            return

        snapshots = {}
        for row, entry in enumerate(self._policy_rows):
            snapshots[self._row_key(entry)] = tuple(
                (
                    self.policy_table.item(row, column).text(),
                    self.policy_table.item(row, column).toolTip(),
                )
                if self.policy_table.item(row, column) is not None
                else ("", "")
                for column in range(self.policy_table.columnCount())
            )

        show = self.show_obsolete_models.isChecked()
        filtered = []
        for entry in self._policy_rows:
            if isinstance(entry, dict) and entry.get("row_kind") == "obsolete_training":
                filtered.append(entry)
                continue
            policy = self._entry_policy(entry)
            if policy is not None and policy.id in issues and not show:
                continue
            filtered.append(entry)
        self._policy_rows = filtered
        self.policy_table.setRowCount(len(filtered))

        for row, entry in enumerate(filtered):
            snapshot = snapshots.get(self._row_key(entry), ())
            for column in range(self.policy_table.columnCount()):
                text, tooltip = snapshot[column] if column < len(snapshot) else ("", "")
                item = QTableWidgetItem(text)
                item.setToolTip(tooltip)
                self.policy_table.setItem(row, column, item)
            policy = self._entry_policy(entry)
            issue = issues.get(getattr(policy, "id", ""))
            if issue is not None:
                scientific_status, compatibility, reason = issue
                self.policy_table.setItem(row, 4, QTableWidgetItem(scientific_status))
                self.policy_table.setItem(row, 6, QTableWidgetItem(compatibility))
                self.policy_table.item(row, 4).setToolTip(reason)
                self.policy_table.item(row, 6).setToolTip(reason)

        self._resize_policy_table_to_entries()
        if selected_key:
            self._select_row_key(selected_key)
        elif self._policy_rows:
            self.policy_table.selectRow(0)
        self._policy_selection_changed()
