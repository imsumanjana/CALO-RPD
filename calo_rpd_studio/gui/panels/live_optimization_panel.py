"""Live optimization telemetry, repeated-run navigation, and portfolio-aware previews."""
from __future__ import annotations

import json
import math
from collections import Counter

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from calo_rpd_studio.gui.plotting.scientific_plot import ScientificPlotWidget
from calo_rpd_studio.gui.widgets.section_card import SectionCard
from calo_rpd_studio.gui.widgets.workspace_page import WorkspacePage
from calo_rpd_studio.portfolio.catalog import OUTPUT_REQUIREMENTS
from calo_rpd_studio.power_system.case_loader import CaseLoader
from calo_rpd_studio.power_system.case_model import BUS_I, VM


class LiveOptimizationPanel(WorkspacePage):
    """Monitor every repeated run without mixing their convergence histories.

    v3.4.2 keeps an independent telemetry store for each repeated run. ``Auto-follow`` advances
    monotonically to newly-started runs while a user can pin any observed run from the selector.
    Portfolio outputs selected before execution are also exposed as soon as their data dependency
    exists: true live traces update from optimizer telemetry, single-run power-system plots appear
    after a run is committed, and repeated-run summaries are marked provisional until complete.
    """

    AUTO_MODE = "Automatic (recommended)"
    OBJECTIVE_MODE = "Best feasible objective"
    VIOLATION_MODE = "Best constraint violation"
    CONSTRAINT_COMPONENT_MODE = "Constraint decomposition (CALO)"
    FEASIBILITY_MODE = "Feasible population ratio (CALO)"
    DIVERSITY_MODE = "Population diversity (CALO)"
    OPERATOR_SUCCESS_MODE = "Operator success rate (CALO)"

    LIVE_VIEW_KEY = "__live__"
    AUTO_RUN_KEY = -1
    LIVE_PORTFOLIO_KEYS = {
        "objective_convergence",
        "constraint_convergence",
        "constraint_decomposition",
        "calo_operator_success",
        "calo_operator_usage",
        "calo_regime_timeline",
    }
    FINAL_ONLY_KEYS = {
        "wilcoxon_holm",
        "effect_sizes",
        "friedman_ranking",
        "critical_difference",
        "descriptive_statistics",
        "cvar_curve",
        "contingency_matrix",
        "throughput_batch_scaling",
        "device_speedup",
        "parity_scatter",
    }

    def __init__(self, state, manager, parent=None) -> None:
        super().__init__(
            "Live Optimization",
            "Monitor every repeated run, switch among active/completed runs, and preview selected portfolio evidence as soon as it becomes scientifically available.",
            parent,
        )
        self.state = state
        self.manager = manager
        self.current_run_index: int | None = None
        self._run_series: dict[int, dict[str, dict[str, tuple[list[int], list[float]]]]] = {}
        self._run_latest_telemetry: dict[int, dict] = {}
        self._run_operator_counts: dict[int, dict[str, Counter[str]]] = {}
        self._regime_codes: dict[str, int] = {}
        self._known_runs: set[int] = set()
        self._portfolio_colorbar = None

        # Backward-compatible aliases point at the currently displayed repeated run.
        self.objective_series: dict[str, tuple[list[int], list[float]]] = {}
        self.violation_series: dict[str, tuple[list[int], list[float]]] = {}
        self.constraint_component_series: dict[str, tuple[list[int], list[float]]] = {}
        self.feasibility_series: dict[str, tuple[list[int], list[float]]] = {}
        self.diversity_series: dict[str, tuple[list[int], list[float]]] = {}
        self.operator_success_series: dict[str, tuple[list[int], list[float]]] = {}
        self.operator_usage_series: dict[str, tuple[list[int], list[float]]] = {}
        self.regime_series: dict[str, tuple[list[int], list[float]]] = {}
        self.preview_selection_by_metric: dict[str, set[str]] = {}
        self.preview_current_labels: list[str] = []

        scroll = QScrollArea()
        scroll.setObjectName("LiveOptimizationScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        content = QWidget()
        content.setObjectName("LiveOptimizationContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 8, 8)
        content_layout.setSpacing(16)

        telemetry = SectionCard("Current telemetry")
        grid = QGridLayout()
        grid.setHorizontalSpacing(24)
        grid.setVerticalSpacing(8)
        telemetry.layout_root.addLayout(grid)
        self.labels: dict[str, QLabel] = {}
        names = [
            "Algorithm",
            "Repeated run",
            "Iteration",
            "Evaluations",
            "Best feasible objective",
            "Best constraint violation",
            "Feasible incumbent",
            "CALO operator",
            "CALO regime",
            "Population diversity",
            "Feasible population ratio",
            "Epsilon-feasible ratio",
            "Adaptive epsilon",
            "Bus-voltage CV",
            "Generator-Q CV",
            "Generator-P CV",
            "Branch-thermal CV",
            "Evaluations to first feasibility",
            "Reward",
        ]
        rows_per_column = 7
        for index, name in enumerate(names):
            row = index % rows_per_column
            col = (index // rows_per_column) * 2
            key = QLabel(name)
            key.setObjectName("MetricLabel")
            value = QLabel("—")
            value.setObjectName("ContextValue")
            self.labels[name] = value
            grid.addWidget(key, row, col)
            grid.addWidget(value, row, col + 1)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        grid.setColumnStretch(5, 1)
        content_layout.addWidget(telemetry)

        run_row = QHBoxLayout()
        run_row.addWidget(QLabel("Repeated run view"))
        self.run_selector = QComboBox()
        self.run_selector.addItem("Auto-follow latest started run", self.AUTO_RUN_KEY)
        self.run_selector.currentIndexChanged.connect(self._run_selection_changed)
        run_row.addWidget(self.run_selector)
        self.run_note = QLabel(
            "Telemetry from all runs is retained independently. Auto-follow advances to newly started repeated runs; choose a run number to pin it."
        )
        self.run_note.setWordWrap(True)
        self.run_note.setObjectName("HelpText")
        run_row.addWidget(self.run_note, 1)
        content_layout.addLayout(run_row)

        portfolio_row = QHBoxLayout()
        portfolio_row.addWidget(QLabel("Live/portfolio view"))
        self.portfolio_view = QComboBox()
        self.portfolio_view.currentIndexChanged.connect(self._portfolio_view_changed)
        portfolio_row.addWidget(self.portfolio_view)
        self.portfolio_status = QLabel()
        self.portfolio_status.setWordWrap(True)
        self.portfolio_status.setObjectName("HelpText")
        portfolio_row.addWidget(self.portfolio_status, 1)
        content_layout.addLayout(portfolio_row)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Convergence metric"))
        self.metric = QComboBox()
        self.metric.addItems(
            [
                self.AUTO_MODE,
                self.OBJECTIVE_MODE,
                self.VIOLATION_MODE,
                self.CONSTRAINT_COMPONENT_MODE,
                self.FEASIBILITY_MODE,
                self.DIVERSITY_MODE,
                self.OPERATOR_SUCCESS_MODE,
            ]
        )
        self.metric.currentTextChanged.connect(self._redraw_plot)
        mode_row.addWidget(self.metric)
        self.metric_note = QLabel(
            "Automatic mode shows constraint-violation convergence until the monitored algorithms have reached feasibility, then switches to best-feasible objective."
        )
        self.metric_note.setWordWrap(True)
        self.metric_note.setObjectName("HelpText")
        mode_row.addWidget(self.metric_note, 1)
        content_layout.addLayout(mode_row)

        self.plot = ScientificPlotWidget(
            title="Live convergence",
            xlabel="Objective-function evaluations",
            ylabel=self.OBJECTIVE_MODE,
            square_preview=True,
            square_export=True,
            square_preview_size=720,
            auto_fit_visible_data=True,
            auto_include_zero=True,
            auto_scale_padding=0.08,
        )
        self.plot.configure_preview_series(
            self._preview_options,
            self._preview_selection,
            self._apply_preview_selection,
        )
        content_layout.addWidget(self.plot, 0, Qt.AlignmentFlag.AlignHCenter)
        content_layout.addStretch(1)

        scroll.setWidget(content)
        self.layout_root.addWidget(scroll, 1)

        manager.progress.connect(self.update_progress)
        manager.started.connect(lambda _: self.reset())
        manager.run_completed.connect(self._on_run_completed)
        manager.run_failed.connect(self._on_run_failed)
        manager.completed.connect(self.load_experiment)
        manager.cancelled.connect(self.load_experiment)
        self.state.config_changed.connect(lambda *_: self._refresh_portfolio_views())

        self._refresh_portfolio_views()
        self._show_waiting_message()

    @staticmethod
    def _new_bucket() -> dict[str, dict[str, tuple[list[int], list[float]]]]:
        return {
            "objective": {},
            "violation": {},
            "constraint": {},
            "feasibility": {},
            "diversity": {},
            "operator_success": {},
            "operator_usage": {},
            "regime": {},
        }

    def _bucket(self, run_index: int) -> dict[str, dict[str, tuple[list[int], list[float]]]]:
        run_index = int(run_index)
        if run_index not in self._run_series:
            self._run_series[run_index] = self._new_bucket()
        self._ensure_run_option(run_index)
        return self._run_series[run_index]

    def _ensure_run_option(self, run_index: int) -> None:
        run_index = int(run_index)
        if run_index in self._known_runs:
            return
        self._known_runs.add(run_index)
        self.run_selector.blockSignals(True)
        try:
            ordered = sorted(self._known_runs)
            current_data = self.run_selector.currentData()
            self.run_selector.clear()
            self.run_selector.addItem("Auto-follow latest started run", self.AUTO_RUN_KEY)
            for value in ordered:
                self.run_selector.addItem(f"Repeated run {value}", value)
            target = self.run_selector.findData(current_data)
            self.run_selector.setCurrentIndex(max(0, target))
        finally:
            self.run_selector.blockSignals(False)

    def _activate_run(self, run_index: int | None) -> None:
        self.current_run_index = None if run_index is None else int(run_index)
        if self.current_run_index is None:
            bucket = self._new_bucket()
        else:
            bucket = self._bucket(self.current_run_index)
        self.objective_series = bucket["objective"]
        self.violation_series = bucket["violation"]
        self.constraint_component_series = bucket["constraint"]
        self.feasibility_series = bucket["feasibility"]
        self.diversity_series = bucket["diversity"]
        self.operator_success_series = bucket["operator_success"]
        self.operator_usage_series = bucket["operator_usage"]
        self.regime_series = bucket["regime"]
        telemetry = self._run_latest_telemetry.get(self.current_run_index or -1)
        if telemetry:
            self._display_telemetry(telemetry)

    def reset(self) -> None:
        self.current_run_index = None
        self._run_series = {}
        self._run_latest_telemetry = {}
        self._run_operator_counts = {}
        self._regime_codes = {}
        self._known_runs = set()
        self._activate_run(None)
        self.preview_selection_by_metric = {}
        self.preview_current_labels = []
        self.run_selector.blockSignals(True)
        self.run_selector.clear()
        self.run_selector.addItem("Auto-follow latest started run", self.AUTO_RUN_KEY)
        self.run_selector.blockSignals(False)
        if hasattr(self, "metric"):
            self.metric.blockSignals(True)
            self.metric.setCurrentText(self.AUTO_MODE)
            self.metric.blockSignals(False)
        self._refresh_portfolio_views()
        self._show_waiting_message()

    def _show_waiting_message(self) -> None:
        self._remove_colorbar()
        self.plot.show_message(
            "Waiting for optimizer telemetry. Start or resume an experiment to populate live convergence.",
            title="Live convergence",
            xlabel="Objective-function evaluations",
            ylabel="Convergence metric",
        )

    @staticmethod
    def _append_point(
        store: dict[str, tuple[list[int], list[float]]], label: str, x: int, y: float
    ) -> None:
        xs, ys = store.setdefault(label, ([], []))
        if xs and x < xs[-1]:
            return
        if xs and x == xs[-1]:
            ys[-1] = y
        else:
            xs.append(x)
            ys.append(y)

    @staticmethod
    def _has_points(series: dict[str, tuple[list[int], list[float]]]) -> bool:
        return any(bool(xs) and bool(ys) for xs, ys in series.values())

    def _automatic_mode(self) -> str:
        seen = {label for label, (xs, ys) in self.violation_series.items() if xs and ys}
        feasible = {label for label, (xs, ys) in self.objective_series.items() if xs and ys}
        if seen and seen.issubset(feasible):
            return self.OBJECTIVE_MODE
        return self.VIOLATION_MODE

    def _preview_metric_key(self) -> str:
        portfolio_key = self.portfolio_view.currentData() if hasattr(self, "portfolio_view") else None
        if portfolio_key and portfolio_key != self.LIVE_VIEW_KEY:
            return f"portfolio:{portfolio_key}"
        requested = self.metric.currentText() if hasattr(self, "metric") else self.AUTO_MODE
        return self._automatic_mode() if requested == self.AUTO_MODE else requested

    def _preview_options(self) -> list[tuple[str, str]]:
        return [(label, label) for label in self.preview_current_labels]

    def _preview_selection(self) -> set[str] | None:
        return self.preview_selection_by_metric.get(self._preview_metric_key())

    def _apply_preview_selection(self, selected: set[str] | None) -> None:
        key = self._preview_metric_key()
        if selected is None:
            self.preview_selection_by_metric.pop(key, None)
        else:
            self.preview_selection_by_metric[key] = set(selected)
        self._redraw_plot()

    def _filter_preview_series(self, series):
        self.preview_current_labels = list(series.keys())
        selected = self.preview_selection_by_metric.get(self._preview_metric_key())
        if selected is None:
            return dict(series)
        return {label: values for label, values in series.items() if label in selected}

    def _remove_colorbar(self) -> None:
        if self._portfolio_colorbar is not None:
            try:
                self._portfolio_colorbar.remove()
            except Exception:
                pass
            self._portfolio_colorbar = None

    def _draw_series(
        self,
        series,
        title: str,
        ylabel: str,
        empty_message: str,
        *,
        include_zero: bool = False,
        xlabel: str = "Objective-function evaluations",
    ) -> None:
        self._remove_colorbar()
        if not self._has_points(series):
            self.preview_current_labels = []
            self.plot.show_message(empty_message, title=title, xlabel=xlabel, ylabel=ylabel)
            return
        preview_series = self._filter_preview_series(series)
        if not preview_series:
            self.plot.show_message(
                "No preview series are selected. Open Plot Tools → Preview series and select one or more entries.",
                title=title,
                xlabel=xlabel,
                ylabel=ylabel,
            )
            return
        self.plot.set_auto_scale_context(include_zero=include_zero)
        self.plot.plot_xy_series(preview_series, title, xlabel, ylabel)

    def _apply_axis(self, title: str, xlabel: str, ylabel: str) -> None:
        meta = self.plot.manager.records[self.plot.plot_id].metadata
        meta["title"] = title
        meta["xlabel"] = xlabel
        meta["ylabel"] = ylabel
        self.plot.manager.apply(self.plot.plot_id, self.plot.style)

    def _run_selection_changed(self) -> None:
        selected = self.run_selector.currentData()
        if selected == self.AUTO_RUN_KEY:
            target = max(self._known_runs, default=None)
        else:
            target = int(selected) if selected is not None else None
        self._activate_run(target)
        self._redraw_plot()

    def _refresh_portfolio_views(self) -> None:
        if not hasattr(self, "portfolio_view"):
            return
        selected_data = self.portfolio_view.currentData()
        requested = list(getattr(getattr(self.state.config, "portfolio", None), "requested_outputs", []))
        self.portfolio_view.blockSignals(True)
        try:
            self.portfolio_view.clear()
            self.portfolio_view.addItem("Live telemetry metric", self.LIVE_VIEW_KEY)
            for key in requested:
                requirement = OUTPUT_REQUIREMENTS.get(key)
                self.portfolio_view.addItem(requirement.label if requirement else key, key)
            index = self.portfolio_view.findData(selected_data)
            self.portfolio_view.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.portfolio_view.blockSignals(False)
        self._portfolio_view_changed()

    def _portfolio_view_changed(self) -> None:
        key = self.portfolio_view.currentData()
        live = key in (None, self.LIVE_VIEW_KEY)
        self.metric.setEnabled(live)
        if live:
            self.portfolio_status.setText(
                "LIVE — telemetry updates while the selected repeated run is executing."
            )
        elif key in self.LIVE_PORTFOLIO_KEYS:
            self.portfolio_status.setText(
                "LIVE — this selected Portfolio Manager output is available directly from runtime telemetry."
            )
        elif key in self.FINAL_ONLY_KEYS:
            self.portfolio_status.setText(
                "FINAL-STAGE — this evidence depends on completed paired runs, validation/statistics, or dedicated accelerator records."
            )
        else:
            self.portfolio_status.setText(
                "PROGRESSIVE — this selected output appears as soon as its required completed-run data is committed."
            )
        self._redraw_plot()

    def _redraw_plot(self) -> None:
        portfolio_key = self.portfolio_view.currentData() if hasattr(self, "portfolio_view") else None
        if portfolio_key and portfolio_key != self.LIVE_VIEW_KEY:
            self._redraw_portfolio_preview(str(portfolio_key))
            return

        requested = self.metric.currentText() if hasattr(self, "metric") else self.AUTO_MODE
        mode = self._automatic_mode() if requested == self.AUTO_MODE else requested
        run_suffix = "" if self.current_run_index is None else f" — repeated run {self.current_run_index}"

        if mode == self.VIOLATION_MODE:
            self._draw_series(
                self.violation_series,
                f"Constraint-violation convergence{run_suffix}",
                "Best normalized constraint violation",
                "No convergence telemetry has been received yet.",
                include_zero=True,
            )
            self.metric_note.setText(
                "Showing normalized constraint-violation convergence because feasibility has not yet been reached by every monitored optimizer."
                if requested == self.AUTO_MODE
                else "Constraint violation is the correct convergence metric before a feasible incumbent exists."
            )
        elif mode == self.OBJECTIVE_MODE:
            self._draw_series(
                self.objective_series,
                f"Best-feasible objective convergence{run_suffix}",
                "Best feasible objective",
                "No feasible incumbent has been reached yet. Use Automatic mode or Best constraint violation to monitor progress.",
            )
            self.metric_note.setText(
                "Showing monotonic best-feasible objective convergence against objective-function evaluations."
            )
        elif mode == self.CONSTRAINT_COMPONENT_MODE:
            self._draw_series(
                self.constraint_component_series,
                f"Constraint decomposition{run_suffix}",
                "Normalized constraint component",
                "Constraint-component telemetry is available for CALO Core v2 runs.",
                include_zero=True,
            )
            self.metric_note.setText(
                "The decomposition identifies whether voltage, generator-Q, generator-P, branch-thermal, or power-flow constraints dominate infeasibility."
            )
        elif mode == self.FEASIBILITY_MODE:
            self._draw_series(
                self.feasibility_series,
                f"Feasibility evolution{run_suffix}",
                "Population ratio",
                "Feasible-population telemetry is available for CALO Core v2 runs.",
                include_zero=True,
            )
            self.metric_note.setText(
                "Exact feasible ratio and adaptive epsilon-feasible ratio are shown separately; final reported solutions still require exact feasibility."
            )
        elif mode == self.DIVERSITY_MODE:
            self._draw_series(
                self.diversity_series,
                f"Population diversity{run_suffix}",
                "Normalized decision-space diversity",
                "Population-diversity telemetry is available for CALO Core v2 runs.",
                include_zero=True,
            )
            self.metric_note.setText(
                "Population and elite diversity help diagnose premature collapse and excessive dispersion."
            )
        else:
            self._draw_series(
                self.operator_success_series,
                f"CALO operator success rate{run_suffix}",
                "Recent success rate",
                "Operator-success telemetry is available for CALO Core v2 runs.",
                include_zero=True,
            )
            self.metric_note.setText(
                "Recent operator success rates are measured online and combined with the learned policy to adapt CALO during the current run."
            )

    def _experiment_rows(self, *, verified_only: bool = False) -> list[dict]:
        experiment_id = getattr(self.state, "current_experiment_id", None)
        if not experiment_id:
            return []
        return self.state.database.list_runs(experiment_id, verified_only=verified_only)

    @staticmethod
    def _row_result(row: dict) -> dict:
        try:
            return json.loads(row.get("result_json", "{}"))
        except (TypeError, json.JSONDecodeError):
            return {}

    def _rows_for_active_run(self, rows: list[dict]) -> list[dict]:
        if self.current_run_index is None:
            return rows
        return [row for row in rows if int(row.get("run_index", -1)) + 1 == self.current_run_index]

    @staticmethod
    def _best_row(rows: list[dict]) -> dict | None:
        candidates = []
        for row in rows:
            result = LiveOptimizationPanel._row_result(row)
            objective = result.get("best_objective", np.inf)
            if bool(result.get("feasible")) and isinstance(objective, (int, float)) and np.isfinite(objective):
                candidates.append((float(objective), row))
        return min(candidates, key=lambda item: item[0])[1] if candidates else (rows[0] if rows else None)

    @staticmethod
    def _solution_scenario(row: dict) -> dict | None:
        result = LiveOptimizationPanel._row_result(row)
        state = (result.get("metadata") or {}).get("solution_state") or {}
        scenarios = state.get("scenarios") or []
        return scenarios[0] if scenarios else None

    def _show_portfolio_wait(self, key: str, message: str) -> None:
        req = OUTPUT_REQUIREMENTS.get(key)
        label = req.label if req else key
        self.preview_current_labels = []
        self._remove_colorbar()
        self.plot.show_message(message, title=label, xlabel="Evidence progress", ylabel="")

    def _redraw_portfolio_preview(self, key: str) -> None:
        run_suffix = "" if self.current_run_index is None else f" — repeated run {self.current_run_index}"
        if key == "objective_convergence":
            self._draw_series(
                self.objective_series,
                f"Objective convergence{run_suffix}",
                "Best feasible objective",
                "No feasible-objective telemetry is available yet.",
            )
            return
        if key == "constraint_convergence":
            self._draw_series(
                self.violation_series,
                f"Constraint convergence{run_suffix}",
                "Best normalized constraint violation",
                "No constraint telemetry is available yet.",
                include_zero=True,
            )
            return
        if key == "constraint_decomposition":
            self._draw_series(
                self.constraint_component_series,
                f"Constraint decomposition{run_suffix}",
                "Normalized constraint component",
                "No constraint-component telemetry is available yet.",
                include_zero=True,
            )
            return
        if key == "calo_operator_success":
            self._draw_series(
                self.operator_success_series,
                f"CALO operator success{run_suffix}",
                "Recent success rate",
                "No CALO operator-success telemetry is available yet.",
                include_zero=True,
            )
            return
        if key == "calo_operator_usage":
            self._draw_series(
                self.operator_usage_series,
                f"CALO operator utilization{run_suffix}",
                "Cumulative utilization share",
                "No CALO operator-selection telemetry is available yet.",
                include_zero=True,
            )
            return
        if key == "calo_regime_timeline":
            self._draw_series(
                self.regime_series,
                f"CALO cognitive-regime timeline{run_suffix}",
                "Regime code",
                "No CALO regime telemetry is available yet.",
                include_zero=True,
            )
            if self._regime_codes:
                mapping = ", ".join(f"{code}={name}" for name, code in self._regime_codes.items())
                self.portfolio_status.setText(f"LIVE — regime codes: {mapping}")
            return

        rows = self._experiment_rows(
            verified_only=key in {"best_validated_voltage_profile", "best_validated_branch_heatmap"}
        )
        active_rows = self._rows_for_active_run(rows)
        # Publication-grade "best validated" previews must never fall back to an
        # independently verified but infeasible run.  This mirrors the export gate.
        if key.startswith("best_validated_"):
            active_rows = [
                row for row in active_rows if bool(self._row_result(row).get("feasible"))
            ]
        req = OUTPUT_REQUIREMENTS.get(key)
        required = int(req.minimum_runs if req else 1)
        completed_run_count = len({int(row.get("run_index", -1)) for row in rows})
        total_planned = int(getattr(self.state.config, "runs", max(required, 1)))

        if key in self.FINAL_ONLY_KEYS:
            self._show_portfolio_wait(
                key,
                f"This output is intentionally final-stage evidence. Completed repeated runs: {completed_run_count}/{total_planned}. Generate it after the paired experiment/statistical dependencies are complete.",
            )
            return

        if key in {
            "median_convergence",
            "convergence_uncertainty_band",
            "objective_boxplot",
            "objective_violin",
            "feasible_run_probability",
            "evaluations_to_feasibility",
            "objective_violation_scatter",
        }:
            if not rows:
                self._show_portfolio_wait(key, "Waiting for the first completed run to be committed.")
                return
            self.portfolio_status.setText(
                f"PROVISIONAL — {completed_run_count}/{total_planned} repeated runs have committed evidence. The preview updates after each completed run."
            )
            if key in {"median_convergence", "convergence_uncertainty_band"}:
                self._draw_progressive_median(rows, uncertainty=key == "convergence_uncertainty_band")
            elif key in {"objective_boxplot", "objective_violin"}:
                self._draw_progressive_distribution(rows, violin=key == "objective_violin")
            elif key == "feasible_run_probability":
                self._draw_feasible_probability(rows)
            elif key == "evaluations_to_feasibility":
                self._draw_evaluations_to_feasibility(rows)
            else:
                self._draw_objective_violation(rows)
            return

        if not active_rows:
            qualifier = " independently verified" if key.startswith("best_validated_") else ""
            self._show_portfolio_wait(
                key,
                f"Waiting for a{qualifier} completed result for repeated run {self.current_run_index or '—'}. Single-run plots become available immediately after the run is committed.",
            )
            return

        if key.startswith("best_validated_"):
            self.portfolio_status.setText(
                "VALIDATED — preview uses only independently verified, feasible completed evidence."
            )
        else:
            independently_verified = all(
                str(row.get("validation_status", "")).lower() == "verified" for row in active_rows
            )
            qualifier = "independently verified" if independently_verified else "preview; independent validation may still be pending"
            self.portfolio_status.setText(
                f"AVAILABLE — {len(active_rows)} completed algorithm result(s) for repeated run {self.current_run_index}; {qualifier}."
            )

        if key == "voltage_profile" or key == "best_validated_voltage_profile":
            series = {}
            if key == "voltage_profile":
                try:
                    base_case = CaseLoader.load(str(self.state.config.case_name))
                    series["Initial base case"] = (
                        [int(value) for value in base_case.bus[:, BUS_I]],
                        [float(value) for value in base_case.bus[:, VM]],
                    )
                except Exception:
                    # A live preview must never break the optimization UI because a
                    # custom case cannot be reloaded for its initial trace.
                    pass
            for row in active_rows:
                scenario = self._solution_scenario(row)
                if not scenario:
                    continue
                x = [int(v) for v in scenario.get("bus_numbers", [])]
                y = [float(v) for v in scenario.get("vm_pu", [])]
                if x and y:
                    series[str(row.get("algorithm", "Optimizer"))] = (x, y)
            self._draw_series(series, f"Optimized bus-voltage profile{run_suffix}", "Voltage magnitude (p.u.)", "Voltage state is unavailable.", xlabel="Bus number")
            return

        row = self._best_row(active_rows)
        if row is None:
            self._show_portfolio_wait(key, "No completed result is available for this preview.")
            return
        scenario = self._solution_scenario(row)
        algorithm = str(row.get("algorithm", "Optimizer"))
        if key == "voltage_heatmap" and scenario:
            self._draw_heatmap(np.asarray(scenario.get("vm_pu", []), dtype=float), scenario.get("bus_numbers", []), f"Bus-voltage heatmap — {algorithm}", "Bus")
        elif key in {"branch_loading", "branch_loading_heatmap", "best_validated_branch_heatmap"} and scenario:
            values = np.asarray(scenario.get("loading_percent", []), dtype=float)
            if key == "branch_loading":
                labels = list(range(1, len(values) + 1))
                self._draw_bar(labels, values, f"Optimized branch loading — {algorithm}", "Branch index", "Loading (%)")
            else:
                labels = [
                    f"{a}-{b}"
                    for a, b in zip(
                        scenario.get("branch_from_bus", range(len(values))),
                        scenario.get("branch_to_bus", range(len(values))),
                    )
                ]
                self._draw_heatmap(values, labels, f"Branch-loading heatmap — {algorithm}", "Branch")
        elif key == "generator_reactive_power" and scenario:
            self._draw_bar(
                scenario.get("generator_bus", []),
                np.asarray(scenario.get("qg_mvar", []), dtype=float),
                f"Generator reactive power — {algorithm}",
                "Generator bus",
                "Q (MVAr)",
            )
        elif key == "control_changes":
            result = self._row_result(row)
            controls = result.get("decoded_controls") or {}
            labels = [str(k) for k, v in controls.items() if isinstance(v, (int, float))]
            values = np.asarray([float(controls[k]) for k in labels], dtype=float)
            self._draw_bar(labels, values, f"Optimized ORPD controls — {algorithm}", "Control", "Physical value")
        else:
            self._show_portfolio_wait(
                key,
                "The selected output requires evidence that is not yet available in the committed run. It remains available from Publication & Portfolio Export when its dependency is satisfied.",
            )

    def _draw_heatmap(self, values: np.ndarray, labels, title: str, xlabel: str) -> None:
        self.preview_current_labels = []
        self._remove_colorbar()
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            self.plot.show_message("Heatmap data are unavailable.", title=title, xlabel=xlabel, ylabel="")
            return
        self.plot.axis.clear()
        image = self.plot.axis.imshow(values.reshape(1, -1), aspect="auto")
        self.plot.axis.set_yticks([0])
        self.plot.axis.set_yticklabels(["Value"])
        labels = list(labels)
        if labels:
            step = max(1, len(labels) // 16)
            ticks = np.arange(0, len(labels), step)
            self.plot.axis.set_xticks(ticks)
            self.plot.axis.set_xticklabels([str(labels[i]) for i in ticks], rotation=45, ha="right")
        self._portfolio_colorbar = self.plot.figure.colorbar(image, ax=self.plot.axis, shrink=0.75)
        self._apply_axis(title, xlabel, "")

    def _draw_bar(self, labels, values: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
        self.preview_current_labels = []
        self._remove_colorbar()
        values = np.asarray(values, dtype=float)
        if values.size == 0:
            self.plot.show_message("Plot data are unavailable.", title=title, xlabel=xlabel, ylabel=ylabel)
            return
        self.plot.axis.clear()
        positions = np.arange(values.size)
        self.plot.axis.bar(positions, values)
        labels = [str(value) for value in labels]
        if labels:
            step = max(1, len(labels) // 20)
            ticks = positions[::step]
            self.plot.axis.set_xticks(ticks)
            self.plot.axis.set_xticklabels(labels[::step], rotation=55, ha="right")
        self._apply_axis(title, xlabel, ylabel)

    @staticmethod
    def _aligned_histories(rows: list[dict]) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
        by_algorithm: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        max_eval = 0
        for row in rows:
            result = LiveOptimizationPanel._row_result(row)
            metadata = result.get("metadata") or {}
            x = np.asarray(metadata.get("convergence_evaluations", []), dtype=float)
            y = np.asarray(metadata.get("best_feasible_objective_history", []), dtype=float)
            n = min(x.size, y.size)
            if n == 0:
                continue
            x, y = x[:n], y[:n]
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.any():
                x, y = x[mask], y[mask]
                max_eval = max(max_eval, int(x[-1]))
                by_algorithm.setdefault(str(row.get("algorithm", "Optimizer")), []).append((x, y))
        if max_eval <= 0:
            return np.asarray([]), {}
        grid = np.unique(np.linspace(0, max_eval, min(300, max_eval + 1), dtype=int)).astype(float)
        aligned: dict[str, list[np.ndarray]] = {}
        for algorithm, histories in by_algorithm.items():
            series = []
            for x, y in histories:
                indices = np.searchsorted(x, grid, side="right") - 1
                indices = np.clip(indices, 0, len(y) - 1)
                values = y[indices].astype(float, copy=True)
                values[grid < x[0]] = np.nan
                series.append(values)
            aligned[algorithm] = series
        return grid, aligned

    def _draw_progressive_median(self, rows: list[dict], *, uncertainty: bool) -> None:
        self.preview_current_labels = []
        self._remove_colorbar()
        grid, aligned = self._aligned_histories(rows)
        if grid.size == 0 or not aligned:
            self.plot.show_message("No feasible convergence history has been committed yet.", title="Median convergence", xlabel="Objective-function evaluations", ylabel="Best feasible objective")
            return
        self.plot.axis.clear()
        for algorithm, series in aligned.items():
            matrix = np.asarray(series, dtype=float)
            valid_columns = np.any(np.isfinite(matrix), axis=0)
            if not np.any(valid_columns):
                continue
            local_grid = grid[valid_columns]
            local_matrix = matrix[:, valid_columns]
            median = np.nanmedian(local_matrix, axis=0)
            line = self.plot.axis.plot(local_grid, median, label=algorithm)[0]
            if uncertainty and local_matrix.shape[0] >= 2:
                q1 = np.nanpercentile(local_matrix, 25, axis=0)
                q3 = np.nanpercentile(local_matrix, 75, axis=0)
                self.plot.axis.fill_between(local_grid, q1, q3, alpha=0.18, color=line.get_color())
        self._apply_axis("Provisional median feasible convergence" + (" with IQR" if uncertainty else ""), "Objective-function evaluations", "Best feasible objective")

    def _objective_records(self, rows: list[dict]) -> dict[str, list[float]]:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            result = self._row_result(row)
            value = result.get("best_objective")
            if bool(result.get("feasible")) and isinstance(value, (int, float)) and np.isfinite(value):
                grouped.setdefault(str(row.get("algorithm", "Optimizer")), []).append(float(value))
        return grouped

    def _draw_progressive_distribution(self, rows: list[dict], *, violin: bool) -> None:
        self.preview_current_labels = []
        self._remove_colorbar()
        grouped = self._objective_records(rows)
        if not grouped:
            self.plot.show_message("No feasible completed objective values are available yet.", title="Objective distribution", xlabel="Algorithm", ylabel="Final feasible objective")
            return
        labels = list(grouped)
        data = [grouped[label] for label in labels]
        self.plot.axis.clear()
        if violin and all(len(values) >= 2 for values in data):
            self.plot.axis.violinplot(data, showmedians=True)
        else:
            self.plot.axis.boxplot(data, tick_labels=labels, showfliers=True)
        self.plot.axis.set_xticks(np.arange(1, len(labels) + 1))
        self.plot.axis.set_xticklabels(labels, rotation=45, ha="right")
        self._apply_axis("Provisional final feasible objective distribution", "Algorithm", "Final feasible objective")

    def _draw_feasible_probability(self, rows: list[dict]) -> None:
        counts: dict[str, list[bool]] = {}
        for row in rows:
            result = self._row_result(row)
            counts.setdefault(str(row.get("algorithm", "Optimizer")), []).append(bool(result.get("feasible")))
        labels = list(counts)
        values = np.asarray([100.0 * np.mean(counts[label]) for label in labels], dtype=float)
        self._draw_bar(labels, values, "Provisional feasible-run probability", "Algorithm", "Feasible runs (%)")
        self.plot.axis.set_ylim(0, 105)
        self.plot.canvas.draw_idle()

    def _draw_evaluations_to_feasibility(self, rows: list[dict]) -> None:
        grouped: dict[str, list[float]] = {}
        for row in rows:
            result = self._row_result(row)
            first = (result.get("metadata") or {}).get("first_feasible_evaluation")
            if first is not None:
                grouped.setdefault(str(row.get("algorithm", "Optimizer")), []).append(float(first))
        if not grouped:
            self._show_portfolio_wait("evaluations_to_feasibility", "No completed run has reached feasibility yet.")
            return
        labels = list(grouped)
        self.preview_current_labels = []
        self._remove_colorbar()
        self.plot.axis.clear()
        self.plot.axis.boxplot([grouped[label] for label in labels], tick_labels=labels)
        self.plot.axis.set_xticklabels(labels, rotation=45, ha="right")
        self._apply_axis("Provisional evaluations to first feasibility", "Algorithm", "Objective-function evaluations")

    def _draw_objective_violation(self, rows: list[dict]) -> None:
        self.preview_current_labels = []
        self._remove_colorbar()
        self.plot.axis.clear()
        any_points = False
        for algorithm in sorted({str(row.get("algorithm", "Optimizer")) for row in rows}):
            x, y = [], []
            for row in rows:
                if str(row.get("algorithm", "Optimizer")) != algorithm:
                    continue
                result = self._row_result(row)
                violation = result.get("total_constraint_violation")
                objective = result.get("best_objective")
                if isinstance(violation, (int, float)) and isinstance(objective, (int, float)) and np.isfinite(violation) and np.isfinite(objective):
                    x.append(float(violation)); y.append(float(objective))
            if x:
                any_points = True
                self.plot.axis.scatter(x, y, label=algorithm, alpha=0.75)
        if not any_points:
            self.plot.show_message("No completed objective/violation records are available yet.", title="Objective–violation relationship", xlabel="Constraint violation", ylabel="Objective")
            return
        self._apply_axis("Provisional objective–violation relationship", "Final normalized constraint violation", "Final objective")

    def _load_rows_into_runs(self, rows: list[dict]) -> None:
        for row in rows:
            run_number = int(row.get("run_index", 0)) + 1
            bucket = self._bucket(run_number)
            result = self._row_result(row)
            metadata = result.get("metadata") or {}
            evaluations = [int(v) for v in metadata.get("convergence_evaluations", [])]
            algorithm = str(row.get("algorithm") or result.get("algorithm") or "Optimizer")
            for y, store_name in (
                (metadata.get("best_feasible_objective_history", []), "objective"),
                (metadata.get("best_constraint_violation_history", []), "violation"),
            ):
                xs, ys = [], []
                for x_value, y_value in zip(evaluations, y):
                    try:
                        value = float(y_value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(value):
                        xs.append(int(x_value)); ys.append(value)
                if xs:
                    bucket[store_name][algorithm] = (xs, ys)
            histories = metadata.get("constraint_component_histories") or metadata.get("diagnostics_history") or {}
            for key_name, label in (
                ("bus_voltage", "Bus voltage"),
                ("generator_q", "Generator Q"),
                ("generator_p", "Generator P"),
                ("branch_thermal", "Branch thermal"),
                ("power_flow", "Power flow"),
                ("best_bus_voltage", "Bus voltage"),
                ("best_generator_q", "Generator Q"),
                ("best_generator_p", "Generator P"),
                ("best_branch_thermal", "Branch thermal"),
                ("best_power_flow", "Power flow"),
            ):
                values = histories.get(key_name, [])
                if values:
                    bucket["constraint"][f"{algorithm} · {label}"] = (evaluations[: len(values)], [float(v) for v in values])
            diagnostics = metadata.get("diagnostics_history") or {}
            for key_name, label in (("feasible_ratio", "Exact feasible ratio"), ("epsilon_feasible_ratio", "Epsilon-feasible ratio")):
                values = diagnostics.get(key_name, [])
                if values:
                    bucket["feasibility"][f"{algorithm} · {label}"] = (evaluations[: len(values)], [float(v) for v in values])
            for key_name, label in (("population_diversity", "Population diversity"), ("elite_diversity", "Elite diversity")):
                values = diagnostics.get(key_name, [])
                if values:
                    bucket["diversity"][f"{algorithm} · {label}"] = (evaluations[: len(values)], [float(v) for v in values])
            success_history = metadata.get("operator_success_history") or []
            for operator in metadata.get("operator_names", []):
                values = [float(item.get(operator, 0.0)) for item in success_history]
                if values:
                    bucket["operator_success"][f"{algorithm} · {operator}"] = (evaluations[: len(values)], values)

    def view_state(self) -> dict:
        """Return lightweight reproducible UI state; scientific curves remain reconstructed from DB."""
        return {
            "run_selector": self.run_selector.currentData(),
            "portfolio_view": self.portfolio_view.currentData(),
            "metric": self.metric.currentText(),
            "preview_selection_by_metric": {key: sorted(value) for key, value in self.preview_selection_by_metric.items()},
        }

    def restore_view_state(self, payload: dict | None) -> None:
        data = dict(payload or {})
        self.preview_selection_by_metric = {
            str(key): {str(item) for item in value}
            for key, value in dict(data.get("preview_selection_by_metric") or {}).items()
        }
        portfolio_key = data.get("portfolio_view")
        idx = self.portfolio_view.findData(portfolio_key)
        if idx >= 0:
            self.portfolio_view.setCurrentIndex(idx)
        metric = str(data.get("metric", "") or "")
        idx = self.metric.findText(metric)
        if idx >= 0:
            self.metric.setCurrentIndex(idx)
        selected_run = data.get("run_selector", self.AUTO_RUN_KEY)
        idx = self.run_selector.findData(selected_run)
        if idx >= 0:
            self.run_selector.setCurrentIndex(idx)
        self._redraw_plot()

    def load_experiment(self, experiment_id: str) -> None:
        """Restore all repeated-run histories after completion/cancellation."""
        self.reset()
        if not experiment_id:
            return
        rows = self.state.database.list_runs(experiment_id)
        if not rows:
            return
        self._load_rows_into_runs(rows)
        target = max(self._known_runs, default=None)
        selected = self.run_selector.currentData()
        if selected != self.AUTO_RUN_KEY and selected in self._known_runs:
            target = int(selected)
        self._activate_run(target)
        self._redraw_plot()

    def _display_telemetry(self, data: dict) -> None:
        algorithm = str(data.get("algorithm", "—"))
        run_index = int(data.get("run_index", self.current_run_index or 1))
        evaluations = int(data.get("evaluations", 0) or 0)
        self.labels["Algorithm"].setText(algorithm)
        self.labels["Repeated run"].setText(str(run_index))
        self.labels["Iteration"].setText(str(data.get("iteration", "—")))
        self.labels["Evaluations"].setText(str(evaluations if evaluations else "—"))
        feasible_best = data.get("best_feasible_objective")
        self.labels["Best feasible objective"].setText(
            f"{float(feasible_best):.10g}"
            if isinstance(feasible_best, (int, float)) and math.isfinite(float(feasible_best))
            else "Not reached"
        )
        violation = data.get("best_constraint_violation")
        self.labels["Best constraint violation"].setText(
            f"{float(violation):.10g}"
            if isinstance(violation, (int, float)) and math.isfinite(float(violation))
            else "—"
        )
        self.labels["Feasible incumbent"].setText(str(data.get("feasible", "—")))
        self.labels["CALO operator"].setText(str(data.get("calo_operator", "—")))
        self.labels["CALO regime"].setText(str(data.get("calo_regime", "—")))
        self.labels["Population diversity"].setText(f"{data['diversity']:.5g}" if "diversity" in data else "—")
        self.labels["Feasible population ratio"].setText(f"{data['feasible_ratio']:.5g}" if "feasible_ratio" in data else "—")
        self.labels["Epsilon-feasible ratio"].setText(f"{data['epsilon_feasible_ratio']:.5g}" if "epsilon_feasible_ratio" in data else "—")
        self.labels["Adaptive epsilon"].setText(f"{data['epsilon']:.5g}" if "epsilon" in data else "—")
        components = data.get("constraint_components") or {}
        for key, label in {
            "bus_voltage": "Bus-voltage CV",
            "generator_q": "Generator-Q CV",
            "generator_p": "Generator-P CV",
            "branch_thermal": "Branch-thermal CV",
        }.items():
            value = components.get(key)
            self.labels[label].setText(f"{float(value):.5g}" if value is not None else "—")
        first = data.get("first_feasible_evaluation")
        self.labels["Evaluations to first feasibility"].setText(str(first) if first is not None else "Not reached")
        self.labels["Reward"].setText(f"{data['reward']:.5g}" if "reward" in data else "—")

    def update_progress(self, data: dict) -> None:
        if data.get("phase") in {"run_completed", "run_failed"} or "evaluations" not in data:
            return
        algorithm = str(data.get("algorithm", "—"))
        run_index = int(data.get("run_index", 1))
        evaluations = int(data.get("evaluations", 0) or 0)
        bucket = self._bucket(run_index)
        self._run_latest_telemetry[run_index] = dict(data)

        feasible_best = data.get("best_feasible_objective")
        if isinstance(feasible_best, (int, float)) and math.isfinite(float(feasible_best)):
            self._append_point(bucket["objective"], algorithm, evaluations, float(feasible_best))
        violation = data.get("best_constraint_violation")
        if isinstance(violation, (int, float)) and math.isfinite(float(violation)):
            self._append_point(bucket["violation"], algorithm, evaluations, float(violation))

        components = data.get("constraint_components") or {}
        for key, label in {
            "bus_voltage": "Bus voltage",
            "generator_q": "Generator Q",
            "generator_p": "Generator P",
            "branch_thermal": "Branch thermal",
            "power_flow": "Power flow",
        }.items():
            value = components.get(key)
            if value is not None:
                self._append_point(bucket["constraint"], f"{algorithm} · {label}", evaluations, float(value))
        if "feasible_ratio" in data:
            self._append_point(bucket["feasibility"], f"{algorithm} · Exact feasible ratio", evaluations, float(data["feasible_ratio"]))
        if "epsilon_feasible_ratio" in data:
            self._append_point(bucket["feasibility"], f"{algorithm} · Epsilon-feasible ratio", evaluations, float(data["epsilon_feasible_ratio"]))
        if "diversity" in data:
            self._append_point(bucket["diversity"], f"{algorithm} · Population diversity", evaluations, float(data["diversity"]))
        if "elite_diversity" in data:
            self._append_point(bucket["diversity"], f"{algorithm} · Elite diversity", evaluations, float(data["elite_diversity"]))
        for operator_name, rate in (data.get("operator_success_rates") or {}).items():
            self._append_point(bucket["operator_success"], f"{algorithm} · {operator_name}", evaluations, float(rate))

        operator = str(data.get("calo_operator", "")).strip()
        if operator and operator != "—":
            counts = self._run_operator_counts.setdefault(run_index, {}).setdefault(algorithm, Counter())
            counts[operator] += 1
            total = max(1, sum(counts.values()))
            for operator_name, count in counts.items():
                self._append_point(bucket["operator_usage"], f"{algorithm} · {operator_name}", evaluations, count / total)

        regime = str(data.get("calo_regime", "")).strip()
        if regime and regime != "—":
            if regime not in self._regime_codes:
                self._regime_codes[regime] = len(self._regime_codes) + 1
            self._append_point(bucket["regime"], algorithm, evaluations, float(self._regime_codes[regime]))

        selected = self.run_selector.currentData()
        if selected == self.AUTO_RUN_KEY:
            # Monotonic auto-follow prevents the canvas from bouncing backward when parallel jobs
            # interleave telemetry. Every run remains selectable and no telemetry is discarded.
            if self.current_run_index is None or run_index >= self.current_run_index:
                self._activate_run(run_index)
        elif int(selected) == run_index:
            self._activate_run(run_index)

        if self.current_run_index == run_index:
            self._display_telemetry(data)
            portfolio_key = self.portfolio_view.currentData()
            if portfolio_key in (self.LIVE_VIEW_KEY, *self.LIVE_PORTFOLIO_KEYS):
                self._redraw_plot()

    def _on_run_completed(self, _run_id: str, _algorithm: str, run_index: int) -> None:
        self._ensure_run_option(int(run_index))
        key = self.portfolio_view.currentData()
        if key not in (self.LIVE_VIEW_KEY, *self.LIVE_PORTFOLIO_KEYS):
            self._redraw_plot()

    def _on_run_failed(self, _failure_id: str, _algorithm: str, run_index: int) -> None:
        self._ensure_run_option(int(run_index))
        if self.run_selector.currentData() == self.AUTO_RUN_KEY:
            self.run_note.setText(
                f"Repeated run {run_index} reported a failed job. Other run telemetry remains available; inspect Experiment Manager for the failure record."
            )
