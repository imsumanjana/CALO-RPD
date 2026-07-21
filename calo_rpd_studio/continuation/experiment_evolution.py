"""Scientifically explicit experiment extension/revision workflows for CALO-RPD v5."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


class ExtensionProtocol(str, Enum):
    ALL_PAIRED = "all_paired"
    DETERMINISTIC_SUBSET = "deterministic_subset"
    MANUAL_EXPLORATORY = "manual_exploratory"

    @property
    def publication_eligible(self) -> bool:
        return self in {self.ALL_PAIRED, self.DETERMINISTIC_SUBSET}


@dataclass(frozen=True, slots=True)
class ExtensionPlan:
    experiment_id: str
    revision_id: str
    revision_number: int
    run_target: int
    evaluation_target: int
    extension_mode: str
    publication_eligible: bool
    protocol: dict


class ExperimentEvolutionService:
    """Create immutable execution-horizon revisions without overwriting earlier evidence."""

    def __init__(self, database) -> None:
        self.database = database

    def _base_config(self, experiment_id: str) -> ExperimentConfig:
        experiment = self.database.get_experiment(experiment_id)
        if experiment is None:
            raise KeyError(f"Unknown experiment: {experiment_id}")
        return ExperimentConfig.from_dict(json.loads(experiment["config_json"]))

    def ensure_original_revision(self, experiment_id: str) -> dict:
        existing = self.database.list_experiment_revisions(experiment_id)
        if existing:
            return existing[0]
        config = self._base_config(experiment_id)
        return self.database.create_experiment_revision(
            experiment_id,
            run_target=int(config.runs),
            evaluation_target=int(config.budget.max_evaluations),
            extension_mode="original",
            publication_eligible=True,
            protocol={"source": "original experiment definition"},
            status="completed"
            if str(self.database.get_experiment(experiment_id).get("campaign_status", ""))
            == "completed"
            else "running",
        )

    def extend_run_count(
        self, experiment_id: str, new_total_runs: int
    ) -> tuple[ExtensionPlan, ExperimentConfig]:
        original = self.ensure_original_revision(experiment_id)
        revisions = self.database.list_experiment_revisions(experiment_id)
        config = self._base_config(experiment_id)
        # New independent runs join the latest completed publication-eligible evidence horizon.
        # A post-hoc exploratory long run must never silently redefine the primary comparison horizon.
        eligible = [
            row
            for row in revisions
            if bool(row.get("publication_eligible")) and str(row.get("status")) == "completed"
        ]
        primary = eligible[-1] if eligible else original
        current_target = (
            max(int(row["run_target"]) for row in revisions) if revisions else int(config.runs)
        )
        current_target = max(current_target, int(config.runs))
        if int(new_total_runs) <= current_target:
            raise ValueError(f"New run target must exceed the current target ({current_target})")
        config.runs = int(new_total_runs)
        config.budget.max_evaluations = int(primary["evaluation_target"])
        config.max_iterations = max(int(config.max_iterations), int(primary["evaluation_target"]))
        row = self.database.create_experiment_revision(
            experiment_id,
            run_target=int(new_total_runs),
            evaluation_target=int(primary["evaluation_target"]),
            extension_mode="increase_run_count",
            publication_eligible=True,
            protocol={
                "paired_seed_extension": True,
                "preserve_existing_run_indices": True,
                "new_run_indices": [current_target, int(new_total_runs) - 1],
                "source_horizon": int(primary["evaluation_target"]),
            },
            # Run-count growth extends the current primary evidence branch. A newer post-hoc
            # exploratory revision must never become the scientific parent merely because it was
            # created later in wall-clock time.
            parent_revision_id=str(primary["id"]),
            status="planned",
        )
        config.extension_experiment_id = str(experiment_id)
        config.experiment_revision_id = str(row["id"])
        config.extension_mode = "increase_run_count"
        config.extension_publication_eligible = True
        return self._plan(row), config

    def extend_evaluation_horizon(
        self,
        experiment_id: str,
        new_evaluation_target: int,
        *,
        protocol: ExtensionProtocol = ExtensionProtocol.ALL_PAIRED,
        run_indices: tuple[int, ...] = (),
        algorithm_names: tuple[str, ...] = (),
        execution_strategy: str = "exact_continue",
        source_horizon: int | None = None,
    ) -> tuple[ExtensionPlan, ExperimentConfig]:
        self.ensure_original_revision(experiment_id)
        revisions = self.database.list_experiment_revisions(experiment_id)
        latest = revisions[-1]
        completed_revisions = [
            row for row in revisions if str(row.get("status", "")) == "completed"
        ]
        latest_completed = completed_revisions[-1] if completed_revisions else revisions[0]
        strategy = str(execution_strategy or "exact_continue").strip().lower()
        if strategy not in {"exact_continue", "recompute_from_seed"}:
            raise ValueError(f"Unsupported horizon-extension execution strategy: {strategy}")
        if int(new_evaluation_target) <= 0:
            raise ValueError("Extended evaluation target must be positive")
        # Post-hoc exploratory horizons do not redefine the primary comparison horizon. A later
        # all-paired revision may therefore legitimately target (for example) 10k FE even if one
        # exploratory run was previously continued to 20k FE.
        eligible_completed = [
            row
            for row in revisions
            if bool(row.get("publication_eligible")) and str(row.get("status")) == "completed"
        ]
        primary = eligible_completed[-1] if eligible_completed else revisions[0]
        if protocol.publication_eligible and int(new_evaluation_target) <= int(
            primary["evaluation_target"]
        ):
            raise ValueError(
                f"Publication-eligible evaluation target must exceed the current primary horizon ({int(primary['evaluation_target'])})"
            )
        if protocol is ExtensionProtocol.DETERMINISTIC_SUBSET and not run_indices:
            raise ValueError("A deterministic subset extension requires predeclared run indices")
        base_config = self._base_config(experiment_id)
        all_algorithms = {str(name) for name in base_config.algorithms}
        selected_algorithms = (
            {str(name) for name in algorithm_names} if algorithm_names else set(all_algorithms)
        )
        if (
            protocol in {ExtensionProtocol.ALL_PAIRED, ExtensionProtocol.DETERMINISTIC_SUBSET}
            and selected_algorithms != all_algorithms
        ):
            raise ValueError(
                "Publication-eligible horizon extension must include every algorithm in the paired experiment. "
                "Use manual_exploratory for a post-hoc algorithm subset."
            )
        if protocol is ExtensionProtocol.ALL_PAIRED and run_indices:
            raise ValueError(
                "all_paired horizon extension must include every paired run; do not provide a run subset"
            )
        publication_eligible = protocol.publication_eligible
        if strategy == "exact_continue":
            resolved_source = int(
                source_horizon
                or (
                    primary["evaluation_target"]
                    if publication_eligible
                    else latest_completed["evaluation_target"]
                )
            )
            if resolved_source <= 0:
                raise ValueError("Exact continuation requires a positive preserved source horizon")
            if int(new_evaluation_target) <= resolved_source:
                raise ValueError(
                    f"Exact continuation target {int(new_evaluation_target)} FE must exceed source horizon {resolved_source} FE"
                )
        else:
            resolved_source = 0
        protocol_payload = {
            "protocol": protocol.value,
            "run_indices": [int(i) for i in run_indices],
            "algorithms": sorted(selected_algorithms),
            "selection_timing": "predeclared" if publication_eligible else "post_hoc_manual",
            "primary_statistics_eligible": publication_eligible,
            "execution_strategy": strategy,
            "source_horizon": int(resolved_source),
            "trajectory_semantics": (
                "exact optimizer-state continuation from the recorded checkpoint"
                if strategy == "exact_continue"
                else "paired rerun from the original seed under the new horizon; not the same trajectory segment"
            ),
            "warning": "Manual post-hoc selective extension is exploratory only"
            if not publication_eligible
            else "",
        }
        if publication_eligible:
            parent_revision_id = str(primary["id"])
        elif strategy == "exact_continue":
            source_revisions = [
                candidate
                for candidate in revisions
                if int(candidate.get("evaluation_target", 0)) == int(resolved_source)
                and str(candidate.get("status", "")) == "completed"
            ]
            parent_revision_id = str((source_revisions[-1] if source_revisions else latest)["id"])
        else:
            parent_revision_id = str(latest["id"])
        target_run_count = int(
            primary["run_target"] if publication_eligible else latest_completed["run_target"]
        )
        row = self.database.create_experiment_revision(
            experiment_id,
            run_target=target_run_count,
            evaluation_target=int(new_evaluation_target),
            extension_mode="extend_evaluation_horizon",
            publication_eligible=publication_eligible,
            protocol=protocol_payload,
            parent_revision_id=parent_revision_id,
            status="planned",
        )
        config = self._base_config(experiment_id)
        config.runs = target_run_count
        config.budget.max_evaluations = int(new_evaluation_target)
        # max_iterations is a safety ceiling, not additional objective budget. Raise it only enough
        # to avoid preventing a legitimate FE-horizon continuation from entering its new segment.
        config.max_iterations = max(int(config.max_iterations), int(new_evaluation_target))
        config.extension_experiment_id = str(experiment_id)
        config.experiment_revision_id = str(row["id"])
        config.extension_mode = "extend_evaluation_horizon"
        config.extension_publication_eligible = bool(publication_eligible)
        config.extension_run_indices = [int(i) for i in run_indices]
        config.extension_algorithm_names = sorted(selected_algorithms)
        config.extension_execution_strategy = strategy
        config.extension_source_horizon = int(resolved_source)
        config.require_exact_run_checkpoint_for_horizon_extension = strategy == "exact_continue"
        # Exact continuation requires a run checkpoint. A caller must not silently replay with a
        # changed budget because adaptive algorithms may depend on the configured horizon.
        return self._plan(row), config

    @staticmethod
    def _plan(row: dict) -> ExtensionPlan:
        return ExtensionPlan(
            experiment_id=str(row["experiment_id"]),
            revision_id=str(row["id"]),
            revision_number=int(row["revision_number"]),
            run_target=int(row["run_target"]),
            evaluation_target=int(row["evaluation_target"]),
            extension_mode=str(row["extension_mode"]),
            publication_eligible=bool(row["publication_eligible"]),
            protocol=dict(row.get("protocol", {})),
        )
