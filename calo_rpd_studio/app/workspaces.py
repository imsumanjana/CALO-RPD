"""Stable workspace identities and versioned migration for CALO-RPD Studio.

v6.2 persists stable workspace keys under schema 3.  Numerical indexes are treated only as
legacy presentation data and are never authoritative scientific/navigation identities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


WORKSPACE_SCHEMA_VERSION = 3
WORKSPACE_LAYOUT_ID = "calo_rpd_v620_policy_first_layout"


@dataclass(frozen=True, slots=True)
class WorkspaceSpec:
    key: str
    title: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class WorkspaceMigrationReport:
    source_schema: int
    target_schema: int
    source_identity: str
    target_key: str
    migrated: bool
    warning: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_schema": int(self.source_schema),
            "target_schema": int(self.target_schema),
            "source_identity": str(self.source_identity),
            "target_key": str(self.target_key),
            "migrated": bool(self.migrated),
            "warning": str(self.warning),
        }


WORKSPACE_SPECS: tuple[WorkspaceSpec, ...] = (
    WorkspaceSpec("dashboard", "Dashboard"),
    WorkspaceSpec("calo_intelligence", "CALO Intelligence"),
    WorkspaceSpec("power_system", "Power System"),
    WorkspaceSpec("orpd", "ORPD Formulation"),
    WorkspaceSpec("algorithms", "Algorithms"),
    WorkspaceSpec("portfolio", "Portfolio Manager"),
    WorkspaceSpec("scenarios", "Robust Scenarios"),
    WorkspaceSpec("experiment", "Experiment Manager"),
    WorkspaceSpec("live_optimization", "Live Optimization"),
    WorkspaceSpec("statistics", "Statistical Analysis"),
    WorkspaceSpec("results", "Results Explorer"),
    WorkspaceSpec("validation", "Validation & Audit"),
    WorkspaceSpec("publication", "Publication Export"),
    WorkspaceSpec("resume_center", "Resume Center"),
    WorkspaceSpec("settings", "Application Settings"),
    WorkspaceSpec("benchmark", "Benchmark & Evidence"),
)

WORKSPACE_KEYS = tuple(spec.key for spec in WORKSPACE_SPECS)
WORKSPACE_INDEX = {key: index for index, key in enumerate(WORKSPACE_KEYS)}
WORKSPACE_TITLE = {spec.key: spec.title for spec in WORKSPACE_SPECS}
WORKSPACES = [(spec.title, spec.description) for spec in WORKSPACE_SPECS]

# v5.9 positional layout. Used ONLY when restoring legacy persisted workspace_index values.
LEGACY_V59_INDEX_TO_KEY = {
    0: "dashboard",
    1: "power_system",
    2: "orpd",
    3: "algorithms",
    4: "portfolio",
    5: "calo_intelligence",
    6: "scenarios",
    7: "experiment",
    8: "live_optimization",
    9: "statistics",
    10: "results",
    11: "validation",
    12: "publication",
    13: "resume_center",
    14: "settings",
    15: "benchmark",
}

# Historical aliases from prototypes/docs.  They are accepted only during migration.
WORKSPACE_KEY_ALIASES = {
    "calo": "calo_intelligence",
    "intelligence": "calo_intelligence",
    "power": "power_system",
    "formulation": "orpd",
    "robust_scenarios": "scenarios",
    "experiment_manager": "experiment",
    "live": "live_optimization",
    "audit": "validation",
    "publication_export": "publication",
}


def workspace_key_for_index(index: int) -> str:
    try:
        return WORKSPACE_KEYS[int(index)]
    except (IndexError, TypeError, ValueError) as exc:
        raise KeyError(f"Unknown v6 workspace index: {index!r}") from exc


def workspace_index_for_key(key: str) -> int:
    canonical = WORKSPACE_KEY_ALIASES.get(str(key), str(key))
    try:
        return WORKSPACE_INDEX[canonical]
    except KeyError as exc:
        raise KeyError(f"Unknown workspace key: {key!r}") from exc


def migrate_legacy_workspace_index(index: int, *, source_schema: int | None = None) -> str:
    """Translate a persisted positional workspace to a stable key.

    Schema 2+ positional values refer to the reordered v6 presentation stack.  Schema 0/1 values
    refer to the v5.9 positional layout.  Schema 3 never needs an index for authoritative restore.
    """
    try:
        value = int(index)
    except (TypeError, ValueError) as exc:
        raise KeyError(f"Invalid legacy workspace index: {index!r}") from exc
    if source_schema is not None and int(source_schema) >= 2:
        return workspace_key_for_index(value)
    try:
        return LEGACY_V59_INDEX_TO_KEY[value]
    except KeyError as exc:
        raise KeyError(f"Unknown legacy v5.9 workspace index: {value}") from exc


def migrate_workspace_ui(ui: dict | None, *, fallback_key: str = "dashboard") -> tuple[dict, WorkspaceMigrationReport]:
    """Migrate any supported historical UI payload to the v6.2 schema-3 keyed contract.

    Invalid/unknown identities fail *conservatively* to Dashboard instead of unlocking a later
    scientific page.  Workflow gates are still re-evaluated by ``WorkflowManager`` after migration.
    """
    source = dict(ui or {})
    try:
        source_schema = int(source.get("workspace_schema_version", 0) or 0)
    except (TypeError, ValueError):
        source_schema = 0

    raw_key = str(source.get("workspace_key", "") or "").strip()
    migrated = source_schema != WORKSPACE_SCHEMA_VERSION
    warning = ""
    source_identity = raw_key or str(source.get("workspace_index", ""))

    if raw_key:
        key = WORKSPACE_KEY_ALIASES.get(raw_key, raw_key)
        if key not in WORKSPACE_INDEX:
            warning = f"Unknown historical workspace key {raw_key!r}; restored conservatively to {fallback_key!r}."
            key = fallback_key
            migrated = True
        elif key != raw_key:
            migrated = True
    else:
        try:
            key = migrate_legacy_workspace_index(
                int(source.get("workspace_index", 0) or 0), source_schema=source_schema
            )
            migrated = True
        except (KeyError, TypeError, ValueError):
            key = fallback_key
            warning = "Historical workspace identity could not be resolved; restored conservatively to Dashboard."
            migrated = True

    if key not in WORKSPACE_INDEX:
        key = fallback_key
        warning = warning or "Workspace migration produced an invalid key; restored to Dashboard."
        migrated = True

    migrated_ui = dict(source)
    migrated_ui.update(
        {
            "workspace_schema_version": WORKSPACE_SCHEMA_VERSION,
            "workspace_layout_id": WORKSPACE_LAYOUT_ID,
            "workspace_key": key,
            # Compatibility only; not authoritative.
            "workspace_index": workspace_index_for_key(key),
        }
    )
    report = WorkspaceMigrationReport(
        source_schema=source_schema,
        target_schema=WORKSPACE_SCHEMA_VERSION,
        source_identity=source_identity,
        target_key=key,
        migrated=migrated,
        warning=warning,
    )
    return migrated_ui, report
