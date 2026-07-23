"""Dependency-light structural validation of the v6.2 GUI/workflow contract.

This does not replace running the real PyQt6 GUI suite. It verifies source-level invariants even in
headless build environments where PyQt6 is unavailable.
"""

from __future__ import annotations

import ast
from pathlib import Path

from calo_rpd_studio.app.workspaces import WORKSPACE_KEYS, WORKSPACE_SCHEMA_VERSION


def validate_gui_contract(repo_root: str | Path) -> dict:
    root = Path(repo_root)
    main_path = root / "calo_rpd_studio" / "app" / "main_window.py"
    workflow_path = root / "calo_rpd_studio" / "app" / "workflow_manager.py"
    errors: list[str] = []
    warnings: list[str] = []

    main_text = main_path.read_text(encoding="utf-8")
    workflow_text = workflow_path.read_text(encoding="utf-8")
    ast.parse(main_text)
    ast.parse(workflow_text)

    expected_prefix = ("dashboard", "calo_intelligence", "power_system")
    if tuple(WORKSPACE_KEYS[:3]) != expected_prefix:
        errors.append(f"Canonical workspace prefix is {WORKSPACE_KEYS[:3]!r}, expected {expected_prefix!r}")
    if WORKSPACE_SCHEMA_VERSION < 3:
        errors.append("Workspace schema version is older than v6.2 schema 3")
    if '"workspace_schema_version": WORKSPACE_SCHEMA_VERSION' not in main_text:
        errors.append("MainWindow does not persist the canonical workspace schema version")
    if "SessionRecoveryJournal" not in main_text:
        errors.append("MainWindow does not integrate durable session recovery")
    if "policy_training_changed" not in main_text:
        errors.append("Global Training Exclusive Lock signal integration is missing")
    if "governing_policy" not in workflow_text.lower():
        warnings.append("Workflow source does not visibly reference governing-policy readiness")

    # Critical navigation logic must not dereference QStackedWidget pages by numeric ``pages[n]``.
    tree = ast.parse(main_text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if node.value.attr == "pages" and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                errors.append(f"Hard-coded numeric pages[{node.slice.value}] reference remains in MainWindow")

    return {"ok": not errors, "errors": errors, "warnings": warnings, "workspace_count": len(WORKSPACE_KEYS)}


__all__ = ["validate_gui_contract"]
