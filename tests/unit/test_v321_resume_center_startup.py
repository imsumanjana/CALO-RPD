from __future__ import annotations

import ast
from pathlib import Path


def test_resume_center_constructor_does_not_reference_undefined_manager_name():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "calo_rpd_studio"
        / "gui"
        / "panels"
        / "resume_center_panel.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    panel = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ResumeCenterPanel"
    )
    init = next(
        node for node in panel.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    undefined_manager_loads = [
        node
        for node in ast.walk(init)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "manager"
    ]
    assert undefined_manager_loads == []
