from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace


def test_collection_guard_omits_named_and_parameterized_protected_scientific_tests():
    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "guard_fixture", root / "scripts/qa/omission_plugin.py"
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    notices = []
    config = SimpleNamespace(
        getoption=lambda option: True,
        hook=SimpleNamespace(pytest_deselected=lambda items: notices.extend(items)),
    )

    def item(path, name, parameters):
        return SimpleNamespace(
            path=root / path, nodeid=path + "::" + name, callspec=SimpleNamespace(params=parameters)
        )

    items = [
        item("tests/scientific/test_a.py", "test_solver[case118]", {"name": "case118"}),
        item("tests/scientific/test_a.py", "test_case300_reference_bus", {}),
        item("tests/scientific/test_a.py", "test_solver[case30]", {"name": "case30"}),
        item("tests/unit/test_guard.py", "test_case118_rejected_without_loading", {}),
    ]
    plugin.pytest_collection_modifyitems(config, items)
    assert len(items) == len(notices) == 2
    assert [x.nodeid for x in items] == [
        "tests/scientific/test_a.py::test_solver[case30]",
        "tests/unit/test_guard.py::test_case118_rejected_without_loading",
    ]
    assert all(
        x._calo_omission_reason == "protected-case scientific execution not authorized"
        for x in notices
    )


def test_collection_inventory_is_durable_before_test_execution(tmp_path):
    import json

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "inventory_fixture", root / "scripts/qa/omission_plugin.py"
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    destination = tmp_path / "inventory.json"
    config = SimpleNamespace(getoption=lambda option: str(destination))
    plugin.pytest_configure(config)
    plugin.pytest_collection_finish(
        SimpleNamespace(items=[SimpleNamespace(nodeid="tests/a.py::test_one")])
    )
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert data["exit_code"] is None
    assert data["selected"] == data["not_executed"] == ["tests/a.py::test_one"]
    assert data["outcomes"] == {}


def test_partial_inventory_retries_transient_replace_denial_and_preserves_content(
    tmp_path, monkeypatch
):
    import json

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "retry_fixture", root / "scripts/qa/omission_plugin.py"
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    destination = tmp_path / "inventory.json"
    plugin.pytest_configure(SimpleNamespace(getoption=lambda option: str(destination)))
    original = plugin.os.replace
    attempts = []

    def temporarily_denied(source, target):
        attempts.append((source, target))
        if len(attempts) < 3:
            raise PermissionError("Synthetic transient report reader")
        return original(source, target)

    monkeypatch.setattr(plugin.os, "replace", temporarily_denied)
    monkeypatch.setattr(plugin.time, "sleep", lambda delay: None)
    plugin.pytest_collection_finish(
        SimpleNamespace(items=[SimpleNamespace(nodeid="tests/a.py::test_one")])
    )
    assert len(attempts) == 3
    assert json.loads(destination.read_text(encoding="utf-8"))["not_executed"] == [
        "tests/a.py::test_one"
    ]


def test_partial_inventory_does_not_ignore_permanent_replace_failure(tmp_path, monkeypatch):
    import pytest

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "permanent_fixture", root / "scripts/qa/omission_plugin.py"
    )
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    destination = tmp_path / "inventory.json"
    plugin.pytest_configure(SimpleNamespace(getoption=lambda option: str(destination)))
    attempts = []

    def denied(source, target):
        attempts.append((source, target))
        raise PermissionError("Synthetic persistent report failure")

    monkeypatch.setattr(plugin.os, "replace", denied)
    monkeypatch.setattr(plugin.time, "sleep", lambda delay: None)
    with pytest.raises(PermissionError):
        plugin.pytest_collection_finish(
            SimpleNamespace(items=[SimpleNamespace(nodeid="tests/a.py::test_one")])
        )
    assert len(attempts) == 8
    assert destination.with_suffix(".json.tmp").exists()
