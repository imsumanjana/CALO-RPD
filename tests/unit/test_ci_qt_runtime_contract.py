from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
QT_JOBS = ("source", "compatibility", "headless-gui", "artifact")


@pytest.mark.parametrize("job_id", QT_JOBS)
def test_qt_jobs_install_linux_libraries_before_importing_or_testing(job_id):
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"][job_id]["steps"]
    runtime = next(step for step in steps if step.get("id") == "qt_runtime")
    imports = next(step for step in steps if step.get("id") == "qt_import")
    assert runtime["if"] == "runner.os == 'Linux'"
    assert runtime["shell"] == "bash"
    assert "apt-get update" in runtime["run"]
    assert "apt-get install --no-install-recommends -y" in runtime["run"]
    assert {"libegl1", "libgl1", "libxkbcommon0"} <= set(runtime["run"].split())
    assert "from PyQt6 import QtCore, QtGui, QtWidgets" in imports["run"]
    assert steps.index(runtime) < steps.index(imports)
    consumers = [
        index
        for index, step in enumerate(steps)
        if "pytest" in step.get("run", "") or "validate_packaged_gui" in step.get("run", "")
    ]
    assert consumers and steps.index(imports) < min(consumers)
    assert not runtime.get("continue-on-error", False)
    assert not imports.get("continue-on-error", False)


def test_compatibility_dependency_commands_propagate_native_failures():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["compatibility"]["steps"]
    install = next(
        step for step in steps if step["name"] == "Install CPU compatibility environment"
    )
    assert install["shell"] == "bash"
    assert "python -m pip check" in install["run"]
    assert not install.get("continue-on-error", False)
