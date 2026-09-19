"""Run explicit engineering tests with disposable desktop settings and complete inventory."""

from __future__ import annotations
import os
from pathlib import Path
import runpy
import sys
import tempfile


def main():
    root = Path(__file__).resolve().parents[2]
    runtime = Path(tempfile.mkdtemp(prefix="calo-acceptance-state-"))
    for name in (
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    ):
        path = runtime / name.lower()
        path.mkdir()
        os.environ[name] = str(path)
    os.environ["CALO_ACCEPTANCE_STATE_ROOT"] = str(runtime)
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts/qa"))
    from PyQt6.QtCore import QSettings

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    for scope in (QSettings.Scope.UserScope, QSettings.Scope.SystemScope):
        QSettings.setPath(QSettings.Format.IniFormat, scope, str(runtime / "qt-settings"))
    sys.argv = ["pytest", *sys.argv[1:]]
    runpy.run_module("pytest", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
