"""Application bootstrap."""

from __future__ import annotations

import logging
import sys

from PyQt6.QtWidgets import QApplication

from calo_rpd_studio.gui.themes.theme_manager import apply_theme

from .experiment_manager import ExperimentManager
from .main_window import MainWindow
from .settings_manager import SettingsManager
from .state_manager import AppState


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("CALO-RPD Studio")
    app.setOrganizationName("CALO-RPD")
    from calo_rpd_studio.version import VERSION

    app.setApplicationVersion(VERSION)

    settings = SettingsManager()
    state = AppState()
    requested_theme = str(settings.value("appearance", "light"))
    state.theme = apply_theme(app, requested_theme)

    manager = ExperimentManager(state)
    window = MainWindow(state, manager, settings)

    def on_theme_changed(name: str) -> None:
        state.theme = apply_theme(app, name)

    state.theme_changed.connect(on_theme_changed)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
