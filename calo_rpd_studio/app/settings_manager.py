"""Persistent desktop preferences."""

from PyQt6.QtCore import QSettings


class SettingsManager:
    def __init__(self):
        self.settings = QSettings("CALO-RPD", "CALO-RPD Studio")

    def value(self, key, default=None):
        return self.settings.value(key, default)

    def set_value(self, key, value):
        self.settings.setValue(key, value)
