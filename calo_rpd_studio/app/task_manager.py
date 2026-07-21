"""Shared Qt thread-pool service."""

from PyQt6.QtCore import QThreadPool


class TaskManager:
    def __init__(self):
        self.pool = QThreadPool.globalInstance()

    def active_count(self):
        return self.pool.activeThreadCount()
