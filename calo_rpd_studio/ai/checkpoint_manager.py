"""Versioned CALO checkpoint discovery."""
from pathlib import Path
class CheckpointManager:
    def __init__(self,directory):self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
    def list(self):return sorted(self.directory.glob('*.pt'))
    def latest(self):
        files=self.list();return files[-1] if files else None
