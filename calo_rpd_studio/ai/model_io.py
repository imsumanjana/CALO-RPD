"""CALO policy model serialization helpers."""
from pathlib import Path
import hashlib,torch
def load_checkpoint(path):return torch.load(Path(path),map_location='cpu',weights_only=False)
def checkpoint_sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
