"""Trusted exact CALO optimizer-state checkpoints for v5 run continuation."""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch

from calo_rpd_studio.ai.model_io import load_trusted_resume, write_trusted_resume_hash

FORMAT = "calo_exact_run_checkpoint_v5"


def save_exact_run_checkpoint(path: str | Path, payload: dict) -> str:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapped = {"format": FORMAT, **dict(payload)}
    with tempfile.NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    try:
        torch.save(wrapped, temporary)
        # Validate the serialized container before atomically promoting it over the latest resume state.
        torch.load(temporary, map_location="cpu", weights_only=False)
        temporary.replace(destination)
        return write_trusted_resume_hash(destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_exact_run_checkpoint(path: str | Path) -> dict:
    payload = load_trusted_resume(Path(path).expanduser().resolve(), map_location="cpu")
    if str(payload.get("format", "")) != FORMAT:
        raise ValueError("Unsupported CALO exact-run checkpoint format")
    return payload
