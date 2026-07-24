"""Trusted exact CALO optimizer-state checkpoints for v5 run continuation."""

from __future__ import annotations

from pathlib import Path

from calo_rpd_studio.ai.model_io import durable_trusted_torch_save, load_trusted_resume

FORMAT = "calo_exact_run_checkpoint_v5"


def save_exact_run_checkpoint(path: str | Path, payload: dict) -> str:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    wrapped = {"format": FORMAT, **dict(payload)}
    # v6.5 publishes checkpoint bytes + HMAC trust metadata as one atomically replaced envelope.
    # The external .sha256 sidecar remains compatibility metadata only and is no longer a trust
    # dependency, eliminating the old checkpoint/sidecar publication window.
    return durable_trusted_torch_save(wrapped, destination)


def load_exact_run_checkpoint(path: str | Path) -> dict:
    payload = load_trusted_resume(Path(path).expanduser().resolve(), map_location="cpu")
    if str(payload.get("format", "")) != FORMAT:
        raise ValueError("Unsupported CALO exact-run checkpoint format")
    return payload
