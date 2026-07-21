"""Operational run-continuation bindings shared by every execution backend.

This module deliberately contains no scientific optimization logic.  It only maps the immutable
campaign/run identity to the exact optimizer checkpoint path selected by the experiment manager.
Keeping this mapping in one place prevents CPU, CUDA, persistent-device and XPU workers from
silently using different continuation semantics.
"""

from __future__ import annotations

from pathlib import Path


def bind_exact_run_checkpoint(config, item):
    """Bind CALO's exact checkpoint/resume paths for *item* and return ``config``.

    ``config`` is expected to be an already-copied per-job configuration.  Non-CALO items are
    returned unchanged.  Requested function-evaluation accounting is unaffected by this binding.
    """

    if item is None or str(getattr(item, "label", "")) != "CALO":
        return config

    parameters = dict(getattr(config, "algorithm_parameters", {}) or {})
    values = dict(parameters.get("CALO", {}) or {})
    root = str(getattr(config, "run_checkpoint_root", "") or "").strip()
    run_index = int(getattr(item, "run_index", 0))

    if root:
        safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(item.label))
        # A new experiment revision must never overwrite an older revision's exact checkpoint.
        # This preserves the ability to audit or continue an earlier evidence horizon later.
        revision_id = str(getattr(config, "experiment_revision_id", "") or "original_unversioned")
        checkpoint_path = Path(root) / revision_id / f"{safe_label}_run_{run_index:05d}.resume.pt"
        values["run_checkpoint_path"] = str(checkpoint_path)
        values["checkpoint_interval_evaluations"] = int(
            getattr(config, "checkpoint_interval_evaluations", 500)
        )

    resume_map = dict(getattr(config, "extension_checkpoint_paths", {}) or {})
    key = f"{item.label}:{run_index}"
    if key in resume_map:
        values["resume_run_checkpoint"] = str(resume_map[key])
        # Segment zero is the original execution.  A resumed/extended trajectory starts a new
        # provenance segment without pretending that the earlier horizon was planned identically.
        values["continuation_segment_index"] = max(
            1, int(values.get("continuation_segment_index", 1))
        )

    parameters["CALO"] = values
    config.algorithm_parameters = parameters
    return config
