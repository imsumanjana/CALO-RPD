"""Accounted command-line boundary for independent TSH-CALO training.

The established command parser and lifecycle runner remain in ``_train_tsh_calo_core``.  This public
entry point preserves every command and exit code while adding the canonical training/learning-
health/total counted-work split to machine-readable events and human-inspectable JSON summaries.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path

from calo_rpd_studio.algorithms.calo.tsh_calo_evaluation_accounting import (
    plan_training_evaluation_accounting,
)

from . import _train_tsh_calo_core as _core


for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)

_core_main = _core.main
_core_load_plan = _core.load_plan
_MISSING = object()


def _valid_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _manifest_accounting(payload: dict) -> dict | None:
    path = str(payload.get("manifest_path", "") or "").strip()
    if not path:
        return None
    source = Path(path).expanduser()
    if not source.is_file():
        return None
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    accounting = manifest.get("evaluation_accounting")
    return dict(accounting) if isinstance(accounting, dict) else None


def _selected_accounting(accounting: dict) -> tuple[dict, bool] | None:
    selected = accounting.get("cumulative")
    cumulative = isinstance(selected, dict)
    if not cumulative:
        selected = accounting
    if not isinstance(selected, dict):
        return None
    values = {
        "training_candidate_evaluations": selected.get(
            "training_candidate_evaluations"
        ),
        "generalization_guard_candidate_evaluations": selected.get(
            "generalization_guard_candidate_evaluations"
        ),
        "total_counted_candidate_evaluations": selected.get(
            "total_counted_candidate_evaluations"
        ),
    }
    if not all(_valid_count(value) for value in values.values()):
        return None
    if values["total_counted_candidate_evaluations"] != (
        values["training_candidate_evaluations"]
        + values["generalization_guard_candidate_evaluations"]
    ):
        return None
    return ({key: int(value) for key, value in values.items()}, cumulative)


def _augment_output_payload(payload: dict, plan) -> dict:
    result = dict(payload)
    segment = plan_training_evaluation_accounting(plan).to_dict()
    manifest = _manifest_accounting(result)
    selected = _selected_accounting(manifest) if manifest is not None else None
    if selected is None:
        accounting = segment
        values = dict(segment)
        cumulative = False
    else:
        accounting = manifest
        values, cumulative = selected
    result["evaluation_accounting"] = accounting
    result.update(
        {
            "total_training_candidate_evaluations": values[
                "training_candidate_evaluations"
            ],
            "total_generalization_guard_candidate_evaluations": values[
                "generalization_guard_candidate_evaluations"
            ],
            "total_counted_candidate_evaluations": values[
                "total_counted_candidate_evaluations"
            ],
            "legacy_candidate_evaluation_fields_are_training_only": True,
        }
    )
    if cumulative:
        result.update(
            {
                "cumulative_training_candidate_evaluations": values[
                    "training_candidate_evaluations"
                ],
                "cumulative_generalization_guard_candidate_evaluations": values[
                    "generalization_guard_candidate_evaluations"
                ],
                "cumulative_total_counted_candidate_evaluations": values[
                    "total_counted_candidate_evaluations"
                ],
            }
        )
    return result


def _accounted_output_text(text: str, plan) -> str:
    """Augment one supported JSON output line without changing non-JSON diagnostics."""

    if plan is None or not isinstance(text, str):
        return text
    prefix = ""
    encoded = text
    if text.startswith(_core.TRAINING_EVENT_PREFIX):
        prefix = _core.TRAINING_EVENT_PREFIX
        encoded = text[len(prefix) :]
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict):
        return text
    return prefix + json.dumps(
        _augment_output_payload(payload, plan),
        sort_keys=True,
        allow_nan=False,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the original command with an accounted, deterministic output boundary."""

    holder = {"plan": None}

    def capturing_load_plan(*args, **kwargs):
        plan = _core_load_plan(*args, **kwargs)
        holder["plan"] = plan
        return plan

    def accounted_print(*args, **kwargs):
        values = list(args)
        if len(values) == 1 and isinstance(values[0], str):
            values[0] = _accounted_output_text(values[0], holder["plan"])
        return builtins.print(*values, **kwargs)

    previous_load_plan = _core.load_plan
    previous_print = _core.__dict__.get("print", _MISSING)
    _core.load_plan = capturing_load_plan
    _core.print = accounted_print
    try:
        return int(_core_main(argv))
    finally:
        _core.load_plan = previous_load_plan
        if previous_print is _MISSING:
            delattr(_core, "print")
        else:
            _core.print = previous_print


__all__ = tuple(
    sorted(
        name
        for name in globals()
        if not name.startswith("_")
        and name not in {"Path", "annotations", "builtins", "json"}
    )
)


if __name__ == "__main__":
    raise SystemExit(main())
