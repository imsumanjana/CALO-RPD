"""Generate the current experiment-configuration JSON schema from serialized defaults.

The loader remains the authority for semantic and cross-field validation.  This generator keeps
the structural schema synchronized with every field emitted by ``ExperimentConfig.to_dict()`` and
adds the important scalar constraints that external editors can validate before loading.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESTINATION = (
    ROOT / "calo_rpd_studio" / "data" / "schemas" / "experiment_config.schema.json"
)


def _infer(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": ["number", "null"]}
    if isinstance(value, bool):
        return {"type": "boolean", "default": value}
    if isinstance(value, int):
        return {"type": "integer", "default": value}
    if isinstance(value, float):
        return {"type": "number", "default": value}
    if isinstance(value, str):
        return {"type": "string", "default": value}
    if isinstance(value, list):
        item_schema = _infer(value[0]) if value else {}
        return {"type": "array", "default": value, "items": item_schema}
    if isinstance(value, dict):
        if not value:
            return {"type": "object", "default": {}}
        return {
            "type": "object",
            "properties": {str(key): _infer(item) for key, item in value.items()},
            "additionalProperties": False,
            "default": value,
        }
    raise TypeError(f"Unsupported schema default type: {type(value).__name__}")


def build_schema() -> dict[str, Any]:
    defaults = ExperimentConfig().to_dict()
    properties = {key: _infer(value) for key, value in defaults.items()}

    properties["case_name"].pop("default", None)
    properties["algorithms"].update(minItems=1)
    properties["runs"].update(minimum=1)
    properties["master_seed"].update(minimum=0)
    properties["population_size"].update(minimum=2)
    properties["parallel_workers"].update(minimum=1)
    properties["execution_backend"] = {
        "type": "string",
        "enum": ["cuda_preferred", "cpu_only"],
        "default": "cuda_preferred",
    }
    properties["execution_purpose"] = {
        "type": "string",
        "enum": ["exploratory", "formal"],
        "default": "exploratory",
    }
    properties["requested_compute_device"] = {
        "type": "string",
        "pattern": r"^(auto|cpu|cuda|cuda:[0-9]+)$",
        "default": "auto",
    }
    properties["cuda_vram_budget_fraction"] = {
        "type": "number",
        "const": 0.8,
        "default": 0.8,
        "description": "Fixed ceiling: 80% of VRAM free at the task admission boundary.",
    }
    properties["cuda_oom_retry_count"].update(minimum=0)
    properties["cuda_minimum_microbatch"].update(minimum=1)
    properties["cross_run_batch_window_ms"].update(exclusiveMinimum=0)
    properties["max_cross_run_batch"].update(minimum=1)
    properties["calibration_repetitions"].update(minimum=1)
    properties["telemetry_iteration_interval"].update(minimum=1)
    properties["checkpoint_interval_evaluations"].update(minimum=1)
    for reviewed_control_field in (
        "generator_voltage_buses",
        "transformer_branch_indices",
    ):
        properties["variables"]["properties"][reviewed_control_field] = {
            "type": ["array", "null"],
            "items": {"type": "integer", "minimum": 0},
            "uniqueItems": True,
            "default": None,
            "description": (
                "Explicit independently reviewed controls; null preserves the bundled IEEE "
                "profile and an empty array explicitly declares no controls of this kind."
            ),
        }

    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CALO-RPD Studio Experiment Configuration",
        "description": (
            "Current scientist configuration. Legacy execution-tuning and XPU fields are accepted "
            "only by the compatibility loader and are never emitted into a new configuration."
        ),
        "type": "object",
        "required": [
            "case_name",
            "algorithms",
            "runs",
            "master_seed",
            "population_size",
            "budget",
            "objective",
            "variables",
            "scenarios",
        ],
        "properties": properties,
        "additionalProperties": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--check", action="store_true")
    arguments = parser.parse_args()
    rendered = json.dumps(build_schema(), indent=2, ensure_ascii=False) + "\n"
    destination = arguments.output.resolve()
    if arguments.check:
        if not destination.is_file() or destination.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Experiment schema is stale: {destination}")
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
