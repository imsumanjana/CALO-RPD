"""Stable scientific fingerprints for exact result reuse and duplicate protection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


_ARTIFACT_KEYWORDS = (
    "checkpoint",
    "repository_file",
    "repository_path",
    "case_path",
    "custom_case",
    "training_snapshot",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_artifact_paths(value, key: str = ""):
    if isinstance(value, dict):
        return {str(k): _replace_artifact_paths(v, str(k)) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_artifact_paths(v, key) for v in value]
    if isinstance(value, str) and any(token in key.lower() for token in _ARTIFACT_KEYWORDS):
        path = Path(value).expanduser()
        if path.is_file():
            return {"artifact_name": path.name, "sha256": _file_sha256(path)}
    return value


def _normalise(value):
    if isinstance(value, dict):
        return {
            str(k): _normalise(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalise(v) for v in value]
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, float):
        return format(value, ".17g")
    return value


def stable_sha256(payload: dict) -> str:
    encoded = json.dumps(_normalise(payload), separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def experiment_fingerprint(config) -> str:
    data = config.to_dict()
    # Operational settings do not change the mathematical experiment and therefore must not
    # prevent scientifically exact reuse.
    for key in (
        "output_directory",
        "parallel_workers",
        "execution_backend",
        "gpu_utilization_target",
        "cpu_utilization_target",
        "gpu_memory_limit",
        "gpu_parallel_jobs",
        "xpu_utilization_target",
        "xpu_memory_limit",
        "xpu_parallel_jobs",
        "system_memory_limit",
        "cuda_task_share",
        "xpu_task_share",
        "cpu_task_share",
        "strict_device_shares",
        "runtime_compute_device",
        "throughput_profile_path",
        "telemetry_iteration_interval",
        "buffered_trace_writes",
        "resume_campaign_id",
        "portfolio",
        "portfolio_id",
        "resume_enabled",
        "checkpoint_interval_evaluations",
        "safe_pause",
        "reuse_compatible_results",
        "extension_experiment_id",
        "experiment_revision_id",
        "extension_mode",
        "extension_publication_eligible",
        "extension_run_indices",
        "extension_algorithm_names",
        "extension_execution_strategy",
        "extension_source_horizon",
        "require_exact_run_checkpoint_for_horizon_extension",
        "run_checkpoint_root",
        "extension_checkpoint_paths",
        "extension_existing_run_ids",
    ):
        data.pop(key, None)
    data.pop("algorithms", None)
    data.pop("runs", None)
    return stable_sha256(_replace_artifact_paths(data))


def run_fingerprint(config, algorithm: str, run_index: int, seeds) -> str:
    return stable_sha256(
        {
            "experiment": experiment_fingerprint(config),
            "algorithm": algorithm,
            "algorithm_parameters": dict(config.algorithm_parameters.get(algorithm, {})),
            "run_index": int(run_index),
            "seeds": {
                "algorithm_seed": int(seeds.algorithm_seed),
                "scenario_seed": int(seeds.scenario_seed),
                "ai_inference_seed": int(seeds.ai_inference_seed),
            },
        }
    )
