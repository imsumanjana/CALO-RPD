"""Cryptographic freeze manifest for final CALO benchmark campaigns.

The manifest is intentionally explicit: final benchmark execution is allowed only when the
mathematical implementation, policy checkpoint, training repository snapshot, default CALO
hyperparameters, mixed-variable decoder, and feasibility rules match the frozen hashes.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable


FREEZE_SCHEMA_VERSION = 1
DEFAULT_FREEZE_RELATIVE_PATHS = (
    "calo_rpd_studio/accelerated/device.py",
    "calo_rpd_studio/accelerated/device_resident_orpd.py",
    "calo_rpd_studio/accelerated/torch_decoder.py",
    "calo_rpd_studio/accelerated/torch_power_flow.py",
    "calo_rpd_studio/accelerated/torch_orpd.py",
    "calo_rpd_studio/accelerated/runtime_context.py",
    "calo_rpd_studio/accelerated/throughput_engine.py",
    "calo_rpd_studio/algorithms/base_optimizer.py",
    "calo_rpd_studio/algorithms/torch_suite.py",
    "calo_rpd_studio/algorithms/calo/ai_controller.py",
    "calo_rpd_studio/algorithms/calo/archives.py",
    "calo_rpd_studio/algorithms/calo/cognitive_state.py",
    "calo_rpd_studio/algorithms/calo/diagnostics.py",
    "calo_rpd_studio/algorithms/calo/diversity_manager.py",
    "calo_rpd_studio/algorithms/calo/environmental_selection.py",
    "calo_rpd_studio/algorithms/calo/learning_operators.py",
    "calo_rpd_studio/algorithms/calo/operator_credit.py",
    "calo_rpd_studio/algorithms/calo/optimizer.py",
    "calo_rpd_studio/algorithms/calo/policy_network.py",
    "calo_rpd_studio/algorithms/calo/recovery.py",
    "calo_rpd_studio/algorithms/calo/reward.py",
    "calo_rpd_studio/algorithms/calo/success_memory.py",
    "calo_rpd_studio/algorithms/calo/training.py",
    "calo_rpd_studio/algorithms/calo/heterogeneous_training.py",
    "calo_rpd_studio/algorithms/registry.py",
    "calo_rpd_studio/compute/persistent_accelerator_worker.py",
    "calo_rpd_studio/compute/persistent_accelerator_sidecar.py",
    "calo_rpd_studio/compute/persistent_training_actor.py",
    "calo_rpd_studio/compute/resource_scheduler.py",
    "calo_rpd_studio/compute/training_actor_worker.py",
    "calo_rpd_studio/experiments/experiment_config.py",
    "calo_rpd_studio/experiments/experiment_runner.py",
    "calo_rpd_studio/app/experiment_manager.py",
    "calo_rpd_studio/portfolio/models.py",
    "calo_rpd_studio/portfolio/catalog.py",
    "calo_rpd_studio/portfolio/planner.py",
    "calo_rpd_studio/portfolio/fingerprint.py",
    "calo_rpd_studio/portfolio/exporter.py",
    "calo_rpd_studio/resume/models.py",
    "calo_rpd_studio/resume/service.py",
    "calo_rpd_studio/results/database.py",
    "calo_rpd_studio/results/result_store.py",
    "calo_rpd_studio/orpd/constraint_violation.py",
    "calo_rpd_studio/orpd/constraints.py",
    "calo_rpd_studio/orpd/feasibility_rules.py",
    "calo_rpd_studio/orpd/mixed_variable_handler.py",
    "calo_rpd_studio/orpd/variable_decoder.py",
    "calo_rpd_studio/orpd/problem.py",
    "calo_rpd_studio/robustness/cvar.py",
    "calo_rpd_studio/robustness/scenario.py",
    "calo_rpd_studio/robustness/scenario_generator.py",
    "calo_rpd_studio/robustness/load_uncertainty.py",
    "calo_rpd_studio/robustness/renewable_uncertainty.py",
    "calo_rpd_studio/robustness/contingencies.py",
    "calo_rpd_studio/robustness/robust_objectives.py",
    "calo_rpd_studio/power_system/case_validation.py",
    "calo_rpd_studio/power_system/case_loader.py",
    "calo_rpd_studio/power_system/ybus.py",
    "calo_rpd_studio/power_system/ac_power_flow.py",
    "calo_rpd_studio/power_system/pv_pq_switching.py",
    "calo_rpd_studio/power_system/independent_validator.py",
    "calo_rpd_studio/results/publication_export.py",
    "calo_rpd_studio/benchmarking/campaign.py",
    "calo_rpd_studio/benchmarking/package.py",
    "calo_rpd_studio/benchmarking/evidence.py",
    "calo_rpd_studio/visualization/publication_evidence.py",
    "calo_rpd_studio/visualization/font_preflight.py",
    "calo_rpd_studio/ai/model_io.py",
    "calo_rpd_studio/gui/panels/experiment_manager_panel.py",
    "calo_rpd_studio/gui/panels/live_optimization_panel.py",
    "calo_rpd_studio/version.py",
    "calo_rpd_studio/data/trained_models/calo_policy_v2.json",
    "calo_rpd_studio/data/trained_models/calo_policy_v2.pt",
    "calo_rpd_studio/data/frozen/historical_training_snapshot_v2.json",
)


@dataclass(frozen=True, slots=True)
class FreezeVerification:
    passed: bool
    manifest_path: str
    checked_files: int
    missing_files: tuple[str, ...]
    changed_files: tuple[str, ...]
    manifest_sha256: str
    message: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_hash(payload: dict) -> str:
    clean = dict(payload)
    clean.pop("manifest_sha256", None)
    data = json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def project_root_from_module() -> Path:
    return Path(__file__).resolve().parents[2]


def create_freeze_manifest(
    destination: str | Path,
    *,
    project_root: str | Path | None = None,
    relative_paths: Iterable[str] = DEFAULT_FREEZE_RELATIVE_PATHS,
    software_version: str = "3.4.3",
    note: str = "CALO-RPD v3.4.3 publication-export completion, bounded reproducibility bundling, and NaN-safe statistics release frozen before final benchmark/test execution",
) -> Path:
    root = Path(project_root) if project_root is not None else project_root_from_module()
    root = root.resolve()
    files: dict[str, dict[str, object]] = {}
    missing: list[str] = []
    for relative in relative_paths:
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        files[relative] = {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
    if missing:
        raise FileNotFoundError("Cannot freeze CALO; required files are missing: " + ", ".join(missing))

    from calo_rpd_studio.algorithms.registry import SPECS

    payload = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "software_version": software_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "frozen_scope": {
            "mathematical_equations": True,
            "operator_definitions": True,
            "state_vector": True,
            "archive_rules": True,
            "ppo_architecture": True,
            "policy_checkpoint": True,
            "training_dataset_snapshot": True,
            "hyperparameters": True,
            "constraint_handling": True,
            "accelerator_power_flow": True,
            "accelerator_constraint_evaluator": True,
            "torch_canonical_baselines": True,
            "mixed_variable_tensor_decoder": True,
            "persistent_accelerator_workers": True,
            "cross_run_batching": True,
            "automatic_batch_calibration": True,
            "measured_throughput_scheduling": True,
            "persistent_policy_training_actors": True,
            "cross_episode_policy_rollout_batching": True,
            "portfolio_dependency_planning": True,
            "scientific_fingerprint_reuse": True,
            "campaign_resume_journal": True,
            "portfolio_artifact_resume": True,
            "device_resident_execution": True,
            "gpu_maximum_100_percent_cuda_when_available": True,
            "cuda_only_execution": True,
            "grouped_tensor_pv_pq_switching": True,
            "fractional_tail_weighted_cvar": True,
            "case_specific_formulation_profiles": True,
            "scenario_manifest_and_validation": True,
            "independent_q_limit_validation": True,
            "verified_feasible_publication_statistics": True,
            "verified_only_article_package": True,
            "asynchronous_fairness_audit": True,
            "fairness_audit_progress_and_cpu_throttling": True,
            "multi_run_live_telemetry": True,
            "progressive_portfolio_live_preview": True,
            "ieee300_matched_pypower_validation": True,
            "unclipped_reference_q_reporting": True,
            "bounded_reproducibility_bundle": True,
            "export_subprogress_and_safe_cancel": True,
            "asynchronous_standard_publication_export": True,
            "nan_safe_convergence_statistics": True,
            "safe_hashed_checkpoint_loading": True,
            "font_preflight_and_metadata": True,
        },
        "calo_default_parameters": SPECS["CALO"].default_parameters,
        "files": files,
        "benchmark_rule": "No CALO tuning is permitted after TEST campaign execution begins.",
    }
    payload["manifest_sha256"] = _canonical_json_hash(payload)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return destination


def verify_freeze_manifest(
    manifest_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> FreezeVerification:
    manifest = Path(manifest_path)
    if not manifest.is_file():
        return FreezeVerification(False, str(manifest), 0, (), (), "", "Freeze manifest does not exist.")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    expected_manifest_hash = str(payload.get("manifest_sha256", ""))
    actual_manifest_hash = _canonical_json_hash(payload)
    root = Path(project_root) if project_root is not None else project_root_from_module()
    root = root.resolve()
    missing: list[str] = []
    changed: list[str] = []
    files = payload.get("files", {})
    for relative, meta in files.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if _sha256(path) != str(meta.get("sha256", "")):
            changed.append(relative)
    manifest_valid = bool(expected_manifest_hash) and expected_manifest_hash == actual_manifest_hash
    passed = manifest_valid and not missing and not changed
    if not manifest_valid:
        message = "Freeze manifest integrity check failed."
    elif missing:
        message = f"Freeze verification failed: {len(missing)} frozen file(s) are missing."
    elif changed:
        message = f"Freeze verification failed: {len(changed)} frozen file(s) changed."
    else:
        message = f"Frozen CALO verified across {len(files)} files."
    return FreezeVerification(
        passed,
        str(manifest),
        len(files),
        tuple(missing),
        tuple(changed),
        actual_manifest_hash,
        message,
    )
