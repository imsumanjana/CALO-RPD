"""Measure the independent TSH-CALO ten-epoch CUDA update boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from calo_rpd_studio.accelerated.cuda_timing import (
    CUDA_NUMERICAL_TIME_SHARE_TARGET,
    POLICY_EPOCHS_PER_BOUNDARY,
    measure_cuda_window,
)
from calo_rpd_studio.ai.model_io import durable_write_bytes
from calo_rpd_studio.algorithms.calo.topology_context import (
    ScenarioDescriptor,
    TopologyAwarePolicyState,
    build_topology_graph_state,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_policy import GroupActionMask
from calo_rpd_studio.algorithms.calo.tsh_calo_training import (
    IndependentTSHCALOTrainer,
    TSHCALORolloutBatch,
    TSHCALOTrainingConfig,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_training_resources import (
    TSHCALOTrainingResourceEnvelope,
)
from calo_rpd_studio.compute.source_identity import resolve_source_identity
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ORPDVariableDecoder
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.power_system.case_loader import CaseLoader


DEVELOPMENT_CASES = ("case30", "case57")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_new_json(path: str | Path, payload: dict) -> Path:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite CUDA policy evidence: {destination}")
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    durable_write_bytes(destination, encoded)
    return destination


def _build_policy_state(case_name: str) -> TopologyAwarePolicyState:
    case = CaseLoader.load(case_name)
    decoder = ORPDVariableDecoder(case, ORPDVariableConfig())
    power_flow = run_ac_power_flow(case)
    if not power_flow.converged:
        raise RuntimeError(f"Development-case setup power flow did not converge for {case_name}")
    topology = build_topology_graph_state(
        case,
        decoder,
        np.full(decoder.dimension, 0.5),
        power_flow,
        [ScenarioDescriptor(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)],
    )
    return TopologyAwarePolicyState(np.linspace(0.0, 1.0, 32), topology)


def _build_rollout(trainer: IndependentTSHCALOTrainer, state: TopologyAwarePolicyState):
    groups = np.asarray(state.topology.control_groups, dtype=int)
    contexts = np.arange(len(groups), dtype=int) % 4
    mask = GroupActionMask.from_control_groups(groups)
    first, first_logp, first_value = trainer.sample_action(
        state, mask, groups, contexts, deterministic=True
    )
    second, second_logp, second_value = trainer.sample_action(
        state, mask, groups, contexts, deterministic=True
    )
    return TSHCALORolloutBatch(
        states=(state, state),
        actions=(first, second),
        old_log_probabilities=np.asarray([first_logp, second_logp]),
        old_values=np.asarray([first_value, second_value]),
        advantages=np.asarray([1.0, -0.4]),
        returns=np.asarray([0.8, -0.2]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=DEVELOPMENT_CASES, default="case30")
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--warmup-updates", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.updates < 1 or args.warmup_updates < 0:
        raise ValueError("Update count must be positive and warmup count must be non-negative")
    if not args.run_id.strip():
        raise ValueError("CUDA policy evidence requires a non-empty run ID")

    source = resolve_source_identity(require_durable=True)
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Physical CUDA is required for CUDA policy qualification")
    started_at = _utc_now()
    state = _build_policy_state(args.case)
    topology = state.topology
    population_size = max(2, int(topology.control_features.shape[0]))
    envelope = TSHCALOTrainingResourceEnvelope(
        rollout_capacity=2,
        maximum_population_size=population_size,
        maximum_topology_nodes=int(topology.node_features.shape[0]),
        maximum_topology_edges=int(topology.edge_index.shape[1]),
        maximum_topology_controls=int(topology.control_features.shape[0]),
        maximum_scenarios=int(topology.scenario_features.shape[0]),
    )
    seed_manifest = hashlib.sha256(
        f"{args.run_id}:{args.case}:{args.seed}".encode("utf-8")
    ).hexdigest()
    config = TSHCALOTrainingConfig(
        training_run_id=f"cuda-policy-hot-path:{args.run_id.strip()}",
        development_cases=(args.case,),
        seed_manifest_sha256=seed_manifest,
        resource_envelope=envelope,
        seed=int(args.seed),
        ppo_epochs=POLICY_EPOCHS_PER_BOUNDARY,
        device="cuda",
        allow_cpu_fallback=False,
    )
    trainer = IndependentTSHCALOTrainer(config)
    try:
        if trainer.device.type != "cuda":
            raise RuntimeError(f"Policy trainer resolved to unexpected device {trainer.device}")
        batch = _build_rollout(trainer, state)
        for _ in range(int(args.warmup_updates)):
            trainer.update(batch)
        device_index = (
            int(trainer.device.index)
            if trainer.device.index is not None
            else int(torch.cuda.current_device())
        )
        torch.cuda.synchronize(device_index)
        torch.cuda.reset_peak_memory_stats(device_index)

        def _run_updates():
            metrics = None
            for _ in range(int(args.updates)):
                metrics = trainer.update(batch)
            return metrics

        metrics, timing = measure_cuda_window(
            _run_updates,
            device=str(trainer.device),
            label="tsh_calo_ten_epoch_updates",
        )
        if metrics is None:
            raise RuntimeError("CUDA policy hot-path window returned no update metrics")
        provenance = trainer.device_provenance()
        admission = provenance["memory_admission"]
        model_on_cuda = all(
            parameter.device.type == "cuda" for parameter in trainer.network.parameters()
        )
        allocated = int(torch.cuda.memory_allocated(device_index))
        peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
        process_ceiling = int(admission["process_ceiling_bytes"])
        dedicated_vram_verified = bool(
            model_on_cuda
            and allocated > 0
            and peak_allocated >= allocated
            and peak_allocated <= process_ceiling
        )
        boundary_verified = bool(
            config.ppo_epochs == POLICY_EPOCHS_PER_BOUNDARY
            and provenance["per_ppo_epoch_cpu_metric_transfer"] is False
            and provenance["ppo_host_synchronization"]
            == "one packed metrics transfer after the complete configured PPO epoch block"
        )
        finite_metrics = all(np.isfinite(float(value)) for value in metrics.values())
        qualification_passed = bool(
            timing.target_met
            and dedicated_vram_verified
            and boundary_verified
            and finite_metrics
            and str(admission["selected_device"]).startswith("cuda")
            and not str(admission["fallback_reason"])
        )
        evidence = {
            "schema_version": "calo-rpd-cuda-policy-hot-path-v1",
            "run_id": args.run_id.strip(),
            "started_at": started_at,
            "completed_at": _utc_now(),
            "source_commit": source.source_commit,
            "source_identity_kind": source.source_identity_kind,
            "tracked_source_clean": source.tracked_source_clean,
            "parameters": {
                "case": args.case,
                "seed": int(args.seed),
                "ppo_epochs_per_update": int(config.ppo_epochs),
                "measured_updates": int(args.updates),
                "warmup_updates": int(args.warmup_updates),
                "measured_policy_epochs": int(args.updates) * int(config.ppo_epochs),
            },
            "runtime": {
                "torch": str(torch.__version__),
                "torch_cuda_runtime": str(torch.version.cuda or ""),
                "device": str(trainer.device),
                "device_name": str(torch.cuda.get_device_name(device_index)),
                "allocated_bytes": allocated,
                "peak_allocated_bytes": peak_allocated,
                "process_ceiling_bytes": process_ceiling,
            },
            "timing": timing.to_dict(),
            "training_device_provenance": provenance,
            "dedicated_vram_execution_verified": dedicated_vram_verified,
            "ten_epoch_boundary_verified": boundary_verified,
            "finite_metrics": finite_metrics,
            "qualification_passed": qualification_passed,
            "protected_cases_opened": False,
            "candidate_exported": False,
            "policy_registered": False,
            "policy_qualified": False,
            "policy_activated": False,
            "claim_scope": (
                "more than or equal to 95% CUDA event-time share for independent ten-epoch PPO "
                "update windows on this development case only"
                if qualification_passed
                else "no CUDA policy numerical-time-share qualification claim"
            ),
            "target_cuda_time_share": CUDA_NUMERICAL_TIME_SHARE_TARGET,
        }
    finally:
        trainer.close()
    destination = _write_new_json(args.output, evidence)
    print(f"evidence_path={destination}")
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    return 0 if qualification_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
