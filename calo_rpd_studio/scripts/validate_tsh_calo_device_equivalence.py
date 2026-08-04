"""Validate one immutable TSH-CALO candidate's deterministic CPU/CUDA policy equivalence."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from calo_rpd_studio.ai.model_io import durable_write_bytes
from calo_rpd_studio.algorithms.calo.topology_context import (
    ScenarioDescriptor,
    TopologyAwarePolicyState,
    build_topology_graph_state,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_inference import (
    QualificationCandidateAuthority,
    TSHCALOInferenceController,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_policy import GroupActionMask
from calo_rpd_studio.algorithms.calo.tsh_calo_policy_artifact import inspect_tsh_calo_candidate
from calo_rpd_studio.algorithms.calo.tsh_calo_qualification_campaign import _candidate_binding
from calo_rpd_studio.algorithms.calo.tsh_calo_shield import (
    OODCalibration,
    SafetyEnvelope,
    SlidingWindowContextualBandit,
    ood_calibration_sha256,
    topology_ood_signature,
)
from calo_rpd_studio.compute.source_identity import resolve_source_identity
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ORPDVariableDecoder
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow
from calo_rpd_studio.power_system.case_loader import CaseLoader


DEVELOPMENT_CASES = ("case30", "case57")
EVIDENCE_SCHEMA = "tsh-calo-candidate-cpu-cuda-equivalence-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_sha256(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_new_json(path: str | Path, payload: dict) -> Path:
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite device-equivalence evidence: {destination}")
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


def _fit_development_calibration(states: list[TopologyAwarePolicyState]) -> OODCalibration:
    matrix = np.stack([topology_ood_signature(state) for state in states])
    mean = matrix.mean(axis=0)
    raw_scale = matrix.std(axis=0, ddof=1) if len(matrix) > 1 else np.zeros(matrix.shape[1])
    scale = np.where(raw_scale > 1e-8, raw_scale, 1.0)
    scores = np.sqrt(np.mean(np.square((matrix - mean) / scale), axis=1))
    calibration = OODCalibration(
        mean,
        scale,
        attenuation_start=max(1e-6, float(np.max(scores))),
        minimum_neural_weight=0.0,
    )
    calibration.validate()
    return calibration


def _maximum_absolute_difference(left, right) -> float:
    first = np.asarray(left)
    second = np.asarray(right)
    if first.shape != second.shape:
        return float("inf")
    if not first.size:
        return 0.0
    return float(np.max(np.abs(first.astype(float) - second.astype(float))))


def _numeric_comparison(left, right, *, rtol: float, atol: float) -> dict:
    first = np.asarray(left)
    second = np.asarray(right)
    same_shape = first.shape == second.shape
    finite = bool(
        same_shape
        and np.all(np.isfinite(first.astype(float)))
        and np.all(np.isfinite(second.astype(float)))
    )
    close = bool(
        finite and np.allclose(first.astype(float), second.astype(float), rtol=rtol, atol=atol)
    )
    return {
        "same_shape": same_shape,
        "finite": finite,
        "maximum_absolute_difference": _maximum_absolute_difference(first, second),
        "within_tolerance": close,
    }


def _tensor_values(value) -> np.ndarray:
    array: np.ndarray = value.detach().cpu().numpy()
    return array


def _compare_results(cpu_result, cuda_result, *, rtol: float, atol: float) -> dict:
    if cpu_result.fallback.disposition.value != "execute_policy":
        raise RuntimeError(f"CPU policy was not executable: {cpu_result.fallback.reason}")
    if cuda_result.fallback.disposition.value != "execute_policy":
        raise RuntimeError(f"CUDA policy was not executable: {cuda_result.fallback.reason}")
    required = (
        cpu_result.learner_operators,
        cpu_result.group_parameters,
        cpu_result.operator_probabilities,
        cpu_result.shield_trace,
        cuda_result.learner_operators,
        cuda_result.group_parameters,
        cuda_result.operator_probabilities,
        cuda_result.shield_trace,
    )
    if any(value is None for value in required):
        raise RuntimeError("Device-equivalence inference returned an incomplete policy result")
    cpu_trace = cpu_result.shield_trace
    cuda_trace = cuda_result.shield_trace
    assert cpu_trace is not None and cuda_trace is not None
    comparisons = {
        "group_parameters": _numeric_comparison(
            _tensor_values(cpu_result.group_parameters),
            _tensor_values(cuda_result.group_parameters),
            rtol=rtol,
            atol=atol,
        ),
        "operator_probabilities": _numeric_comparison(
            _tensor_values(cpu_result.operator_probabilities),
            _tensor_values(cuda_result.operator_probabilities),
            rtol=rtol,
            atol=atol,
        ),
        "shield_uncertainty": _numeric_comparison(
            _tensor_values(cpu_trace.uncertainty),
            _tensor_values(cuda_trace.uncertainty),
            rtol=rtol,
            atol=atol,
        ),
        "shield_mixture_weights": _numeric_comparison(
            _tensor_values(cpu_trace.mixture_weights),
            _tensor_values(cuda_trace.mixture_weights),
            rtol=rtol,
            atol=atol,
        ),
        "value_estimate": _numeric_comparison(
            [cpu_result.value_estimate],
            [cuda_result.value_estimate],
            rtol=rtol,
            atol=atol,
        ),
        "ood_score": _numeric_comparison(
            [cpu_trace.ood_score], [cuda_trace.ood_score], rtol=rtol, atol=atol
        ),
        "ood_attenuation": _numeric_comparison(
            [cpu_trace.ood_attenuation],
            [cuda_trace.ood_attenuation],
            rtol=rtol,
            atol=atol,
        ),
    }
    exact = {
        "regime": cpu_result.regime == cuda_result.regime,
        "learner_operators": bool(
            np.array_equal(
                _tensor_values(cpu_result.learner_operators),
                _tensor_values(cuda_result.learner_operators),
            )
        ),
        "shield_action_mask": bool(
            np.array_equal(
                _tensor_values(cpu_trace.action_mask),
                _tensor_values(cuda_trace.action_mask),
            )
        ),
        "intervention_reasons": cpu_trace.intervention_reasons == cuda_trace.intervention_reasons,
    }
    return {
        "exact": exact,
        "numeric": comparisons,
        "passed": all(exact.values())
        and all(item["within_tolerance"] for item in comparisons.values()),
    }


def _controller(
    binding: dict,
    calibration: OODCalibration,
    authority: QualificationCandidateAuthority,
    *,
    device: str,
    seed: int,
) -> TSHCALOInferenceController:
    return TSHCALOInferenceController(
        binding,
        ood_calibration=calibration,
        expected_ood_calibration_sha256=ood_calibration_sha256(calibration),
        deterministic=True,
        seed=seed,
        requested_device=device,
        allow_cpu_fallback=False,
        baseline_fallback_permitted=False,
        _qualification_authority=authority,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--case", action="append", choices=DEVELOPMENT_CASES, dest="cases")
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    cases = tuple(args.cases or DEVELOPMENT_CASES)
    if len(set(cases)) != len(cases):
        raise ValueError("Device-equivalence cases must be unique")
    if not args.run_id.strip():
        raise ValueError("Device-equivalence evidence requires a run ID")
    if args.rtol < 0.0 or args.atol < 0.0:
        raise ValueError("Device-equivalence tolerances cannot be negative")

    source = resolve_source_identity(require_durable=True)
    artifact = inspect_tsh_calo_candidate(
        args.candidate, expected_sha256=str(args.candidate_sha256).lower()
    )
    if artifact.artifact_kind != "ensemble_policy" or artifact.ensemble_size < 2:
        raise ValueError("Device equivalence requires an immutable TSH-CALO ensemble")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Physical CUDA is required for candidate device equivalence")
    device_index = int(torch.cuda.current_device())
    started_at = _utc_now()
    states = [_build_policy_state(case_name) for case_name in cases]
    calibration = _fit_development_calibration(states)
    design = {
        "schema_version": EVIDENCE_SCHEMA,
        "run_id": args.run_id.strip(),
        "source_commit": source.source_commit,
        "source_policy_sha256": artifact.sha256,
        "development_cases": list(cases),
        "seed": int(args.seed),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "deterministic": True,
    }
    design_sha = _canonical_sha256(design)
    authority = QualificationCandidateAuthority(
        args.run_id.strip(),
        design_sha,
        artifact.sha256,
        source.source_commit,
        cases,
        ood_calibration_sha256(calibration),
    )
    binding = _candidate_binding(artifact, calibration)
    torch.cuda.synchronize(device_index)
    allocated_before = int(torch.cuda.memory_allocated(device_index))
    torch.cuda.reset_peak_memory_stats(device_index)
    cpu_controller = _controller(binding, calibration, authority, device="cpu", seed=int(args.seed))
    cuda_controller = _controller(
        binding, calibration, authority, device=f"cuda:{device_index}", seed=int(args.seed)
    )
    torch.cuda.synchronize(device_index)
    allocated_after_load = int(torch.cuda.memory_allocated(device_index))
    records = []
    for case_name, state in zip(cases, states, strict=True):
        learner_groups = np.asarray(state.topology.control_groups, dtype=int)
        learner_contexts: np.ndarray = np.arange(len(learner_groups), dtype=int) % 4
        mask = GroupActionMask.from_control_groups(learner_groups)
        safety = SafetyEnvelope(len(learner_groups), len(learner_groups), True)
        cpu_result = cpu_controller.decide(
            state,
            mask,
            learner_groups,
            learner_contexts,
            bandit=SlidingWindowContextualBandit(32, 0.35),
            safety=safety,
        )
        cuda_result = cuda_controller.decide(
            state,
            mask,
            learner_groups,
            learner_contexts,
            bandit=SlidingWindowContextualBandit(32, 0.35),
            safety=safety,
        )
        torch.cuda.synchronize(device_index)
        records.append(
            {
                "case": case_name,
                "state_signature_sha256": hashlib.sha256(
                    np.ascontiguousarray(topology_ood_signature(state), dtype=np.float64).tobytes()
                ).hexdigest(),
                "comparison": _compare_results(
                    cpu_result,
                    cuda_result,
                    rtol=float(args.rtol),
                    atol=float(args.atol),
                ),
                "cpu_provenance": cpu_result.provenance,
                "cuda_provenance": cuda_result.provenance,
            }
        )
    peak_allocated = int(torch.cuda.max_memory_allocated(device_index))
    cuda_admission = cuda_controller.admission
    dedicated_vram = bool(
        cuda_admission is not None
        and str(cuda_admission.selected_device).startswith("cuda")
        and cuda_admission.computation_device == "nvidia_gpu"
        and allocated_after_load > allocated_before
        and peak_allocated >= allocated_after_load
        and all(
            parameter.device.type == "cuda"
            for network in cuda_controller.networks
            for parameter in network.parameters()
        )
    )
    passed = bool(
        dedicated_vram and records and all(row["comparison"]["passed"] for row in records)
    )
    evidence = {
        **design,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "source_identity_kind": source.source_identity_kind,
        "tracked_source_clean": source.tracked_source_clean,
        "scientific_design_sha256": design_sha,
        "ood_calibration_sha256": ood_calibration_sha256(calibration),
        "candidate": asdict(artifact),
        "runtime": {
            "torch": str(torch.__version__),
            "torch_cuda_runtime": str(torch.version.cuda or ""),
            "device": f"cuda:{device_index}",
            "device_name": str(torch.cuda.get_device_name(device_index)),
            "allocated_before_bytes": allocated_before,
            "allocated_after_load_bytes": allocated_after_load,
            "peak_allocated_bytes": peak_allocated,
            "cuda_admission": asdict(cuda_admission) if cuda_admission is not None else {},
        },
        "records": records,
        "dedicated_vram_execution_verified": dedicated_vram,
        "equivalence_passed": passed,
        "protected_cases_opened": False,
        "candidate_exported": False,
        "policy_registered": False,
        "policy_qualified": False,
        "policy_activated": False,
        "claim_scope": (
            "deterministic candidate policy CPU/CUDA numerical equivalence on listed development states"
            if passed
            else "no candidate CPU/CUDA equivalence claim"
        ),
    }
    destination = _write_new_json(args.output, evidence)
    print(f"evidence_path={destination}")
    print(json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
