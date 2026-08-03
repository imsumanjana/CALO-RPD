"""Immutable ensemble inference, Safe-80 admission, shield, and explicit fallback for TSH-CALO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import psutil
import torch

from calo_rpd_studio.compute.memory_budget import calculate_available_memory_admission

from .tsh_calo_policy import GroupActionMask, hierarchical_action
from .tsh_calo_policy_artifact import load_tsh_calo_ensemble
from .tsh_calo_qualification import (
    TSH_CALO_QUALIFICATION_RECEIPT_KEY,
    load_tsh_calo_qualification_receipt,
)
from .tsh_calo_schema import (
    TSH_CALO_ACTION_SCHEMA,
    TSH_CALO_ALGORITHM_ID,
    TSH_CALO_ALGORITHM_VERSION,
    TSH_CALO_STATE_SCHEMA,
    TSH_CALO_TRAINING_ENVIRONMENT,
    TSHCALOFeatureFlags,
)
from .tsh_calo_shield import (
    OODCalibration,
    PolicyFallbackDecision,
    SafetyEnvelope,
    ShieldTrace,
    SlidingWindowContextualBandit,
    UncertaintySafetyShield,
    aggregate_policy_ensemble,
    ood_calibration_sha256,
    resolve_policy_fallback,
    topology_ood_signature,
)


FROZEN_CALO_BASELINE_IDENTITY = "CALO-v5.9"


@dataclass(frozen=True, slots=True)
class InferenceMemoryAdmission:
    requested_device: str
    selected_device: str
    computation_device: str
    estimated_working_set_bytes: int
    available_bytes_at_admission: int
    allowance_bytes: int
    fallback_reason: str = ""


@dataclass(slots=True)
class TSHCALOInferenceResult:
    fallback: PolicyFallbackDecision
    regime: int | None
    learner_operators: torch.Tensor | None
    group_parameters: torch.Tensor | None
    operator_probabilities: torch.Tensor | None
    value_estimate: float | None
    shield_trace: ShieldTrace | None
    provenance: dict


def _cpu_admission():
    memory = psutil.virtual_memory()
    return calculate_available_memory_admission(
        total_bytes=int(memory.total),
        available_bytes=int(memory.available),
        requested_fraction=0.80,
    )


def admit_inference_device(
    checkpoint_path: str | Path,
    *,
    requested_device: str = "auto",
    allow_cpu_fallback: bool = True,
) -> InferenceMemoryAdmission:
    """Admit a conservative serialized-model working set against current free memory."""

    source = Path(checkpoint_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    estimated = max(1 << 20, int(source.stat().st_size) * 3)
    requested = str(requested_device or "auto").strip().lower()
    if requested not in {"auto", "cpu", "cuda"} and not requested.startswith("cuda:"):
        raise ValueError("TSH-CALO inference device must be auto, cpu, cuda, or cuda:<index>")
    wants_cuda = requested == "auto" or requested.startswith("cuda")
    cuda_reason = ""
    if wants_cuda and torch.cuda.is_available():
        selected = torch.device("cuda:0" if requested in {"auto", "cuda"} else requested)
        free_bytes, total_bytes = torch.cuda.mem_get_info(selected)
        admission = calculate_available_memory_admission(
            total_bytes=int(total_bytes),
            available_bytes=int(free_bytes),
            requested_fraction=0.80,
        )
        if estimated <= admission.additional_allowance_bytes:
            return InferenceMemoryAdmission(
                requested,
                str(selected),
                "nvidia_gpu",
                estimated,
                admission.available_bytes_at_admission,
                admission.additional_allowance_bytes,
            )
        cuda_reason = "estimated working set exceeds 80% of currently free VRAM"
        if not allow_cpu_fallback:
            raise MemoryError(cuda_reason)
    elif wants_cuda and requested != "auto":
        if not allow_cpu_fallback:
            raise RuntimeError("CUDA inference was requested but CUDA is unavailable")
        cuda_reason = "CUDA unavailable"
    cpu = _cpu_admission()
    if estimated > cpu.additional_allowance_bytes:
        raise MemoryError("TSH-CALO inference working set exceeds 80% of currently available RAM")
    return InferenceMemoryAdmission(
        requested,
        "cpu",
        "cpu",
        estimated,
        cpu.available_bytes_at_admission,
        cpu.additional_allowance_bytes,
        cuda_reason,
    )


class TSHCALOInferenceController:
    """Consume one qualified, active, immutable ensemble binding or fail explicitly."""

    TRACE_SCHEMA = "tsh-calo-inference-trace-v1"

    def __init__(
        self,
        binding: dict,
        *,
        ood_calibration: OODCalibration,
        expected_ood_calibration_sha256: str,
        deterministic: bool,
        seed: int,
        requested_device: str = "auto",
        allow_cpu_fallback: bool = True,
        baseline_fallback_permitted: bool = False,
    ) -> None:
        self.binding = dict(binding or {})
        self.deterministic = bool(deterministic)
        self.baseline_fallback_permitted = bool(baseline_fallback_permitted)
        self.ood_calibration = ood_calibration
        self.expected_ood_calibration_sha256 = str(expected_ood_calibration_sha256).lower()
        self.networks = []
        self.artifact = None
        self.admission: InferenceMemoryAdmission | None = None
        self.rejection_reason = ""
        self.device = torch.device("cpu")
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        try:
            self._load(requested_device, allow_cpu_fallback, int(seed))
        except Exception as exc:
            self.rejection_reason = f"{type(exc).__name__}: {exc}"
            self.networks = []

    def _load(self, requested_device: str, allow_cpu_fallback: bool, seed: int) -> None:
        binding = self.binding
        expected = {
            "policy_algorithm_id": TSH_CALO_ALGORITHM_ID,
            "policy_architecture_version": TSH_CALO_ALGORITHM_VERSION,
            "policy_state_schema_version": TSH_CALO_STATE_SCHEMA,
            "policy_action_schema_version": TSH_CALO_ACTION_SCHEMA,
            "policy_training_environment_version": TSH_CALO_TRAINING_ENVIRONMENT,
        }
        if not bool(binding.get("strict_policy_binding", False)):
            raise ValueError("TSH-CALO requires a strict immutable policy binding")
        for key, value in expected.items():
            if str(binding.get(key, "")) != value:
                raise ValueError(f"TSH-CALO policy binding {key} is incompatible")
        if str(binding.get("policy_qualification_status", "")) != "qualified":
            raise ValueError("TSH-CALO runtime requires a qualified policy")
        if not bool(binding.get("policy_active_at_binding", False)):
            raise ValueError("TSH-CALO runtime requires an explicitly activated policy")
        if (
            str(binding.get("policy_artifact_kind", "")) != "ensemble_policy"
            or int(binding.get("policy_ensemble_size", 0)) < 2
        ):
            raise ValueError("TSH-CALO runtime requires an epistemic ensemble artifact")
        flags = TSHCALOFeatureFlags(**dict(binding.get("policy_feature_flags", {}) or {}))
        flags.validate()
        calibration_sha = ood_calibration_sha256(self.ood_calibration)
        if calibration_sha != self.expected_ood_calibration_sha256:
            raise ValueError("TSH-CALO OOD calibration SHA-256 mismatch")
        if str(binding.get("policy_ood_calibration_sha256", "")).lower() != calibration_sha:
            raise ValueError("TSH-CALO binding does not identify the frozen OOD calibration")
        receipt = load_tsh_calo_qualification_receipt(
            {
                TSH_CALO_QUALIFICATION_RECEIPT_KEY: dict(
                    binding.get("policy_qualification_receipt", {}) or {}
                )
            },
            expected_policy_sha256=str(binding.get("policy_sha256", "")),
        )
        if (
            str(binding.get("policy_qualification_receipt_sha256", "")).lower()
            != receipt.receipt_sha256
        ):
            raise ValueError("TSH-CALO binding qualification receipt SHA-256 mismatch")
        if receipt.ood_calibration_sha256 != calibration_sha:
            raise ValueError("TSH-CALO bound calibration differs from its qualification receipt")
        checkpoint = str(binding.get("policy_checkpoint", "") or "")
        expected_sha = str(binding.get("policy_sha256", "") or "").lower()
        self.admission = admit_inference_device(
            checkpoint,
            requested_device=requested_device,
            allow_cpu_fallback=allow_cpu_fallback,
        )
        self.device = torch.device(self.admission.selected_device)
        try:
            self.networks, self.artifact = load_tsh_calo_ensemble(
                checkpoint, expected_sha256=expected_sha, device=self.device
            )
        except torch.cuda.OutOfMemoryError:
            if self.device.type != "cuda" or not allow_cpu_fallback:
                raise
            torch.cuda.empty_cache()
            cpu = _cpu_admission()
            if self.admission.estimated_working_set_bytes > cpu.additional_allowance_bytes:
                raise MemoryError(
                    "CUDA OOM occurred and the working set cannot fit within 80% of available RAM"
                )
            self.admission = InferenceMemoryAdmission(
                self.admission.requested_device,
                "cpu",
                "cpu",
                self.admission.estimated_working_set_bytes,
                cpu.available_bytes_at_admission,
                cpu.additional_allowance_bytes,
                "CUDA allocation failed after VRAM admission",
            )
            self.device = torch.device("cpu")
            self.networks, self.artifact = load_tsh_calo_ensemble(
                checkpoint, expected_sha256=expected_sha, device=self.device
            )
        if self.artifact.ensemble_size != int(binding["policy_ensemble_size"]):
            raise ValueError("TSH-CALO bound ensemble size does not match the immutable artifact")
        if self.artifact.feature_flags != asdict(flags):
            raise ValueError("TSH-CALO bound feature flags do not match the immutable artifact")
        bound_members = list(binding.get("policy_ensemble_members", []) or [])
        artifact_members = list(self.artifact.training_provenance.get("members", []) or [])
        if json.dumps(bound_members, sort_keys=True, separators=(",", ":")) != json.dumps(
            artifact_members, sort_keys=True, separators=(",", ":")
        ):
            raise ValueError("TSH-CALO bound ensemble members do not match the immutable artifact")
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    def fallback_decision(self, reason: str = "") -> PolicyFallbackDecision:
        return resolve_policy_fallback(
            policy_usable=bool(self.networks) and not reason,
            rejection_reason=reason or self.rejection_reason,
            baseline_fallback_permitted=self.baseline_fallback_permitted,
            tsh_algorithm_identity=TSH_CALO_ALGORITHM_ID,
            frozen_baseline_identity=FROZEN_CALO_BASELINE_IDENTITY,
        )

    def decide(
        self,
        state,
        action_mask: GroupActionMask,
        learner_groups,
        learner_contexts,
        *,
        bandit: SlidingWindowContextualBandit,
        safety: SafetyEnvelope,
    ) -> TSHCALOInferenceResult:
        if not self.networks:
            fallback = self.fallback_decision()
            return TSHCALOInferenceResult(
                fallback, None, None, None, None, None, None, self._provenance()
            )
        try:
            state.validate()
            with torch.inference_mode():
                ensemble = aggregate_policy_ensemble([network(state) for network in self.networks])
                signature = topology_ood_signature(state)
                ood_score, attenuation = self.ood_calibration.score_and_attenuation(signature)
                action = hierarchical_action(
                    ensemble.mean_output,
                    action_mask,
                    deterministic=self.deterministic,
                    generator=None if self.deterministic else self.generator,
                )
                shield = UncertaintySafetyShield().resolve(
                    action=action,
                    disagreement=ensemble.disagreement,
                    ood_score=ood_score,
                    ood_attenuation=attenuation,
                    learner_groups=torch.as_tensor(
                        learner_groups, dtype=torch.long, device=self.device
                    ),
                    learner_contexts=torch.as_tensor(
                        learner_contexts, dtype=torch.long, device=self.device
                    ),
                    bandit=bandit,
                    safety=safety,
                )
                operators = UncertaintySafetyShield.sample(
                    shield,
                    deterministic=self.deterministic,
                    generator=None if self.deterministic else self.generator,
                )
            return TSHCALOInferenceResult(
                self.fallback_decision(),
                action.regime,
                operators,
                action.group_parameters,
                shield.probabilities,
                float(ensemble.mean_output.value.item()),
                shield.trace,
                self._provenance(),
            )
        except Exception as exc:
            reason = f"runtime policy rejection: {type(exc).__name__}: {exc}"
            return TSHCALOInferenceResult(
                self.fallback_decision(reason),
                None,
                None,
                None,
                None,
                None,
                None,
                self._provenance(runtime_rejection=reason),
            )

    def _provenance(self, *, runtime_rejection: str = "") -> dict:
        return {
            "schema_version": self.TRACE_SCHEMA,
            "policy_id": str(self.binding.get("policy_id", "")),
            "policy_sha256": str(self.binding.get("policy_sha256", "")),
            "policy_ensemble_size": int(self.binding.get("policy_ensemble_size", 0) or 0),
            "policy_ood_calibration_sha256": self.expected_ood_calibration_sha256,
            "device_admission": asdict(self.admission) if self.admission is not None else {},
            "runtime_rejection": runtime_rejection or self.rejection_reason,
        }
