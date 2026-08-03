"""Immutable, versioned candidate artifacts for independently trained TSH-CALO policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import torch

from calo_rpd_studio.ai.model_io import (
    checkpoint_sha256,
    durable_torch_save,
    load_checkpoint,
)
from calo_rpd_studio.power_system.case_identity import protected_holdout_matches

from .policy_schema import POLICY_STATE_DIM, infer_checkpoint_schema
from .tsh_calo_policy import TSHCALOPolicyNetwork
from .tsh_calo_schema import (
    DEFAULT_TSH_CALO_FEATURES,
    TSH_CALO_ACTION_SCHEMA,
    TSH_CALO_ALGORITHM_ID,
    TSH_CALO_ALGORITHM_VERSION,
    TSH_CALO_POLICY_ARCHITECTURE,
    TSH_CALO_STATE_SCHEMA,
    TSH_CALO_TRAINING_ENVIRONMENT,
    TSHCALOFeatureFlags,
)


@dataclass(frozen=True, slots=True)
class IndependentTrainingProvenance:
    """Minimum immutable provenance that a TSH-CALO candidate must disclose."""

    training_run_id: str
    training_design_sha256: str
    source_commit: str
    development_cases: tuple[str, ...]
    seed_manifest_sha256: str
    source_kind: str = "independent_policy_training"

    def validate(self) -> None:
        if self.source_kind != "independent_policy_training":
            raise ValueError("TSH-CALO candidates must originate from independent policy training")
        if not self.training_run_id.strip() or not self.source_commit.strip():
            raise ValueError("TSH-CALO training provenance requires a run ID and source commit")
        for label, digest in (
            ("training design", self.training_design_sha256),
            ("seed manifest", self.seed_manifest_sha256),
        ):
            text = str(digest).strip().lower()
            if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
                raise ValueError(f"TSH-CALO {label} SHA-256 is invalid")
        if not self.development_cases:
            raise ValueError("TSH-CALO training provenance requires development-case identities")
        leaked = protected_holdout_matches(self.development_cases)
        if leaked:
            raise ValueError(
                "Protected holdout cases cannot enter TSH-CALO training: " + ", ".join(leaked)
            )


@dataclass(frozen=True, slots=True)
class TSHCALOCandidateArtifact:
    path: str
    sha256: str
    algorithm_id: str
    algorithm_version: str
    state_schema_version: str
    action_schema_version: str
    training_environment_version: str
    feature_flags: dict[str, bool]
    training_provenance: dict


def _validated_feature_flags(
    feature_flags: TSHCALOFeatureFlags | None,
) -> TSHCALOFeatureFlags:
    flags = feature_flags or DEFAULT_TSH_CALO_FEATURES
    flags.validate()
    return flags


def build_tsh_calo_candidate_payload(
    network: TSHCALOPolicyNetwork,
    provenance: IndependentTrainingProvenance,
    *,
    feature_flags: TSHCALOFeatureFlags | None = None,
) -> dict:
    """Build a portable candidate payload without qualifying or activating it."""

    if not isinstance(network, TSHCALOPolicyNetwork):
        raise TypeError("TSH-CALO candidate export requires TSHCALOPolicyNetwork")
    provenance.validate()
    flags = _validated_feature_flags(feature_flags)
    return {
        "model_state_dict": network.state_dict(),
        "architecture": {
            "input_dim": POLICY_STATE_DIM,
            "hidden_dim": network.hidden_dim,
            "graph_steps": network.topology_encoder.message_passing_steps,
        },
        "metadata": {
            "algorithm_id": TSH_CALO_ALGORITHM_ID,
            "runtime_architecture_version": TSH_CALO_ALGORITHM_VERSION,
            "policy_architecture_version": TSH_CALO_POLICY_ARCHITECTURE,
            "state_dimension": POLICY_STATE_DIM,
            "state_schema_version": TSH_CALO_STATE_SCHEMA,
            "action_schema_version": TSH_CALO_ACTION_SCHEMA,
            "training_environment_version": TSH_CALO_TRAINING_ENVIRONMENT,
            "lifecycle_status": "candidate_unqualified",
            "feature_flags": asdict(flags),
            "training_provenance": asdict(provenance),
        },
    }


def save_tsh_calo_candidate(
    path: str | Path,
    network: TSHCALOPolicyNetwork,
    provenance: IndependentTrainingProvenance,
    *,
    feature_flags: TSHCALOFeatureFlags | None = None,
) -> TSHCALOCandidateArtifact:
    """Atomically publish an immutable candidate; registration remains a separate action."""

    target = Path(path).expanduser().resolve()
    payload = build_tsh_calo_candidate_payload(network, provenance, feature_flags=feature_flags)
    durable_torch_save(payload, target)
    return inspect_tsh_calo_candidate(target)


def inspect_tsh_calo_candidate(
    path: str | Path, *, expected_sha256: str | None = None
) -> TSHCALOCandidateArtifact:
    source = Path(path).expanduser().resolve()
    payload = load_checkpoint(source, expected_sha256=expected_sha256, map_location="cpu")
    schema = infer_checkpoint_schema(payload)
    if not bool(schema.get("native_tsh_calo", False)):
        raise ValueError("Policy artifact does not declare the exact TSH-CALO candidate ABI")
    metadata = dict(payload.get("metadata", {}) or {})
    if str(metadata.get("lifecycle_status", "")) != "candidate_unqualified":
        raise ValueError("Portable TSH-CALO artifacts must be exported as unqualified candidates")
    flags = TSHCALOFeatureFlags(**dict(metadata.get("feature_flags", {}) or {}))
    flags.validate()
    provenance = IndependentTrainingProvenance(
        **dict(metadata.get("training_provenance", {}) or {})
    )
    provenance.validate()
    architecture = dict(payload.get("architecture", {}) or {})
    hidden_dim = int(architecture.get("hidden_dim", 0))
    graph_steps = int(architecture.get("graph_steps", 0))
    if hidden_dim < 8 or graph_steps < 1:
        raise ValueError("TSH-CALO candidate network architecture is invalid")
    return TSHCALOCandidateArtifact(
        path=str(source),
        sha256=checkpoint_sha256(source),
        algorithm_id=TSH_CALO_ALGORITHM_ID,
        algorithm_version=TSH_CALO_ALGORITHM_VERSION,
        state_schema_version=TSH_CALO_STATE_SCHEMA,
        action_schema_version=TSH_CALO_ACTION_SCHEMA,
        training_environment_version=TSH_CALO_TRAINING_ENVIRONMENT,
        feature_flags=asdict(flags),
        training_provenance=asdict(provenance),
    )


def load_tsh_calo_candidate(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[TSHCALOPolicyNetwork, TSHCALOCandidateArtifact]:
    """Integrity-check and reconstruct one immutable policy for inference only."""

    artifact = inspect_tsh_calo_candidate(path, expected_sha256=expected_sha256)
    payload = load_checkpoint(path, expected_sha256=expected_sha256, map_location=device)
    architecture = dict(payload["architecture"])
    network = TSHCALOPolicyNetwork(
        hidden_dim=int(architecture["hidden_dim"]),
        graph_steps=int(architecture["graph_steps"]),
    ).to(device)
    network.load_state_dict(payload["model_state_dict"], strict=True)
    network.eval()
    return network, artifact
