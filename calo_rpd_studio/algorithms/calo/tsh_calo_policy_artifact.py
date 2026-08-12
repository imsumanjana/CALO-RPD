"""Immutable, versioned candidate artifacts for independently trained TSH-CALO policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
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
from .tsh_calo_training_resources import validate_tsh_calo_training_device_provenance
from .tsh_calo_training_receipt import load_tsh_calo_training_episode_receipt


@dataclass(frozen=True, slots=True)
class IndependentTrainingProvenance:
    """Minimum immutable provenance that a TSH-CALO candidate must disclose."""

    training_run_id: str
    training_design_sha256: str
    source_commit: str
    development_cases: tuple[str, ...]
    seed_manifest_sha256: str
    training_device_provenance: dict
    training_episode_receipts: tuple[dict, ...]
    source_kind: str = "independent_policy_training"
    development_freeze_commit: str = ""
    development_freeze_sha256: str = ""
    phase4_acceptance_sha256: str = ""
    initialization_policy_sha256: str = ""

    def validate(self) -> None:
        if self.source_kind != "independent_policy_training":
            raise ValueError("TSH-CALO candidates must originate from independent policy training")
        if not self.training_run_id.strip() or not self.source_commit.strip():
            raise ValueError("TSH-CALO training provenance requires a run ID and source commit")
        if (
            self.development_freeze_commit
            or self.development_freeze_sha256
            or self.phase4_acceptance_sha256
            or self.initialization_policy_sha256
        ):
            normalized_freeze = str(self.development_freeze_commit).strip().lower()
            normalized_source = str(self.source_commit).strip().lower()
            if (
                len(normalized_freeze) != 40
                or any(character not in "0123456789abcdef" for character in normalized_freeze)
                or normalized_freeze != normalized_source
            ):
                raise ValueError(
                    "Post-development TSH-CALO training must bind the exact development-freeze commit"
                )
            if str(self.initialization_policy_sha256).strip():
                raise ValueError(
                    "Post-development TSH-CALO training cannot initialize from an old policy"
                )
            if not _is_sha256(self.development_freeze_sha256):
                raise ValueError(
                    "Post-development TSH-CALO training requires the retained development-freeze "
                    "payload SHA-256"
                )
            if not _is_sha256(self.phase4_acceptance_sha256):
                raise ValueError(
                    "Post-development TSH-CALO training requires the explicit Phase 4 acceptance "
                    "receipt SHA-256"
                )
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
        validate_tsh_calo_training_device_provenance(self.training_device_provenance)
        if not self.training_episode_receipts:
            raise ValueError("TSH-CALO candidate requires a completed counted training episode")
        receipts = [
            load_tsh_calo_training_episode_receipt(item) for item in self.training_episode_receipts
        ]
        if any(item.training_run_id != self.training_run_id for item in receipts):
            raise ValueError("TSH-CALO training episode belongs to another training run")
        if any(item.training_design_sha256 != self.training_design_sha256 for item in receipts):
            raise ValueError("TSH-CALO training episode design differs from the candidate")
        if any(item.case_identity not in self.development_cases for item in receipts):
            raise ValueError("TSH-CALO training episode case is undeclared")
        receipt_hashes = [item.receipt_sha256 for item in receipts]
        if len(set(receipt_hashes)) != len(receipt_hashes):
            raise ValueError("TSH-CALO candidate contains duplicate training episode receipts")
        session_ids = [item.session_id for item in receipts]
        if len(set(session_ids)) != len(session_ids):
            raise ValueError("TSH-CALO candidate contains duplicate training session IDs")


@dataclass(frozen=True, slots=True)
class TSHCALOCandidateArtifact:
    path: str
    sha256: str
    algorithm_id: str
    algorithm_version: str
    state_schema_version: str
    action_schema_version: str
    training_environment_version: str
    artifact_kind: str
    ensemble_size: int
    feature_flags: dict[str, bool]
    training_provenance: dict

    @property
    def post_development_eligible(self) -> bool:
        """Whether every member was initialized empty and trained on the frozen source."""

        provenance = dict(self.training_provenance or {})
        if provenance.get("source_kind") == "independent_policy_training_ensemble":
            rows = [
                dict(member.get("training_provenance", {}) or {})
                for member in list(provenance.get("members", []) or [])
            ]
            if len(rows) < 2:
                return False
        else:
            rows = [provenance]
        identities: set[tuple[str, str, str]] = set()
        for row in rows:
            source_commit = str(row.get("source_commit", "")).strip().lower()
            freeze_commit = str(row.get("development_freeze_commit", "")).strip().lower()
            freeze_sha256 = str(row.get("development_freeze_sha256", "")).strip().lower()
            acceptance_sha256 = str(row.get("phase4_acceptance_sha256", "")).strip().lower()
            if (
                len(source_commit) != 40
                or any(character not in "0123456789abcdef" for character in source_commit)
                or freeze_commit != source_commit
                or not _is_sha256(freeze_sha256)
                or not _is_sha256(acceptance_sha256)
                or str(row.get("initialization_policy_sha256", "")).strip()
            ):
                return False
            identities.add((source_commit, freeze_sha256, acceptance_sha256))
        return len(identities) == 1


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
            "artifact_kind": "single_policy_member",
            "ensemble_size": 1,
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
    artifact_kind = str(metadata.get("artifact_kind", "single_policy_member"))
    ensemble_size = int(metadata.get("ensemble_size", 1))
    if artifact_kind == "single_policy_member":
        if ensemble_size != 1:
            raise ValueError("Single TSH-CALO candidate must declare ensemble_size=1")
        provenance = IndependentTrainingProvenance(
            **dict(metadata.get("training_provenance", {}) or {})
        )
        provenance.validate()
        training_provenance = asdict(provenance)
    elif artifact_kind == "ensemble_policy":
        members = list(metadata.get("ensemble_members", []) or [])
        state_dicts = list(payload.get("ensemble_model_state_dicts", []) or [])
        if ensemble_size < 2 or len(members) != ensemble_size or len(state_dicts) != ensemble_size:
            raise ValueError("TSH-CALO ensemble artifact has inconsistent member cardinality")
        source_hashes: list[str] = []
        training_run_ids: list[str] = []
        for member in members:
            if not _is_sha256(str(member.get("source_candidate_sha256", ""))):
                raise ValueError("TSH-CALO ensemble member SHA-256 is invalid")
            provenance = IndependentTrainingProvenance(
                **dict(member.get("training_provenance", {}) or {})
            )
            provenance.validate()
            source_hashes.append(str(member["source_candidate_sha256"]))
            training_run_ids.append(provenance.training_run_id)
        if len(set(source_hashes)) != len(source_hashes):
            raise ValueError("TSH-CALO ensemble cannot duplicate a source candidate")
        if len(set(training_run_ids)) != len(training_run_ids):
            raise ValueError("TSH-CALO ensemble members require independent training-run IDs")
        training_provenance = {
            "source_kind": "independent_policy_training_ensemble",
            "members": members,
        }
    else:
        raise ValueError("Unknown TSH-CALO candidate artifact kind")
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
        artifact_kind=artifact_kind,
        ensemble_size=ensemble_size,
        feature_flags=asdict(flags),
        training_provenance=training_provenance,
    )


def load_tsh_calo_candidate(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[TSHCALOPolicyNetwork, TSHCALOCandidateArtifact]:
    """Integrity-check and reconstruct one immutable policy for inference only."""

    artifact = inspect_tsh_calo_candidate(path, expected_sha256=expected_sha256)
    if artifact.artifact_kind != "single_policy_member":
        raise ValueError("Use load_tsh_calo_ensemble for an ensemble policy artifact")
    payload = load_checkpoint(path, expected_sha256=expected_sha256, map_location=device)
    architecture = dict(payload["architecture"])
    network = TSHCALOPolicyNetwork(
        hidden_dim=int(architecture["hidden_dim"]),
        graph_steps=int(architecture["graph_steps"]),
    ).to(device)
    network.load_state_dict(payload["model_state_dict"], strict=True)
    network.eval()
    return network, artifact


def _is_sha256(value: str) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def assemble_tsh_calo_ensemble_candidate(
    path: str | Path,
    members: Sequence[tuple[str | Path, str]],
) -> TSHCALOCandidateArtifact:
    """Assemble independently trained members without qualifying or activating the ensemble."""

    if len(members) < 2:
        raise ValueError("TSH-CALO epistemic ensemble requires at least two policy members")
    payloads: list[dict] = []
    artifacts: list[TSHCALOCandidateArtifact] = []
    for member_path, expected_sha256 in members:
        artifact = inspect_tsh_calo_candidate(member_path, expected_sha256=expected_sha256)
        if artifact.artifact_kind != "single_policy_member":
            raise ValueError("TSH-CALO ensembles can only be assembled from single policy members")
        artifacts.append(artifact)
        payloads.append(
            load_checkpoint(member_path, expected_sha256=expected_sha256, map_location="cpu")
        )
    source_hashes = [artifact.sha256 for artifact in artifacts]
    training_run_ids = [
        str(artifact.training_provenance.get("training_run_id", "")) for artifact in artifacts
    ]
    if len(set(source_hashes)) != len(source_hashes):
        raise ValueError("TSH-CALO ensemble cannot duplicate a source candidate")
    if len(set(training_run_ids)) != len(training_run_ids):
        raise ValueError("TSH-CALO ensemble members require independent training-run IDs")
    architecture = dict(payloads[0].get("architecture", {}) or {})
    feature_flags = dict(artifacts[0].feature_flags)
    for payload, artifact in zip(payloads[1:], artifacts[1:]):
        if dict(payload.get("architecture", {}) or {}) != architecture:
            raise ValueError("TSH-CALO ensemble members must use the same network architecture")
        if artifact.feature_flags != feature_flags:
            raise ValueError("TSH-CALO ensemble members must use the same feature flags")
    metadata = dict(payloads[0].get("metadata", {}) or {})
    metadata.update(
        {
            "artifact_kind": "ensemble_policy",
            "ensemble_size": len(artifacts),
            "lifecycle_status": "candidate_unqualified",
            "ensemble_members": [
                {
                    "source_candidate_sha256": artifact.sha256,
                    "training_provenance": dict(artifact.training_provenance),
                }
                for artifact in artifacts
            ],
        }
    )
    metadata.pop("training_provenance", None)
    state_dicts = [payload["model_state_dict"] for payload in payloads]
    target = Path(path).expanduser().resolve()
    durable_torch_save(
        {
            "model_state_dict": state_dicts[0],
            "ensemble_model_state_dicts": state_dicts,
            "architecture": architecture,
            "metadata": metadata,
        },
        target,
    )
    return inspect_tsh_calo_candidate(target)


def load_tsh_calo_ensemble(
    path: str | Path,
    *,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> tuple[list[TSHCALOPolicyNetwork], TSHCALOCandidateArtifact]:
    artifact = inspect_tsh_calo_candidate(path, expected_sha256=expected_sha256)
    if artifact.artifact_kind != "ensemble_policy" or artifact.ensemble_size < 2:
        raise ValueError("TSH-CALO runtime requires an assembled epistemic ensemble")
    payload = load_checkpoint(path, expected_sha256=expected_sha256, map_location=device)
    architecture = dict(payload["architecture"])
    networks: list[TSHCALOPolicyNetwork] = []
    for state_dict in payload["ensemble_model_state_dicts"]:
        network = TSHCALOPolicyNetwork(
            hidden_dim=int(architecture["hidden_dim"]),
            graph_steps=int(architecture["graph_steps"]),
        ).to(device)
        network.load_state_dict(state_dict, strict=True)
        network.eval()
        networks.append(network)
    return networks, artifact
