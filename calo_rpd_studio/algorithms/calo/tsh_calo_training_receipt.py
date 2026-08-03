"""Immutable receipts for completed counted TSH-CALO development-training episodes."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json

from calo_rpd_studio.power_system.case_identity import protected_holdout_matches


TSH_CALO_TRAINING_EPISODE_RECEIPT_SCHEMA = "tsh-calo-training-episode-receipt-v1"


def _valid_sha256(value: str) -> bool:
    text = str(value).strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def canonical_reward_sequence_sha256(rewards) -> str:
    """Hash exact finite IEEE-754 reward values without decimal rendering ambiguity."""

    encoded: list[str] = []
    for reward in rewards:
        value = float(reward)
        if not (-float("inf") < value < float("inf")):
            raise ValueError("TSH-CALO episode rewards must be finite")
        encoded.append(value.hex())
    payload = json.dumps(encoded, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingEpisodeReceipt:
    schema_version: str
    session_id: str
    training_run_id: str
    training_design_sha256: str
    session_design_sha256: str
    environment_design_sha256: str
    case_identity: str
    case_checksum: str
    problem_fingerprint: str
    seed: int
    deterministic_policy: bool
    candidate_evaluations: int
    scenario_power_flow_calls: int
    canonical_transition_count: int
    ppo_update_count: int
    canonical_reward_sha256: str
    accounting_complete: bool
    terminal: bool
    receipt_sha256: str = ""

    def unsigned_payload(self) -> dict:
        payload = asdict(self)
        payload.pop("receipt_sha256", None)
        return payload

    def calculated_sha256(self) -> str:
        encoded = json.dumps(
            self.unsigned_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def validate(self) -> None:
        if self.schema_version != TSH_CALO_TRAINING_EPISODE_RECEIPT_SCHEMA:
            raise ValueError("TSH-CALO training episode receipt schema is incompatible")
        for label, value in (
            ("session ID", self.session_id),
            ("training run ID", self.training_run_id),
            ("case identity", self.case_identity),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"TSH-CALO training episode requires a valid {label}")
        for label, value in (
            ("training design", self.training_design_sha256),
            ("session design", self.session_design_sha256),
            ("environment design", self.environment_design_sha256),
            ("case checksum", self.case_checksum),
            ("problem fingerprint", self.problem_fingerprint),
            ("canonical rewards", self.canonical_reward_sha256),
            ("receipt", self.receipt_sha256),
        ):
            if not isinstance(value, str) or not _valid_sha256(value):
                raise ValueError(f"TSH-CALO training episode {label} SHA-256 is invalid")
        leaked = protected_holdout_matches((self.case_identity,))
        if leaked:
            raise ValueError(
                "Protected holdout cases cannot enter TSH-CALO training: " + ", ".join(leaked)
            )
        for label, value in (
            ("seed", self.seed),
            ("candidate evaluations", self.candidate_evaluations),
            ("scenario power-flow calls", self.scenario_power_flow_calls),
            ("canonical transition count", self.canonical_transition_count),
            ("PPO update count", self.ppo_update_count),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"TSH-CALO training episode {label} must be an integer")
        if not isinstance(self.deterministic_policy, bool):
            raise ValueError("TSH-CALO training episode deterministic-policy flag must be Boolean")
        if not isinstance(self.accounting_complete, bool) or not isinstance(self.terminal, bool):
            raise ValueError("TSH-CALO training episode completion flags must be Boolean")
        if self.seed < 0:
            raise ValueError("TSH-CALO training episode seed cannot be negative")
        if self.candidate_evaluations < 1 or self.scenario_power_flow_calls < 1:
            raise ValueError("TSH-CALO training episode accounting must be positive")
        if self.scenario_power_flow_calls < self.candidate_evaluations:
            raise ValueError("TSH-CALO training episode scenario accounting is impossible")
        if self.canonical_transition_count < 1 or self.ppo_update_count < 1:
            raise ValueError("TSH-CALO training episode must contain transitions and PPO updates")
        if not self.accounting_complete or not self.terminal:
            raise ValueError("Only complete terminal TSH-CALO training episodes may issue receipts")
        if self.calculated_sha256() != self.receipt_sha256:
            raise ValueError("TSH-CALO training episode receipt integrity check failed")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)


def build_tsh_calo_training_episode_receipt(**values) -> TSHCALOTrainingEpisodeReceipt:
    receipt = TSHCALOTrainingEpisodeReceipt(
        schema_version=TSH_CALO_TRAINING_EPISODE_RECEIPT_SCHEMA,
        receipt_sha256="",
        **values,
    )
    signed = replace(receipt, receipt_sha256=receipt.calculated_sha256())
    signed.validate()
    return signed


def load_tsh_calo_training_episode_receipt(payload: dict) -> TSHCALOTrainingEpisodeReceipt:
    try:
        receipt = TSHCALOTrainingEpisodeReceipt(**dict(payload or {}))
    except TypeError as exc:
        raise ValueError("TSH-CALO training episode receipt fields are incomplete") from exc
    receipt.validate()
    return receipt
