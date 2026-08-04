"""Independent counted rollout/PPO session for fresh TSH-CALO member training.

This module owns no experiment, registry, qualification, activation, GUI, or production-inference
capability.  A completed session records a development episode receipt in the trainer; candidate
export remains a separate explicit caller action and always remains unqualified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from calo_rpd_studio.ai.model_io import durable_trusted_torch_save, load_trusted_resume

from .tsh_calo_training import (
    IndependentTSHCALORolloutCollector,
    IndependentTSHCALOTrainer,
    TSHCALOTrainingConfig,
)
from .tsh_calo_training_environment import (
    IndependentTSHCALOTrainingEnvironment,
    TSHCALOTrainingEnvironmentConfig,
)
from .tsh_calo_training_receipt import (
    TSHCALOTrainingEpisodeReceipt,
    build_tsh_calo_training_episode_receipt,
    canonical_reward_sequence_sha256,
    load_tsh_calo_training_episode_receipt,
)


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingSessionConfig:
    session_id: str
    deterministic_policy: bool = False

    def validate(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("TSH-CALO training session requires a session ID")
        if not isinstance(self.deterministic_policy, bool):
            raise ValueError("TSH-CALO deterministic-policy flag must be Boolean")

    def scientific_design_hash(
        self,
        training_config: TSHCALOTrainingConfig,
        environment_config: TSHCALOTrainingEnvironmentConfig,
    ) -> str:
        self.validate()
        payload = {
            "schema_version": IndependentTSHCALOTrainingSession.SCHEMA_VERSION,
            "session": asdict(self),
            "training_design_sha256": training_config.scientific_design_hash(),
            "environment_design_sha256": environment_config.scientific_design_hash(training_config),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TSHCALOTrainingSessionResult:
    receipt: TSHCALOTrainingEpisodeReceipt
    update_metrics: tuple[dict[str, float], ...]
    canonical_rewards: tuple[float, ...]


class IndependentTSHCALOTrainingSession:
    """Advance one immutable development-case episode and issue one counted receipt."""

    SCHEMA_VERSION = "tsh-calo-independent-training-session-v2-batched-device-context"

    def __init__(
        self,
        trainer: IndependentTSHCALOTrainer,
        environment: IndependentTSHCALOTrainingEnvironment,
        config: TSHCALOTrainingSessionConfig,
    ) -> None:
        config.validate()
        if (
            trainer.config.scientific_design_hash()
            != environment.training_config.scientific_design_hash()
        ):
            raise ValueError("TSH-CALO trainer and environment scientific designs differ")
        if environment.initialized:
            raise ValueError("A new TSH-CALO session requires an unstarted training environment")
        self.trainer = trainer
        self.environment = environment
        self.config = config
        self.collector = IndependentTSHCALORolloutCollector(trainer)
        self.starting_update_steps = trainer.update_steps
        self.canonical_rewards: list[float] = []
        self.update_metrics: list[dict[str, float]] = []
        self.transition_count = 0
        self.started = False
        self.completed = False
        self.failed = False
        self.receipt: TSHCALOTrainingEpisodeReceipt | None = None

    @property
    def scientific_design_hash(self) -> str:
        return self.config.scientific_design_hash(self.trainer.config, self.environment.config)

    def _assert_advanceable(self) -> None:
        if self.failed:
            raise RuntimeError("TSH-CALO training session failed and cannot continue")
        if self.completed:
            raise RuntimeError("TSH-CALO training session is already complete")

    def _update_policy(self, *, terminal: bool, next_observation) -> None:
        if not self.collector.states:
            raise RuntimeError("TSH-CALO session cannot update without counted transitions")
        bootstrap = (
            0.0
            if terminal
            else self.trainer.estimate_value(
                next_observation.policy_state,
                population_size=self.environment.config.population_size,
            )
        )
        batch = self.collector.build_batch(bootstrap_value=bootstrap)
        metrics = self.trainer.update(batch)
        if not all(np.isfinite(float(value)) for value in metrics.values()):
            raise RuntimeError("TSH-CALO training session produced non-finite update metrics")
        self.update_metrics.append({str(key): float(value) for key, value in metrics.items()})
        self.collector = IndependentTSHCALORolloutCollector(self.trainer)

    def _complete(self) -> TSHCALOTrainingEpisodeReceipt:
        provenance = self.environment.scientific_provenance()
        counted_orpd = dict(provenance["counted_orpd_execution"])
        updates = self.trainer.update_steps - self.starting_update_steps
        receipt = build_tsh_calo_training_episode_receipt(
            session_id=self.config.session_id,
            training_run_id=self.trainer.config.training_run_id,
            training_design_sha256=self.trainer.config.scientific_design_hash(),
            session_design_sha256=self.scientific_design_hash,
            environment_design_sha256=self.environment.config.scientific_design_hash(
                self.trainer.config
            ),
            case_identity=self.environment.config.case_identity,
            case_checksum=self.environment.case_checksum,
            problem_fingerprint=self.environment.problem_fingerprint,
            seed=self.environment.config.seed,
            deterministic_policy=self.config.deterministic_policy,
            candidate_evaluations=int(provenance["candidate_evaluations"]),
            scenario_power_flow_calls=int(provenance["scenario_power_flow_calls"]),
            canonical_transition_count=self.transition_count,
            ppo_update_count=updates,
            canonical_reward_sha256=canonical_reward_sequence_sha256(self.canonical_rewards),
            accounting_complete=bool(provenance["accounting_complete"]),
            terminal=self.environment.terminal,
            counted_orpd_evaluator_computation=str(
                provenance["trusted_orpd_evaluator_computation"]
            ),
            counted_orpd_selected_device=str(counted_orpd["selected_device"]),
            counted_orpd_batch_context_api=bool(counted_orpd["batch_context_api"]),
            counted_orpd_target_evaluations_per_host_boundary=int(
                counted_orpd["target_evaluations_per_host_boundary"]
            ),
            counted_orpd_cpu_cuda_inner_loop_transfers=int(
                counted_orpd["cpu_cuda_inner_loop_transfers"] or 0
            ),
            counted_orpd_context_power_flow_reruns=int(
                counted_orpd["context_power_flow_reruns"] or 0
            ),
        )
        self.trainer.record_training_episode_receipt(receipt.to_dict())
        self.receipt = receipt
        self.completed = True
        return receipt

    def advance(self, *, max_transitions: int | None = None) -> TSHCALOTrainingSessionResult | None:
        """Advance up to a boundary or complete the episode; never export or activate a policy."""

        self._assert_advanceable()
        if max_transitions is not None and int(max_transitions) < 1:
            raise ValueError("TSH-CALO session max_transitions must be positive when supplied")
        limit = None if max_transitions is None else int(max_transitions)
        advanced = 0
        try:
            if not self.started:
                observation = self.environment.reset()
                self.started = True
            else:
                observation = self.environment.observe()
            while limit is None or advanced < limit:
                pending = self.collector.sample(
                    observation.policy_state,
                    observation.action_mask,
                    observation.learner_groups,
                    observation.learner_contexts,
                    deterministic=self.config.deterministic_policy,
                )
                try:
                    step = self.environment.step(pending.action)
                except Exception:
                    self.collector.discard_pending()
                    raise
                self.collector.commit(step.transition, terminal=step.terminal)
                reward = float(step.transition.reward.total)
                self.canonical_rewards.append(reward)
                self.transition_count += 1
                advanced += 1
                capacity = self.trainer.config.resource_envelope.rollout_capacity
                if len(self.collector.states) >= capacity or step.terminal:
                    self._update_policy(
                        terminal=step.terminal,
                        next_observation=step.next_observation,
                    )
                if step.terminal:
                    receipt = self._complete()
                    return TSHCALOTrainingSessionResult(
                        receipt,
                        tuple(self.update_metrics),
                        tuple(self.canonical_rewards),
                    )
                assert step.next_observation is not None
                observation = step.next_observation
            return None
        except Exception:
            self.failed = True
            raise

    def state_dict(self) -> dict:
        if self.failed:
            raise RuntimeError("Failed TSH-CALO training sessions cannot be checkpointed")
        if not self.started:
            raise RuntimeError("TSH-CALO training session must start before checkpointing")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "session_design_sha256": self.scientific_design_hash,
            "session_config": asdict(self.config),
            "trainer": self.trainer.resume_state_dict(),
            "environment": self.environment.state_dict(),
            "collector": self.collector.state_dict(),
            "starting_update_steps": self.starting_update_steps,
            "canonical_rewards": np.asarray(self.canonical_rewards, dtype=np.float64),
            "update_metrics": tuple(self.update_metrics),
            "transition_count": self.transition_count,
            "completed": self.completed,
            "receipt": None if self.receipt is None else self.receipt.to_dict(),
        }

    def save_resume(self, path: str | Path) -> str:
        target = Path(path).expanduser().resolve()
        return durable_trusted_torch_save(self.state_dict(), target)

    @classmethod
    def from_state_dict(
        cls,
        problem,
        training_config: TSHCALOTrainingConfig,
        environment_config: TSHCALOTrainingEnvironmentConfig,
        session_config: TSHCALOTrainingSessionConfig,
        payload: dict,
    ) -> "IndependentTSHCALOTrainingSession":
        if str(payload.get("schema_version", "")) != cls.SCHEMA_VERSION:
            raise ValueError("TSH-CALO training session checkpoint schema is incompatible")
        expected_design = session_config.scientific_design_hash(training_config, environment_config)
        if str(payload.get("session_design_sha256", "")) != expected_design:
            raise ValueError("TSH-CALO training session scientific design changed")
        trainer = IndependentTSHCALOTrainer.from_resume_state_dict(
            dict(payload.get("trainer", {})), expected_config=training_config
        )
        try:
            environment = IndependentTSHCALOTrainingEnvironment.from_state_dict(
                problem,
                training_config,
                environment_config,
                dict(payload.get("environment", {})),
            )
            session = cls.__new__(cls)
            session.trainer = trainer
            session.environment = environment
            session.config = session_config
            session.collector = IndependentTSHCALORolloutCollector.from_state_dict(
                trainer, dict(payload.get("collector", {}))
            )
            session.starting_update_steps = int(payload.get("starting_update_steps", 0))
            session.canonical_rewards = np.asarray(
                payload.get("canonical_rewards", []), dtype=float
            ).tolist()
            session.update_metrics = [
                {str(key): float(value) for key, value in dict(item).items()}
                for item in payload.get("update_metrics", ())
            ]
            session.transition_count = int(payload.get("transition_count", 0))
            session.started = True
            session.completed = bool(payload.get("completed", False))
            session.failed = False
            raw_receipt = payload.get("receipt")
            session.receipt = (
                None
                if raw_receipt is None
                else load_tsh_calo_training_episode_receipt(dict(raw_receipt))
            )
            session._validate_restored_state()
            return session
        except Exception:
            trainer.close()
            raise

    @classmethod
    def load_resume(
        cls,
        path: str | Path,
        *,
        problem,
        training_config: TSHCALOTrainingConfig,
        environment_config: TSHCALOTrainingEnvironmentConfig,
        session_config: TSHCALOTrainingSessionConfig,
    ) -> "IndependentTSHCALOTrainingSession":
        payload = load_trusted_resume(path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError("TSH-CALO training session checkpoint payload is invalid")
        return cls.from_state_dict(
            problem,
            training_config,
            environment_config,
            session_config,
            payload,
        )

    def _validate_restored_state(self) -> None:
        if self.transition_count != len(self.canonical_rewards):
            raise ValueError("TSH-CALO session transition and reward counts disagree")
        if self.starting_update_steps < 0 or self.starting_update_steps > self.trainer.update_steps:
            raise ValueError("TSH-CALO session starting update count is invalid")
        if len(self.update_metrics) != self.trainer.update_steps - self.starting_update_steps:
            raise ValueError("TSH-CALO session metric and PPO update counts disagree")
        if self.completed:
            if self.receipt is None or not self.environment.terminal:
                raise ValueError("Completed TSH-CALO session checkpoint lacks terminal receipt")
            if not any(
                item.get("receipt_sha256") == self.receipt.receipt_sha256
                for item in self.trainer.training_episode_receipts
            ):
                raise ValueError("Completed TSH-CALO session receipt is absent from trainer")
        elif self.receipt is not None or self.environment.terminal:
            raise ValueError("Incomplete TSH-CALO session checkpoint has terminal state")

    def close(self) -> None:
        self.trainer.close()

    def __enter__(self) -> "IndependentTSHCALOTrainingSession":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()
