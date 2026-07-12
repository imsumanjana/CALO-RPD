"""Reproducible PPO-style training for the compact CALO policy controller."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn

from .policy_network import CALOPolicyNetwork


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 20
    episodes_per_epoch: int = 16
    horizon: int = 24
    seed: int = 2026
    learning_rate: float = 3e-4
    gamma: float = 0.97
    entropy_weight: float = 0.01
    hidden_dim: int = 64


class TrainingCancelled(RuntimeError):
    """Raised when the user requests a safe stop between training units."""


def _task(rng, dim):
    choice = int(rng.integers(5))
    shift = rng.uniform(-0.4, 0.4, dim)
    if choice == 0:
        return lambda x: float(np.sum((x - 0.5 - shift * 0.2) ** 2))
    if choice == 1:
        return lambda x: float(
            10 * dim
            + np.sum((x - 0.5) ** 2 * 100 - 10 * np.cos(2 * np.pi * (x - 0.5) * 10))
        )
    if choice == 2:
        return lambda x: float(
            -20 * np.exp(-0.2 * np.sqrt(np.mean(((x - 0.5) * 8) ** 2)))
            - np.exp(np.mean(np.cos(2 * np.pi * (x - 0.5) * 8)))
            + 20
            + np.e
        )
    if choice == 3:
        return lambda x: float(
            np.sum(((x - 0.5) * 6) ** 2) / 4000
            - np.prod(np.cos(((x - 0.5) * 6) / np.sqrt(np.arange(1, dim + 1))))
            + 1
        )
    return lambda x: float(
        np.sum(
            100 * (((x[1:] - 0.5) * 4) - ((x[:-1] - 0.5) * 4) ** 2) ** 2
            + (((x[:-1] - 0.5) * 4) - 1) ** 2
        )
    )


def train_policy(config: TrainingConfig, output_path, progress_callback=None, cancel_callback=None):
    """Train and save a CALO policy with optional progress and cancellation hooks."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    net = CALOPolicyNetwork(14, config.hidden_dim)
    opt = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    history = []

    for epoch in range(config.epochs):
        if cancel_callback and cancel_callback():
            raise TrainingCancelled("CALO policy training was cancelled safely.")
        losses = []
        returns = []
        for episode in range(config.episodes_per_epoch):
            if cancel_callback and cancel_callback():
                raise TrainingCancelled("CALO policy training was cancelled safely.")
            dim = int(rng.integers(5, 21))
            fn = _task(rng, dim)
            pop = rng.random((16, dim))
            values = np.asarray([fn(x) for x in pop])
            prev = float(values.min())
            logps = []
            vals = []
            ent = []
            rewards = []
            success = np.zeros(6)

            for t in range(config.horizon):
                if cancel_callback and cancel_callback():
                    raise TrainingCancelled("CALO policy training was cancelled safely.")
                div = float(np.mean(np.linalg.norm(pop - pop.mean(0), axis=1)) / np.sqrt(dim))
                state = np.r_[
                    div,
                    0,
                    0,
                    t / config.horizon,
                    1.0,
                    0.0,
                    div,
                    max(0, 1 - t / config.horizon),
                    success,
                ]
                st = torch.tensor(state, dtype=torch.float32)
                logits, pars, value = net(st)
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                raw = pars.detach().numpy()
                sigma = 0.005 + raw[2] * 0.295
                best = pop[np.argmin(values)]
                operator = int(action)
                if operator == 0:
                    candidate = np.clip(
                        pop
                        + (0.15 + raw[0] * 1.35)
                        * np.abs(rng.normal(size=pop.shape))
                        * (best - pop),
                        0,
                        1,
                    )
                elif operator == 1:
                    candidate = np.clip(
                        pop + rng.random(pop.shape) * (pop[rng.permutation(len(pop))] - pop),
                        0,
                        1,
                    )
                elif operator == 2:
                    candidate = np.clip(0.7 * pop + 0.3 * best, 0, 1)
                elif operator == 3:
                    candidate = np.clip(best + sigma * rng.normal(size=pop.shape), 0, 1)
                elif operator == 4:
                    candidate = np.clip(pop + 0.5 * rng.random(pop.shape) * (best - pop), 0, 1)
                else:
                    candidate = np.clip(
                        pop + rng.normal(0, max(sigma, 0.08), pop.shape),
                        0,
                        1,
                    )
                candidate_values = np.asarray([fn(x) for x in candidate])
                mask = candidate_values < values
                pop[mask] = candidate[mask]
                values[mask] = candidate_values[mask]
                new = float(values.min())
                reward = float(np.clip((prev - new) / max(abs(prev), 1), -1, 1))
                success[operator] = 0.9 * success[operator] + 0.1 * (reward > 0)
                rewards.append(reward)
                logps.append(dist.log_prob(action))
                vals.append(value)
                ent.append(dist.entropy())
                prev = new

            discounted = []
            g = 0.0
            for reward in rewards[::-1]:
                g = reward + config.gamma * g
                discounted.append(g)
            discounted_tensor = torch.tensor(discounted[::-1], dtype=torch.float32)
            value_tensor = torch.stack(vals)
            advantage = discounted_tensor - value_tensor.detach()
            policy_loss = -(torch.stack(logps) * advantage).mean()
            value_loss = 0.5 * ((value_tensor - discounted_tensor) ** 2).mean()
            entropy = torch.stack(ent).mean()
            loss = policy_loss + value_loss - config.entropy_weight * entropy
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            returns.append(float(sum(rewards)))

            if progress_callback:
                total_units = config.epochs * config.episodes_per_epoch
                done_units = epoch * config.episodes_per_epoch + episode + 1
                progress_callback(
                    int(100 * done_units / max(total_units, 1)),
                    f"Epoch {epoch + 1}/{config.epochs} · episode {episode + 1}/{config.episodes_per_epoch}",
                )

        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(np.mean(losses)),
                "mean_return": float(np.mean(returns)),
            }
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "software_version": "1.0.6",
        "training_seed": config.seed,
        "training_configuration": asdict(config),
        "training_problem_identifiers": [
            "synthetic_sphere_family",
            "synthetic_rastrigin_family",
            "synthetic_ackley_family",
            "synthetic_griewank_family",
            "synthetic_rosenbrock_family",
        ],
        "final_test_systems_used_for_training": False,
        "history": history,
    }
    torch.save({"model_state_dict": net.state_dict(), "metadata": metadata}, output)
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return history
