from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from calo_rpd_studio.algorithms.calo.ai_controller import (
    AIController,
    PolicyInferenceTimeout,
    _POLICY_BROKER_CACHE,
    _POLICY_NETWORK_CACHE,
    _PolicyInferenceBroker,
)
from calo_rpd_studio.algorithms.calo.policy_network import CALOPolicyNetwork
from calo_rpd_studio.algorithms.calo.policy_schema import (
    CALO_RUNTIME_ARCHITECTURE,
    POLICY_ACTION_SCHEMA,
    POLICY_STATE_DIM,
    POLICY_STATE_SCHEMA,
    TRAINING_ENVIRONMENT_VERSION,
)


def _save_policy(path: Path, *, fill: float) -> None:
    network = CALOPolicyNetwork(input_dim=POLICY_STATE_DIM, hidden_dim=16)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.fill_(fill)
    torch.save(
        {
            "model_state_dict": network.state_dict(),
            "architecture": {"input_dim": POLICY_STATE_DIM, "hidden_dim": 16},
            "metadata": {
                "state_dimension": POLICY_STATE_DIM,
                "runtime_architecture_version": CALO_RUNTIME_ARCHITECTURE,
                "state_schema_version": POLICY_STATE_SCHEMA,
                "action_schema_version": POLICY_ACTION_SCHEMA,
                "training_environment_version": TRAINING_ENVIRONMENT_VERSION,
            },
        },
        path,
    )


def test_policy_cache_is_content_addressed_even_when_path_size_and_mtime_match(tmp_path):
    _POLICY_NETWORK_CACHE.clear()
    _POLICY_BROKER_CACHE.clear()
    checkpoint = tmp_path / "policy.pt"
    _save_policy(checkpoint, fill=0.125)
    fixed_ns = 1_700_000_000_000_000_000
    os.utime(checkpoint, ns=(fixed_ns, fixed_ns))
    original_size = checkpoint.stat().st_size

    first = AIController(checkpoint, seed=1, device="cpu")
    first_checksum = first.checksum
    first_network = first.network

    _save_policy(checkpoint, fill=0.875)
    assert checkpoint.stat().st_size == original_size, "fixture must preserve file size"
    os.utime(checkpoint, ns=(fixed_ns, fixed_ns))

    second = AIController(checkpoint, seed=2, device="cpu")
    assert second.checksum != first_checksum
    assert second.network is not first_network


def test_policy_broker_times_out_instead_of_waiting_forever():
    class StallingPolicy(torch.nn.Module):
        def forward(self, x):
            time.sleep(0.4)
            batch = x.shape[0]
            device = x.device
            return (
                torch.zeros((batch, 4), device=device),
                torch.zeros((batch, 6), device=device),
                torch.ones((batch, 6), device=device),
                torch.ones((batch, 6), device=device),
                torch.zeros((batch,), device=device),
            )

    broker = _PolicyInferenceBroker(
        StallingPolicy(), torch.device("cpu"), request_timeout_s=0.1
    )
    started = time.monotonic()
    try:
        with pytest.raises(PolicyInferenceTimeout, match="fail-closed"):
            broker.infer(np.zeros(POLICY_STATE_DIM, dtype=np.float32))
        assert time.monotonic() - started < 0.35
    finally:
        broker.close()
