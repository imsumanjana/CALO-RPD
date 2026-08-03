"""Change-B graph-state, versioning, and topology-encoder invariants."""

from __future__ import annotations

from dataclasses import replace
import inspect

import numpy as np
import pytest
import torch

from calo_rpd_studio.algorithms.calo.policy_schema import (
    CALO_RUNTIME_ARCHITECTURE,
    POLICY_ACTION_SCHEMA,
    POLICY_STATE_SCHEMA,
)
from calo_rpd_studio.algorithms.calo.topology_context import (
    BRANCH_FEATURE_DIM,
    BUS_FEATURE_DIM,
    CONTROL_FEATURE_DIM,
    SCENARIO_FEATURE_DIM,
    ScenarioDescriptor,
    TopologyAwarePolicyState,
    TopologyContextEncoder,
    TopologyGraphState,
    build_topology_graph_state,
)
from calo_rpd_studio.algorithms.calo.tsh_calo_schema import (
    DEFAULT_TSH_CALO_FEATURES,
    TSH_CALO_ACTION_SCHEMA,
    TSH_CALO_ALGORITHM_VERSION,
    TSH_CALO_STATE_SCHEMA,
    TSHCALOFeatureFlags,
)
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ORPDVariableDecoder
from calo_rpd_studio.power_system.ac_power_flow import run_ac_power_flow


def _permuted(state: TopologyGraphState) -> TopologyGraphState:
    node_order = np.arange(len(state.node_features))[::-1]
    inverse = np.empty_like(node_order)
    inverse[node_order] = np.arange(len(node_order))
    edge_order = np.arange(len(state.edge_features))[::-1]
    control_order = np.arange(len(state.control_features))[::-1]
    scenario_order = np.arange(len(state.scenario_features))[::-1]
    return TopologyGraphState(
        node_features=state.node_features[node_order],
        edge_index=inverse[state.edge_index[:, edge_order]],
        edge_features=state.edge_features[edge_order],
        control_features=state.control_features[control_order],
        control_bus_index=inverse[state.control_bus_index[:, control_order]],
        control_groups=state.control_groups[control_order],
        scenario_features=state.scenario_features[scenario_order],
        bus_numbers=tuple(state.bus_numbers[int(i)] for i in node_order),
        branch_indices=tuple(state.branch_indices[int(i)] for i in edge_order),
        control_labels=tuple(state.control_labels[int(i)] for i in control_order),
    )


def test_tsh_calo_versions_are_new_and_experimental_schedule_defaults_off():
    assert TSH_CALO_ALGORITHM_VERSION != CALO_RUNTIME_ARCHITECTURE
    assert TSH_CALO_STATE_SCHEMA != POLICY_STATE_SCHEMA
    assert TSH_CALO_ACTION_SCHEMA != POLICY_ACTION_SCHEMA
    assert DEFAULT_TSH_CALO_FEATURES.population_schedule is False
    assert DEFAULT_TSH_CALO_FEATURES.physics_repair is False
    with pytest.raises(ValueError, match="experimental"):
        TSHCALOFeatureFlags(population_schedule=True).validate()
    TSHCALOFeatureFlags(population_schedule=True, allow_experimental_components=True).validate()


def test_topology_builder_uses_only_an_already_computed_power_flow(toy_case):
    decoder = ORPDVariableDecoder(toy_case, ORPDVariableConfig())
    result = run_ac_power_flow(toy_case)
    state = build_topology_graph_state(
        toy_case,
        decoder,
        np.full(decoder.dimension, 0.5),
        result,
        [ScenarioDescriptor(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)],
    )

    state.validate()
    TopologyAwarePolicyState(np.zeros(32), state).validate()
    assert state.node_features.shape == (toy_case.n_bus, BUS_FEATURE_DIM)
    assert state.edge_features.shape[1] == BRANCH_FEATURE_DIM
    assert state.control_features.shape == (decoder.dimension, CONTROL_FEATURE_DIM)
    assert state.scenario_features.shape == (1, SCENARIO_FEATURE_DIM)
    assert "run_ac_power_flow" not in inspect.getsource(build_topology_graph_state)
    with pytest.raises(ValueError, match="aggregate_features"):
        TopologyAwarePolicyState(np.zeros(31), state).validate()


def test_topology_encoder_is_invariant_to_bus_edge_control_and_scenario_order(toy_case):
    decoder = ORPDVariableDecoder(toy_case, ORPDVariableConfig())
    result = run_ac_power_flow(toy_case)
    scenarios = [
        ScenarioDescriptor(0.8, 0.1, 0.0, 0.0, 0.4, 0.0),
        ScenarioDescriptor(1.2, 0.3, 0.5, 1.0, 0.6, 1.0),
    ]
    state = build_topology_graph_state(
        toy_case, decoder, np.full(decoder.dimension, 0.5), result, scenarios
    )
    permuted = _permuted(state)
    torch.manual_seed(814)
    encoder = TopologyContextEncoder(hidden_dim=16).double().eval()

    original_encoding = encoder(state)
    permuted_encoding = encoder(permuted)

    torch.testing.assert_close(
        original_encoding.graph_embedding,
        permuted_encoding.graph_embedding,
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        original_encoding.group_embeddings,
        permuted_encoding.group_embeddings,
        rtol=1e-12,
        atol=1e-12,
    )


def test_topology_encoder_is_deterministic_and_sensitive_to_declared_topology(toy_case):
    decoder = ORPDVariableDecoder(toy_case, ORPDVariableConfig())
    result = run_ac_power_flow(toy_case)
    state = build_topology_graph_state(
        toy_case,
        decoder,
        np.full(decoder.dimension, 0.5),
        result,
        [ScenarioDescriptor(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)],
    )
    changed_edges = state.edge_features.copy()
    changed_edges[0, 1] += 0.25
    changed = replace(state, edge_features=changed_edges)
    torch.manual_seed(23)
    encoder = TopologyContextEncoder(hidden_dim=12).double().eval()

    first = encoder(state).graph_embedding
    second = encoder(state).graph_embedding
    altered = encoder(changed).graph_embedding

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert not torch.allclose(first, altered, rtol=1e-10, atol=1e-10)


def test_topology_state_rejects_nonfinite_features_and_unknown_bus_indices():
    state = TopologyGraphState(
        node_features=np.zeros((2, BUS_FEATURE_DIM)),
        edge_index=np.asarray([[0], [1]], dtype=np.int64),
        edge_features=np.zeros((1, BRANCH_FEATURE_DIM)),
        control_features=np.zeros((1, CONTROL_FEATURE_DIM)),
        control_bus_index=np.asarray([[0], [1]], dtype=np.int64),
        control_groups=np.asarray([0], dtype=np.int64),
        scenario_features=np.zeros((1, SCENARIO_FEATURE_DIM)),
        bus_numbers=(1, 2),
        branch_indices=(0,),
        control_labels=("Vg@1",),
    )
    state.validate()
    bad_features = state.node_features.copy()
    bad_features[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        replace(state, node_features=bad_features).validate()
    with pytest.raises(ValueError, match="outside"):
        replace(state, edge_index=np.asarray([[0], [2]], dtype=np.int64)).validate()
