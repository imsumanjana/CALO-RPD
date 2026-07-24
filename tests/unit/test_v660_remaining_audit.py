from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
import math

import numpy as np
import pytest

from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.orpd.constraint_violation import ConstraintViolation
from calo_rpd_studio.orpd.constraints import ConstraintToleranceConfig, branch_angle_limit_violation
from calo_rpd_studio.orpd.feasibility_rules import better, sort_key
from calo_rpd_studio.orpd.problem import Evaluation
from calo_rpd_studio.orpd.variable_decoder import ORPDVariableConfig, ORPDVariableDecoder
from calo_rpd_studio.power_system.voltage_stability import kessel_glavitsch_l_index
from calo_rpd_studio.power_system.newton_raphson import _dense_jacobian, MAX_DENSE_FALLBACK_BUSES
from calo_rpd_studio.accelerated.torch_power_flow import build_dense_admittance, MAX_DENSE_TORCH_BUSES
from calo_rpd_studio.statistics.friedman import friedman_test
from calo_rpd_studio.portfolio.exporter import PortfolioExporter


def _root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source(relative: str) -> str:
    return (_root() / relative).read_text(encoding="utf-8")


def test_c02_reusable_decoder_reuses_workspace_without_stale_controls(toy_case):
    config = ORPDVariableConfig(
        generator_voltages=True,
        transformer_taps=True,
        shunt_compensation=False,
    )
    decoder = ORPDVariableDecoder(toy_case, config)
    z0 = np.zeros(decoder.dimension)
    z1 = np.ones(decoder.dimension)
    first, _ = decoder.decode_reusable(z0)
    first_id = id(first)
    second, _ = decoder.decode_reusable(z1)
    assert id(second) == first_id
    # Compatibility decode remains independently owned for callers that retain a case.
    owned, _ = decoder.decode(z0)
    assert owned is not second


def test_c03_m04_m05_one_feasibility_tolerance_and_one_ordering():
    violation = ConstraintViolation(5e-7, {}, feasibility_tolerance=1e-6)
    assert violation.feasible
    a = Evaluation(10.0, True, 5e-7, feasibility_tolerance=1e-6)
    b = Evaluation(9.0, False, 2e-6, feasibility_tolerance=1e-6)
    assert better(a, b)
    assert sort_key(a) < sort_key(b)
    c = Evaluation(100.0, False, 3e-6, feasibility_tolerance=1e-6)
    d = Evaluation(1.0, False, 4e-6, feasibility_tolerance=1e-6)
    assert better(c, d) == (sort_key(c) < sort_key(d))


def test_c05_dense_torch_large_case_is_bounded_before_allocation():
    case = SimpleNamespace(n_bus=MAX_DENSE_TORCH_BUSES + 1)
    with pytest.raises(RuntimeError, match="sparse CPU-reference fallback"):
        build_dense_admittance(case, "cpu")


def test_c06_lindex_rejects_partition_dimension_mismatch(toy_case):
    partition = toy_case.clone()
    partition.bus = np.vstack([partition.bus, partition.bus[-1].copy()])
    partition.bus[-1, 0] = 999
    with pytest.raises(ValueError, match="same bus dimension"):
        kessel_glavitsch_l_index(
            toy_case,
            np.ones(toy_case.n_bus, dtype=complex),
            partition_case=partition,
        )


def test_c07_dense_newton_fallback_is_bounded_before_quadratic_allocation():
    voltage = np.ones(MAX_DENSE_FALLBACK_BUSES + 1, dtype=complex)
    with pytest.raises(RuntimeError, match="Dense Newton-Jacobian fallback is disabled"):
        _dense_jacobian(np.eye(1), voltage, np.array([], dtype=int), np.array([], dtype=int))


def test_h03_vectorized_branch_angle_constraint_is_numerically_correct(toy_case):
    case = toy_case.clone()
    case.branch[0, 11] = -10.0
    case.branch[0, 12] = 10.0
    va = np.array([15.0, 0.0, 0.0])
    value = branch_angle_limit_violation(case, va, ConstraintToleranceConfig(branch_angle_deg=0.0))
    assert value == pytest.approx(0.25)
    text = _source("calo_rpd_studio/orpd/constraints.py")
    assert "np.searchsorted" in text
    assert "for v in rows[:, F_BUS]" not in text


def test_h10_training_uses_separate_persistent_rng_streams():
    text = _source("calo_rpd_studio/algorithms/calo/training.py")
    assert "ppo_minibatch_shuffle" in text
    assert "historical_pretraining" in text
    assert "shuffle_rng.shuffle(indices)" in text
    assert "numpy_generator_stream_states" in text


def test_h14_friedman_degenerate_data_never_returns_nan():
    result = friedman_test([1, 1, 1], [1, 1, 1], [1, 1, 1])
    assert math.isfinite(result["statistic"])
    assert math.isfinite(result["p_value"])
    assert result["p_value"] == 1.0
    assert result["status"] == "degenerate_or_all_tied"


def test_h16_h17_h18_h20_silent_broad_handlers_are_hardened():
    scheduler = _source("calo_rpd_studio/compute/resource_scheduler.py")
    torch_orpd = _source("calo_rpd_studio/accelerated/torch_orpd.py")
    throughput = _source("calo_rpd_studio/accelerated/throughput_engine.py")
    manager = _source("calo_rpd_studio/app/experiment_manager.py")
    assert "except Exception" not in scheduler
    assert "except Exception" not in torch_orpd
    assert "CUDA runtime enumeration failed" in scheduler
    assert "XPU sidecar telemetry failed" in scheduler
    # Remaining broad throughput catches are fail-forward request/error boundaries, not suppression.
    assert "propagate the same scientific failure to every requester" in throughput
    assert "request.error = exc" in throughput
    assert "Unable to persist throughput profile" in manager
    assert "Accelerator pool close failed during shutdown" in manager


def test_h23_campaign_order_has_deterministic_secondary_key():
    text = _source("calo_rpd_studio/results/database.py")
    start = text.index("def list_campaigns")
    segment = text[start : start + 1200]
    assert "ORDER BY updated_at DESC, id DESC" in segment


def test_h25_h27_workspace_restore_uses_semantic_keys_and_structured_failures():
    text = _source("calo_rpd_studio/app/experiment_workspace_restorer.py")
    assert "pages_by_key" in text
    assert "__class__.__name__" not in text
    assert "WorkspaceRestoreError" in text
    assert '"case_load"' in text
    assert '"configuration"' in text


def test_h26_l20_validation_is_read_only_and_unknown_fields_fail_closed():
    config = ExperimentConfig()
    before = config.to_dict()
    config.validate()
    assert config.to_dict() == before
    payload = config.to_dict()
    payload["definitely_typod_field"] = 123
    with pytest.raises(ValueError, match="Unknown experiment configuration field"):
        ExperimentConfig.from_dict(payload)


def test_m16_policy_caches_are_bounded_lru():
    text = _source("calo_rpd_studio/algorithms/calo/ai_controller.py")
    assert "_POLICY_CACHE_MAX_ENTRIES" in text
    assert "OrderedDict" in text
    assert "_evict_policy_caches_locked" in text
    assert "popitem(last=False)" in text


def test_m34_sparse_jacobian_has_non_import_failure_fallback():
    text = _source("calo_rpd_studio/power_system/newton_raphson.py")
    assert "ImportError, RuntimeError, ValueError, TypeError, AttributeError" in text
    assert "MAX_DENSE_FALLBACK_BUSES" in text


def test_m36_inactive_candidates_are_removed_from_linear_solve():
    text = _source("calo_rpd_studio/accelerated/torch_power_flow.py")
    assert "active_rows = torch.where(active)[0]" in text
    assert "jacobian_active = jacobian.index_select(0, active_rows)" in text
    assert "rhs_active = f.index_select(0, active_rows)" in text


def test_m37_cross_scenario_batching_flattens_candidate_scenario_work():
    text = _source("calo_rpd_studio/accelerated/torch_orpd.py")
    assert "flat_records" in text
    assert "total_network_solves = count * len(self.scenarios)" in text
    assert "[record[2] for record in batch_records]" in text


def test_m48_m52_m54_gui_robustness_paths_are_explicit():
    intelligence = _source("calo_rpd_studio/gui/panels/calo_intelligence_panel.py")
    resume = _source("calo_rpd_studio/gui/panels/resume_center_panel.py")
    results = _source("calo_rpd_studio/gui/panels/results_explorer_panel.py")
    main = _source("calo_rpd_studio/app/main_window.py")
    assert "deployable_eligible: bool | None = None" in intelligence
    assert "'deployable_eligible' in locals()" not in intelligence
    assert "validation_resumed" in resume and "portfolio_export_resumed" in resume
    assert "validation_resumed.connect" in main and "portfolio_export_resumed.connect" in main
    assert "def select_run" in results and "return False" in results
    assert "raise KeyError" not in results[results.index("def select_run"):]


def test_m57_lazy_governor_uses_current_allocation_limit_profile():
    text = _source("calo_rpd_studio/app/state_manager.py")
    assert "self.compute_protection_profile.allocation_limit_fraction" in text
    assert "config=GovernorConfig(" in text


def test_l19_corrupt_portfolio_manifest_has_actionable_error(tmp_path):
    manifest = tmp_path / "portfolio_manifest.json"
    manifest.write_text("{ definitely not json", encoding="utf-8")
    with pytest.raises(ValueError, match="unreadable or corrupt"):
        PortfolioExporter._load_manifest(manifest)


def test_l23_stopping_experiment_does_not_zero_verified_results():
    text = _source("calo_rpd_studio/app/workflow_manager.py")
    segment = text[text.index("def mark_experiment_stopped"): text.index("def ", text.index("def mark_experiment_stopped") + 4)]
    assert "verified_results=0" not in segment
    assert "max(0, int(self.verified_results))" in segment


def test_v64_n04_real_development_reuses_validated_config_and_case_templates():
    text = _source("calo_rpd_studio/algorithms/calo/training.py")
    assert "development_case_cache" in text
    assert "ExperimentConfig.load(config_path)" in text
    # Load is outside the per-episode loop in _collect_rollout_chunk.
    segment = text[text.index("def _collect_rollout_chunk"): text.index("def ", text.index("def _collect_rollout_chunk") + 4)]
    assert segment.count("ExperimentConfig.load(config_path)") == 1


def test_v64_n05_n06_synthetic_broker_splits_oversized_and_caches_static_tensors():
    text = _source("calo_rpd_studio/algorithms/calo/device_resident_synthetic.py")
    assert "if len(population) > self.max_candidates" in text
    assert "oversized_request_count" in text
    assert "_STATIC_STACK_CACHE_MAX_ENTRIES" in text
    assert "static_fingerprint" in text
    assert "_STATIC_STACK_CACHE.popitem(last=False)" in text
