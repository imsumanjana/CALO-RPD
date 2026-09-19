"""Required physical-CUDA parity in the explicit acceptance lane, never an emulated pass."""

from __future__ import annotations
import json
import os

import pytest
import torch

from calo_rpd_studio.accelerated.parity_audit import run_configuration_parity_audit
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig


@pytest.mark.parametrize("case", ["case30", "case57"])
def test_physical_cuda_fp64_scientific_parity(case, tmp_path):
    required = os.environ.get("CALO_REQUIRE_PHYSICAL_CUDA") == "1"
    if not required:
        pytest.skip("physical CUDA acceptance requires the separately recorded CUDA lane")
    assert torch.cuda.is_available(), "Required physical CUDA is not visible in this process"
    config = ExperimentConfig()
    config.case_name = case
    config.study_case_plan = [case]
    config.algorithms = ["TLBO"]
    config.algorithm_parameters = {"TLBO": {}}
    config.execution_backend = "cuda_preferred"
    config.requested_compute_device = "cuda:0"
    config.tensor_batch_size = 8
    free_before, total_memory = torch.cuda.mem_get_info(0)
    allocated_before = torch.cuda.memory_allocated(0)
    torch.cuda.reset_peak_memory_stats(0)
    report = run_configuration_parity_audit(config, device="cuda:0", candidates=3)
    incremental_peak = max(0, torch.cuda.max_memory_allocated(0) - allocated_before)
    report.update(
        free_memory_at_admission=free_before,
        total_memory=total_memory,
        incremental_peak_allocated=incremental_peak,
    )
    assert incremental_peak <= 0.8 * free_before, report
    report.update(
        torch_version=torch.__version__, visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")
    )
    (tmp_path / "physical-parity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert report["passed"], report
    assert str(report["device"]).startswith("cuda"), report
    assert report["dtype"] == "float64"
    assert report["candidate_battery_count"] >= 3
