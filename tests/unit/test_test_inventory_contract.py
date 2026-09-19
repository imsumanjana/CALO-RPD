from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from calo_rpd_studio.validation.test_inventory import review_inventory, review_required_lanes


def _inventory():
    return {
        "selected": ["tests/check.py::test_current"],
        "exit_code": 0,
        "outcomes": {"tests/check.py::test_current": {"outcome": "passed", "phases": {}}},
        "deselected": [],
        "collection": [],
    }


def test_missing_execution_and_failure_cannot_be_accepted(tmp_path):
    for outcome in ("failed", "incomplete", "xpassed", "unknown"):
        data = _inventory()
        data["outcomes"]["tests/check.py::test_current"]["outcome"] = outcome
        report = review_inventory(data, {}, source_root=tmp_path)
        assert not report["complete_for_selected_scope"] and report["errors"]
    data = _inventory()
    data["outcomes"] = {}
    assert review_inventory(data, {}, source_root=tmp_path)["errors"]


def test_skips_deselections_and_collection_omissions_are_not_invisible(tmp_path):
    data = _inventory()
    data["outcomes"]["tests/check.py::test_current"] = {"outcome": "skipped", "phases": {}}
    data["deselected"] = [{"nodeid": "tests/check.py::test_other", "reason": "selection"}]
    data["collection"] = [{"nodeid": "tests/unreadable.py", "outcome": "failed"}]
    report = review_inventory(data, {}, source_root=tmp_path)
    assert len(report["unclosed_omissions"]) == 3
    assert not report["complete_for_selected_scope"]


def test_historical_disposition_requires_exact_source_and_executed_successor(tmp_path):
    path = tmp_path / "tests/check.py"
    path.parent.mkdir()
    path.write_text("# reviewed historical contract\n", encoding="utf-8")
    data = _inventory()
    data["selected"].append("tests/check.py::test_old")
    data["outcomes"]["tests/check.py::test_old"] = {
        "outcome": "skipped",
        "phases": {"setup": {"outcome": "skipped", "reason": "historical v1 release gate"}},
    }
    policy = {
        "historical_cases": [
            {
                "nodeid": "tests/check.py::test_old",
                "reason": "historical v1 release gate",
                "source_sha256": hashlib.sha256(
                    path.read_text(encoding="utf-8").encode("utf-8")
                ).hexdigest(),
                "disposition": "historical_release_only",
                "current_successors": ["tests/check.py::test_current"],
            }
        ]
    }
    assert review_inventory(data, policy, source_root=tmp_path)["complete_for_selected_scope"]
    incomplete = copy.deepcopy(data)
    del incomplete["outcomes"]["tests/check.py::test_current"]
    assert not review_inventory(incomplete, policy, source_root=tmp_path)[
        "complete_for_selected_scope"
    ]
    path.write_text("# changed unreviewed historical contract\n", encoding="utf-8")
    assert not review_inventory(data, policy, source_root=tmp_path)["complete_for_selected_scope"]


def test_missing_platforms_and_hidden_cuda_remain_pending():
    required = ["windows_cpu", "physical_cuda", "linux_cpu_gui"]
    report = review_required_lanes(
        required,
        {
            "windows_cpu": {
                "status": "passed",
                "source_stable": True,
                "harness_stable": True,
                "unclosed_omissions": 0,
            },
            "physical_cuda": {
                "status": "passed",
                "source_stable": True,
                "physical_cuda_available": False,
                "harness_stable": True,
                "unclosed_omissions": 0,
            },
        },
    )
    assert set(report["pending"]) == {"physical_cuda", "linux_cpu_gui"}
    assert not report["engineering_lanes_complete"]
    assert report["release_ready"] is False


def test_every_retained_historical_assertion_is_catalogued_without_fake_acceptance():
    root = Path(__file__).resolve().parents[2]
    policy = json.loads(
        (root / "docs/implementation/OMISSION_DISPOSITIONS.json").read_text(encoding="utf-8")
    )
    rows = policy["historical_cases"]
    assert len(rows) == len({row["nodeid"] for row in rows}) == 67
    assert sum(len(row["assertions"]) for row in rows) == 310
    for row in rows:
        path = root / row["nodeid"].split("::")[0]
        assert (
            hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            == row["source_sha256"]
        )
        for target in row["current_successors"]:
            name, function = target.split("::")
            assert f"def {function}(" in (root / name).read_text(encoding="utf-8")


def test_changed_or_unrecorded_validator_cannot_qualify_a_lane():
    for stable in (False, None):
        report = review_required_lanes(
            ["windows_cpu"],
            {
                "windows_cpu": {
                    "status": "passed",
                    "source_stable": True,
                    "harness_stable": stable,
                    "unclosed_omissions": 0,
                }
            },
        )
        assert not report["engineering_lanes_complete"]
        assert "windows_cpu" in report["pending"]


def test_missing_omission_inventory_cannot_qualify_a_lane():
    report = review_required_lanes(
        ["windows_cpu"],
        {"windows_cpu": {"status": "passed", "source_stable": True, "harness_stable": True}},
    )
    assert not report["engineering_lanes_complete"]
    assert "omission inventory" in report["pending"]["windows_cpu"]
