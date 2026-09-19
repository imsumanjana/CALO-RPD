"""Opt-in pytest evidence: inventory every outcome, deselection and collection omission."""

from __future__ import annotations
import json
import os
import platform
from pathlib import Path
import sys
import time

_REPORT = {}
_CONFIG = None


def pytest_addoption(parser):
    group = parser.getgroup("calo-acceptance")
    group.addoption("--evidence-output", help="Exact JSON evidence output path")
    group.addoption("--omit-protected-cases", action="store_true", default=False)


def pytest_configure(config):
    global _CONFIG, _REPORT
    _CONFIG = config
    _REPORT = {
        "schema": "calo-test-execution-inventory-v1",
        "started": time.time(),
        "argv": list(sys.argv),
        "platform": platform.platform(),
        "python": sys.version,
        "lane": os.environ.get("CALO_ACCEPTANCE_LANE", "unspecified"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "qt_platform": os.environ.get("QT_QPA_PLATFORM"),
        "selected": [],
        "deselected": [],
        "collection": [],
        "outcomes": {},
    }


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--omit-protected-cases"):
        return
    retained, omitted = [], []
    for item in items:
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        protected = any(
            str(value) in {"case118", "case300"} for value in parameters.values()
        ) or any(case in item.nodeid for case in ("case118", "case300"))
        scientific = "/tests/scientific/" in str(item.path).replace("\\", "/")
        if scientific and protected:
            item._calo_omission_reason = "protected-case scientific execution not authorized"
            omitted.append(item)
        else:
            retained.append(item)
    if omitted:
        items[:] = retained
        config.hook.pytest_deselected(items=omitted)


def _write_partial_inventory():
    """Retain collection and completed outcomes even when a later process is interrupted."""
    if _CONFIG is None:
        return
    target = _CONFIG.getoption("--evidence-output")
    if target:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(_REPORT, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        # Windows readers/scanners can briefly deny replacement of an open report. Retry
        # boundedly; a persistent evidence-write error still fails the test process closed.
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.025 * (attempt + 1))


def pytest_deselected(items):
    _REPORT["deselected"].extend(
        {
            "nodeid": item.nodeid,
            "reason": getattr(
                item, "_calo_omission_reason", "explicit pytest selection; inspect recorded argv"
            ),
        }
        for item in items
    )


def pytest_collection_finish(session):
    _REPORT["selected"] = [item.nodeid for item in session.items]
    _REPORT["exit_code"] = None
    _REPORT["not_executed"] = list(_REPORT["selected"])
    _write_partial_inventory()


def pytest_collectreport(report):
    if report.failed or report.skipped:
        _REPORT["collection"].append(
            {
                "nodeid": report.nodeid,
                "outcome": report.outcome,
                "reason": str(report.longrepr),
            }
        )


def pytest_runtest_logreport(report):
    item = _REPORT["outcomes"].setdefault(report.nodeid, {"phases": {}})
    item["phases"][report.when] = {
        "outcome": report.outcome,
        "seconds": report.duration,
        "wasxfail": str(getattr(report, "wasxfail", "")),
        "reason": str(report.longrepr) if report.longrepr else "",
        "skip_message": (
            str(report.longrepr[2]).removeprefix("Skipped: ")
            if report.skipped and isinstance(report.longrepr, tuple)
            else ""
        ),
    }

    item["outcome"] = _classify(item)
    if report.when == "teardown":
        _REPORT["not_executed"] = sorted(set(_REPORT["selected"]) - set(_REPORT["outcomes"]))
        _write_partial_inventory()


def _classify(item):
    phases = item["phases"]
    values = [phase["outcome"] for phase in phases.values()]
    return (
        "failed"
        if "failed" in values
        else "skipped"
        if "skipped" in values
        else "xpassed"
        if any(phase["wasxfail"] for phase in phases.values())
        else "passed"
        if phases.get("call", {}).get("outcome") == "passed"
        and phases.get("teardown", {}).get("outcome") == "passed"
        else "incomplete"
    )


def pytest_sessionfinish(session, exitstatus):
    outcomes = _REPORT["outcomes"]
    for item in outcomes.values():
        phases = item["phases"]
        values = [phase["outcome"] for phase in phases.values()]
        item["outcome"] = (
            "failed"
            if "failed" in values
            else "skipped"
            if "skipped" in values
            else "xpassed"
            if any(p["wasxfail"] for p in phases.values())
            else "passed"
            if phases.get("call", {}).get("outcome") == "passed"
            and phases.get("teardown", {}).get("outcome") == "passed"
            else "incomplete"
        )
    _REPORT["not_executed"] = sorted(set(_REPORT["selected"]) - set(outcomes))
    torch = sys.modules.get("torch")
    if torch is not None:
        available = bool(torch.cuda.is_available())
        _REPORT["hardware"] = {
            "torch_version": str(torch.__version__),
            "compiled_cuda": str(torch.version.cuda),
            "physical_cuda_available": available,
            "devices": [
                torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
            ]
            if available
            else [],
        }
    _REPORT.update(exit_code=int(exitstatus), finished=time.time(), release_ready=False)
    target = _CONFIG.getoption("--evidence-output")
    if target:
        path = Path(target)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_REPORT, indent=2, sort_keys=True) + "\n", encoding="utf-8")
