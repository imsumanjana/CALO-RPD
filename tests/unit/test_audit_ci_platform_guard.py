"""The CI safety boundary covers both OS targets without suppressing diagnostics."""

import ast
from pathlib import Path
import re
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_ci_checks_both_platforms_with_the_full_safety_boundary():
    workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    step = next(
        s
        for s in workflow["jobs"]["source"]["steps"]
        if s["name"] == "Verify the typed safety boundary"
    )
    script = step["run"]
    assert step["shell"] == "bash"
    assert "for platform in linux win32" in script
    assert '--platform "$platform"' in script
    assert "--follow-imports=skip" in script and "--check-untyped-defs" in script
    files = set(re.findall(r"calo_rpd_studio/[\w/]+\.py", script))
    assert len(files) == 23
    assert "calo_rpd_studio/compute/device_lease.py" in files
    assert "calo_rpd_studio/compute/memory_budget.py" in files
    assert "calo_rpd_studio/scripts/verify_release_ci_contract.py" in files


def test_windows_dispatch_is_visible_to_the_type_checker():
    source = (ROOT / "calo_rpd_studio/compute/device_lease.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for name in ("_lock_stream", "_unlock_stream"):
        function = next(
            n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name
        )
        conditions = [ast.unparse(n.test) for n in ast.walk(function) if isinstance(n, ast.If)]
        assert "sys.platform == 'win32'" in conditions
        assert not any("os.name" in condition for condition in conditions)
