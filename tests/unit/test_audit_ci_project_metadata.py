"""Source CI installs real project metadata while retaining the locked dependency environment."""

from pathlib import Path
import shlex
import yaml


def test_source_ci_installs_project_after_locked_dependencies_before_validation():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.safe_load((root / ".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["source"]["steps"]
    names = [step["name"] for step in steps]
    install = names.index("Install project metadata without changing locked dependencies")
    assert names.index("Install the hash-locked validation environment") < install
    assert install < names.index("Verify exact, hash-complete dependency locks")
    assert install < names.index("Run unit, integration, regression, and scientific tests")
    command = shlex.split(steps[install]["run"])
    assert command == [
        "python",
        "-m",
        "pip",
        "install",
        "--no-deps",
        "--no-build-isolation",
        "--editable",
        ".",
    ]
    test_command = steps[names.index("Run unit, integration, regression, and scientific tests")][
        "run"
    ]
    assert "test_packaged_gui_validator" not in test_command
    assert "--fail-under=60" in test_command
