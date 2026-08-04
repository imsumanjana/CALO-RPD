from __future__ import annotations

from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[2]


def test_image_drops_root_before_runtime_entrypoint():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "python:3.11-slim-bookworm@sha256:" in dockerfile
    assert "useradd --uid 10001" in dockerfile
    assert dockerfile.index("USER 10001:10001") < dockerfile.index("ENTRYPOINT")
    assert "HOME=/data/home/calo" in dockerfile
    assert "--require-hashes" in dockerfile
    assert "${RUNTIME_LOCK}" in dockerfile
    assert "COPY containers/debian.sources" in dockerfile
    assert (
        "COPY requirements-lock-cpu-py311-linux.txt requirements-lock-cuda128-py311-linux.txt pyproject.toml README.md LICENSE ./"
        in dockerfile
    )
    assert "SOURCE_COMMIT=unavailable" in dockerfile
    assert "SOURCE_TRACKED_CLEAN=false" in dockerfile
    assert "calo_rpd_studio.compute.source_identity" in dockerfile
    assert "python -m pip check" in dockerfile
    assert "libxcb-shape0" in dockerfile
    assert "ctypes.CDLL(str(plugin))" in dockerfile
    assert "python -m pip uninstall --yes setuptools wheel" in dockerfile
    assert "python -m pip uninstall --yes pip" in dockerfile
    debian_sources = (ROOT / "containers/debian.sources").read_text(encoding="utf-8")
    assert debian_sources.count("20260728T000000Z") == 2
    assert "deb.debian.org" not in debian_sources


def test_compose_profiles_are_local_only_read_only_and_non_privileged():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    common = compose["x-calo-common"]
    assert common["ports"] == ["127.0.0.1:${CALO_PORT:-6080}:6080"]
    assert common["user"] == "10001:10001"
    assert common["read_only"] is True
    assert common["cap_drop"] == ["ALL"]
    assert common["security_opt"] == ["no-new-privileges:true"]
    assert common["mem_limit"] == "${CALO_HOST_MEMORY_LIMIT:-24g}"
    assert any(item.startswith("/tmp:rw,nosuid,nodev") for item in common["tmpfs"])
    for service_name in ("cpu", "cuda"):
        service = compose["services"][service_name]
        assert service["environment"]["CALO_DEVICE_LEASE_DIR"] == "/data/device-leases"
        assert service["build"]["args"]["SOURCE_COMMIT"] == "${CALO_SOURCE_COMMIT:-unavailable}"
        assert service["build"]["args"]["SOURCE_TRACKED_CLEAN"] == (
            "${CALO_SOURCE_TRACKED_CLEAN:-false}"
        )
    assert compose["volumes"]["calo-runtime"]["name"] == (
        "${CALO_RUNTIME_VOLUME:-calo-rpd-studio-runtime}"
    )


def test_cuda_profile_requests_one_explicit_nvidia_device():
    compose = yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))
    cuda = compose["services"]["cuda"]
    devices = cuda["deploy"]["resources"]["reservations"]["devices"]
    assert devices == [
        {
            "driver": "nvidia",
            "device_ids": ["${CALO_GPU_DEVICE:-0}"],
            "capabilities": ["gpu"],
        }
    ]
    assert cuda["environment"]["NVIDIA_VISIBLE_DEVICES"] == "${CALO_GPU_DEVICE:-0}"
    assert cuda["build"]["args"]["RUNTIME_LOCK"] == ("requirements-lock-cuda128-py311-linux.txt")


def test_container_context_excludes_generated_policies_and_user_build_notes():
    patterns = {
        line.strip()
        for line in (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert "*.pt" in patterns
    assert "*.pt.sha256" in patterns
    assert "*_lineage" in patterns
    assert "calo_rpd_studio/data/trained_models/*" in patterns
    assert "!calo_rpd_studio/data/trained_models/__init__.py" in patterns
    assert "Docker_Build.txt" in patterns
    assert "publication_export" in patterns
    assert "results_data" in patterns
    assert "**/__pycache__/" in patterns
    assert "**/*.pyc" in patterns
    assert "**/*.pyo" in patterns
    assert "**/AGENTS.md" in patterns


def test_vnc_server_is_reachable_only_from_the_container_loopback():
    entrypoint = (ROOT / "containers" / "entrypoint.py").read_text(encoding="utf-8")
    assert '"-localhost"' in entrypoint
    assert "_supervise_children" in entrypoint
    assert "CALO_APP_PID_FILE" in entrypoint
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "os.kill(pid,0)" in dockerfile


def test_cpu_and_cuda_runtime_locks_are_hash_complete_and_backend_specific():
    expected = {
        "requirements-lock-cpu-py311-linux.txt": (
            "https://download.pytorch.org/whl/cpu",
            "torch==2.10.0+cpu",
        ),
        "requirements-lock-cuda128-py311-linux.txt": (
            "https://download.pytorch.org/whl/cu128",
            "torch==2.10.0+cu128",
        ),
    }
    requirement_pattern = re.compile(
        r"(?ms)^([A-Za-z0-9][A-Za-z0-9_.-]*==[^\s\\]+).*?(?=^[A-Za-z0-9][A-Za-z0-9_.-]*==|\Z)"
    )
    for filename, (index, torch_pin) in expected.items():
        content = (ROOT / filename).read_text(encoding="utf-8")
        assert f"--extra-index-url {index}" in content
        assert torch_pin in content
        blocks = requirement_pattern.findall(content)
        assert len(blocks) >= 25
        for match in requirement_pattern.finditer(content):
            assert "--hash=sha256:" in match.group(0), match.group(1)


def test_ci_lock_and_workflow_are_reproducible_and_supply_chain_pinned():
    lock = (ROOT / "requirements-lock-ci-py311-linux.txt").read_text(encoding="utf-8")
    assert "uv==0.11.29" in lock
    assert "ruff==0.15.22" in lock
    assert "mypy==1.20.2" in lock
    assert "torch==2.10.0+cpu" in lock
    requirement_pattern = re.compile(
        r"(?ms)^([A-Za-z0-9][A-Za-z0-9_.-]*==[^\s\\]+).*?(?=^[A-Za-z0-9][A-Za-z0-9_.-]*==|\Z)"
    )
    for match in requirement_pattern.finditer(lock):
        assert "--hash=sha256:" in match.group(0), match.group(1)

    workflow_text = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    assert set(workflow["jobs"]) == {
        "source",
        "compatibility",
        "headless-gui",
        "artifact",
        "cpu-image",
        "cuda-image",
        "physical-cuda",
    }
    action_references = re.findall(r"(?m)^\s*uses:\s*([^\s#]+)", workflow_text)
    assert action_references
    assert all(re.fullmatch(r"[^@]+@[0-9a-f]{40}", item) for item in action_references)
    assert "--require-hashes" in workflow_text
    assert "--read-only" in workflow_text
    assert "sbom: true" in workflow_text
    assert "provenance: mode=max" in workflow_text
    assert "inputs.run_physical_cuda" in workflow_text
    assert "validate_packaged_gui" in workflow_text
    assert '--forbid-import-root "$GITHUB_WORKSPACE"' in workflow_text
    assert "(cd /tmp && PYTHONPATH= QT_QPA_PLATFORM=offscreen" in workflow_text
    compatibility = workflow["jobs"]["compatibility"]
    compatibility_text = yaml.safe_dump(compatibility, sort_keys=True)
    assert "python -m pip check" in compatibility_text
    assert "python -m pip freeze --all > compatibility-environment.txt" in compatibility_text
    assert "compatibility-${{ runner.os }}-py${{ matrix.python }}-environment" in compatibility_text
