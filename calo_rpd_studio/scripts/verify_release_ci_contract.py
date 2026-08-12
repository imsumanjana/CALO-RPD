"""Verify that CI retains the v12 release-preparation development contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# PyYAML does not publish inline typing metadata. Runtime structure checks below remain the
# authoritative boundary for this untrusted workflow document.
import yaml  # type: ignore[import-untyped]


REQUIRED_JOBS = {
    "source",
    "compatibility",
    "headless-gui",
    "artifact",
    "cpu-image",
    "cuda-image",
    "physical-cuda",
}
REQUIRED_SOURCE_TOKENS = (
    "persist-credentials: false",
    "provenance: mode=max",
    "sbom: true",
    "--read-only",
    "--cap-drop ALL",
    "no-new-privileges",
    "--require-physical-cuda",
    "generate_distribution_manifests",
    "release_policy_scope.py",
    "create_release_preparation.py",
    "finalize_release_records.py",
)


def verify(path: Path) -> dict:
    source = path.resolve(strict=True).read_text(encoding="utf-8")
    payload = yaml.safe_load(source)
    if not isinstance(payload, dict) or not isinstance(payload.get("jobs"), dict):
        raise ValueError("CI workflow is not a job-bearing YAML document")
    jobs = set(payload["jobs"])
    missing_jobs = sorted(REQUIRED_JOBS - jobs)
    missing_tokens = [token for token in REQUIRED_SOURCE_TOKENS if token not in source]
    if missing_jobs or missing_tokens:
        raise ValueError(
            f"CI release contract is incomplete; jobs={missing_jobs}, tokens={missing_tokens}"
        )
    physical = payload["jobs"]["physical-cuda"]
    if "workflow_dispatch" not in str(physical.get("if", "")):
        raise ValueError("Physical CUDA CI must remain explicitly dispatch-gated")
    return {
        "schema": "calo-v12-release-ci-contract-v1",
        "workflow": path.as_posix(),
        "required_jobs": sorted(REQUIRED_JOBS),
        "missing_job_count": 0,
        "missing_contract_token_count": 0,
        "physical_cuda_dispatch_gated": True,
        "publication_or_release_executed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, default=Path(".github/workflows/ci.yml"))
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = verify(arguments.workflow)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
