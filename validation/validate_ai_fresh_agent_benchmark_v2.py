#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    expected_file: str
    source_pattern: str
    distractor: str = ""


CASES = (
    Case(
        "authoritative-app-state",
        "Where is authoritative shared application state owned and changed?",
        "calo_rpd_studio/app/state_manager.py",
        r"class\s+AppState\b",
    ),
    Case(
        "experiment-execution",
        "Where is deterministic experiment execution orchestrated, including failures and exact evaluation counts?",
        "calo_rpd_studio/experiments/experiment_runner.py",
        r"class\s+(CompletedRun|FailedRun|RunExecutionFailure)\b",
    ),
    Case(
        "orpd-evaluation-authority",
        "Where are authoritative ORPD evaluation semantics implemented?",
        "calo_rpd_studio/orpd/problem.py",
        r"class\s+ORPDProblem\b",
    ),
    Case(
        "compute-admission",
        "Where should CUDA CPU resource admission and scheduling behavior be changed?",
        "calo_rpd_studio/compute/resource_scheduler.py",
        r"(class|def)\s+\w+",
    ),
    Case(
        "durable-results",
        "Where is durable experiment result and provenance persistence owned?",
        "calo_rpd_studio/results/database.py",
        r"class\s+ResultDatabase\b",
    ),
    Case(
        "policy-registry-public-boundary",
        "Where should exact policy counted-work accounting be modified without editing the deep lifecycle core?",
        "calo_rpd_studio/algorithms/calo/policy_registry.py",
        r"class\s+PolicyRegistry\b",
        "calo_rpd_studio/algorithms/calo/_policy_registry_core.py",
    ),
    Case(
        "training-campaign-public-boundary",
        "Where should public TSH-CALO training campaign integration behavior be modified?",
        "calo_rpd_studio/algorithms/calo/tsh_calo_training_campaign.py",
        r"class\s+IndependentTSHCALOTrainingCampaign\b",
        "calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py",
    ),
    Case(
        "native-entry-point",
        "Where does native application launch begin?",
        "calo_bootstrap/launcher.py",
        r"def\s+main\b",
    ),
)


def run(root: Path, *args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        args,
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n{proc.stdout}\n{proc.stderr}"
        )
    return proc


def context(root: Path, question: str, semantic: bool) -> str:
    args = [sys.executable, "scripts/ai-index", "context", question, "--semantic" if semantic else "--no-semantic"]
    env = os.environ.copy()
    if semantic:
        env["AI_INDEX_EMBEDDING_PROVIDER"] = "concept"
    return run(root, *args, env=env).stdout


def first_pos(text: str, needle: str) -> int:
    return text.find(needle)


def assert_source(root: Path, case: Case) -> None:
    path = root / case.expected_file
    if not path.is_file():
        raise AssertionError(f"{case.id}: expected source file is missing: {case.expected_file}")
    source = path.read_text(encoding="utf-8", errors="replace")
    if not re.search(case.source_pattern, source):
        raise AssertionError(
            f"{case.id}: independent source verification failed; pattern {case.source_pattern!r} not found in {case.expected_file}"
        )


def check_context(case: Case, mode: str, text: str) -> dict:
    expected_pos = first_pos(text, case.expected_file)
    if expected_pos < 0:
        raise AssertionError(f"{case.id}/{mode}: expected file absent from context: {case.expected_file}")
    distractor_pos = -1
    if case.distractor:
        distractor_pos = first_pos(text, case.distractor)
        if distractor_pos >= 0 and distractor_pos < expected_pos:
            raise AssertionError(
                f"{case.id}/{mode}: distractor ranked ahead of preferred edit target: {case.distractor}"
            )
    encoded = text.encode("utf-8")
    if len(encoded) > 120_000:
        raise AssertionError(f"{case.id}/{mode}: context is unexpectedly large: {len(encoded)} bytes")
    return {
        "mode": mode,
        "bytes": len(encoded),
        "lines": text.count("\n") + (1 if text else 0),
        "expected_position": expected_pos,
        "distractor_position": distractor_pos,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / ".ai-tmp" / "fresh-agent-benchmark-v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Fresh-agent bootstrap contract: curated map first, then freshness, then targeted context.
    repo_map = (root / ".ai/REPO_MAP.md").read_text(encoding="utf-8")
    if ".ai/test-map.json" in repo_map:
        raise AssertionError("REPO_MAP still points fresh agents to deleted monolithic .ai/test-map.json")
    if "query get_tests" not in repo_map:
        raise AssertionError("REPO_MAP does not teach fresh agents to use the v2 get_tests query")

    run(root, sys.executable, "scripts/ai-index", "check")
    run(root, sys.executable, "scripts/ai-agent-guard.py", "--check", "--root", ".")

    # Ensure semantic cache is available, but benchmark correctness never relies on it exclusively.
    sem_env = os.environ.copy()
    sem_env["AI_INDEX_EMBEDDING_PROVIDER"] = "concept"
    run(root, sys.executable, "scripts/ai-index", "embeddings", "update", env=sem_env)

    results: list[dict] = []
    for case in CASES:
        assert_source(root, case)
        det = context(root, case.question, semantic=False)
        sem = context(root, case.question, semantic=True)
        (out_dir / f"{case.id}-deterministic.txt").write_text(det, encoding="utf-8", newline="\n")
        (out_dir / f"{case.id}-semantic.txt").write_text(sem, encoding="utf-8", newline="\n")
        results.append(
            {
                **asdict(case),
                "source_verified": True,
                "deterministic": check_context(case, "deterministic", det),
                "semantic": check_context(case, "semantic", sem),
            }
        )

    # Public-vs-core routing must also be represented in curated architectural metadata.
    semantics = json.loads((root / ".ai/architectural-semantics.json").read_text(encoding="utf-8"))
    rules = {r.get("pattern"): r for r in semantics.get("file_rules", [])}
    public = rules.get("calo_rpd_studio/algorithms/calo/policy_registry.py", {})
    core = rules.get("calo_rpd_studio/algorithms/calo/_policy_registry_core.py", {})
    if public.get("preferred_edit_target") is not True or public.get("public_surface") is not True:
        raise AssertionError("policy_registry.py is not curated as the preferred public edit target")
    if core.get("preferred_edit_target") is not False or core.get("internal_only") is not True:
        raise AssertionError("_policy_registry_core.py is not curated as an internal non-preferred core")

    # Test mapping: require executable tests and reject support/instruction files.
    mapped = json.loads(
        run(
            root,
            sys.executable,
            "scripts/ai-index",
            "query",
            "get_tests",
            "calo_rpd_studio.algorithms.calo.policy_registry.PolicyRegistry",
        ).stdout
    )
    if not isinstance(mapped, list) or not mapped:
        raise AssertionError("PolicyRegistry has no mapped tests")
    bad = [
        p for p in mapped
        if p.lower().endswith(("agents.md", "agent.md", "__init__.py", "conftest.py", "helpers.py"))
    ]
    if bad:
        raise AssertionError("Non-test support files remain mapped to PolicyRegistry: " + ", ".join(bad))
    if not any(re.search(r"(^|/)test_.*\.py$", p) for p in mapped):
        raise AssertionError("PolicyRegistry mapping contains no executable pytest source")

    recent = json.loads(run(root, sys.executable, "scripts/ai-index", "query", "get_recent_changes").stdout)
    findings = json.loads(run(root, sys.executable, "scripts/ai-index", "query", "get_findings").stdout)
    if recent.get("initialization") is True:
        raise AssertionError("Fresh-agent recent changes still report migration initialization")
    if recent.get("newly_indexed_files"):
        raise AssertionError("Fresh-agent recent changes leak migration files as newly indexed")

    report = {
        "schema": "calo-fresh-agent-benchmark-v2",
        "cases": len(results),
        "passed": True,
        "results": results,
        "policy_registry_mapped_tests": mapped,
        "recent_changes": recent,
        "findings": findings,
        "notes": {
            "source_verification": "Each expected route was independently checked against the source file for an authoritative symbol/pattern.",
            "deterministic_required": True,
            "semantic_required_to_not_regress_architectural_ordering": True,
            "scientific_runtime_executed": False,
        },
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print("Fresh-agent benchmark passed.")
    print(f"Cases passed: {len(results)}/{len(results)}")
    for item in results:
        print(
            f"- {item['id']}: {item['expected_file']} | "
            f"det={item['deterministic']['bytes']}B sem={item['semantic']['bytes']}B"
        )
    print(f"PolicyRegistry mapped executable tests: {len(mapped)}")
    print(f"Recent changed files: {len(recent.get('changed_files', []))}")
    print(f"Findings records returned: {len(findings) if isinstance(findings, list) else 'object'}")
    print(f"Report: {report_path}")
    print("No scientific training, experiment, GUI, qualification, or protected-case workload was run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
