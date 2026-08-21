from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

BUNDLE_ROOT = Path(__file__).resolve().parents[2]


def run(repo: Path, *args: str, check: bool = True, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.run(
        [sys.executable, *args], cwd=repo, text=True, encoding="utf-8", errors="replace",
        capture_output=True, env=merged,
    )
    if check and proc.returncode:
        raise AssertionError(f"command failed: {args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(["git", *args], cwd=repo, text=True, encoding="utf-8", errors="replace", capture_output=True)
    assert proc.returncode == 0, proc.stderr
    return proc


def write(repo: Path, rp: str, text: str) -> None:
    p = repo / rp
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8", newline="\n")


def readj(repo: Path, rp: str):
    return json.loads((repo / rp).read_text(encoding="utf-8"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def tree_hashes(repo: Path, prefix: str = ".ai") -> dict[str, str]:
    root = repo / prefix
    return {
        p.relative_to(repo).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def shard_path(repo: Path, kind: str, source: str) -> Path:
    man = readj(repo, ".ai/index/manifest.json")
    return repo / ".ai/index" / kind / (man["files"][source]["key"] + ".json")


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for rp in ("scripts/ai-index", "scripts/ai-agent-guard.py"):
        dst = repo / rp; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(BUNDLE_ROOT / rp, dst)
    for rp in (".ai/architectural-semantics.json", ".ai/semantic-benchmark.json", ".ai/AI_WORKFLOW.md"):
        dst = repo / rp; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(BUNDLE_ROOT / rp, dst)
    write(repo, ".gitignore", ".ai-cache/\n.ai-tmp/\n")
    write(repo, "pyproject.toml", "[project]\nname='calo-rpd-fixture'\nversion='0'\n")
    write(repo, "AGENTS.md", "# Existing CALO instructions\nPreserve deterministic scientific behavior.\n")
    write(repo, "tests/AGENTS.md", "# Existing test instructions\nUse deterministic tests.\n")
    write(repo, "tests/unit/__init__.py", "")
    write(repo, "tests/unit/helpers.py", "HELPER = 1\n")

    state = '''class AppState:\n    """Authoritative application state."""\n    def __init__(self):\n        self.current_experiment_id = ""\n        self.policy_registry = None\n'''
    policy = '''from . import _policy_registry_core as _core\n\nclass PolicyRegistry(_core.PolicyRegistry):\n    """Public policy registry accounting boundary."""\n    def counted_evaluation_count(self, policy_id):\n        return self.training_evaluation_count(policy_id)\n'''
    core = '''class PolicyRegistry:\n    """Deep lifecycle implementation."""\n    def training_evaluation_count(self, policy_id):\n        return 1\n'''
    problem = '''class ORPDProblem:\n    """Shared ORPD evaluator."""\n    def evaluate(self, normalized):\n        return float(sum(normalized))\n\n    def evaluate_with_context(self, normalized):\n        return self.evaluate(normalized), {"counted": True}\n'''
    runner = '''from calo_rpd_studio.orpd.problem import ORPDProblem\n\ndef execute(values):\n    problem = ORPDProblem()\n    return problem.evaluate(values)\n'''
    db = '''class ResultDatabase:\n    """Durable persistence boundary."""\n    def save(self, value):\n        return value\n'''
    resource = '''class ResourceScheduler:\n    """CUDA to CPU scheduling surface."""\n    def admit(self, request):\n        return "cuda" if request else "cpu"\n'''
    training = '''from . import _tsh_calo_training_campaign_core as _core\n\nclass IndependentTSHCALOTrainingCampaign(_core.IndependentTSHCALOTrainingCampaign):\n    pass\n'''
    training_core = '''class IndependentTSHCALOTrainingCampaign:\n    def run(self):\n        return 1\n'''
    collector = '''def collect(failures):\n    failures.append("x")\n'''
    unrelated = '''class SomeUnrelatedClass:\n    def append(self, value):\n        return value\n'''
    generic = '''def status():\n    return "ok"\n'''
    test_policy = '''from calo_rpd_studio.algorithms.calo.policy_registry import PolicyRegistry\n\ndef test_registry():\n    registry = PolicyRegistry()\n    assert registry.counted_evaluation_count("p") == 1\n'''
    test_generic = '''from calo_rpd_studio import generic_status\n\ndef test_unrelated():\n    # The word status alone must not claim symbol coverage.\n    assert generic_status is not None\n'''
    for rp, text in {
        "calo_rpd_studio/app/state_manager.py": state,
        "calo_rpd_studio/algorithms/calo/policy_registry.py": policy,
        "calo_rpd_studio/algorithms/calo/_policy_registry_core.py": core,
        "calo_rpd_studio/algorithms/calo/tsh_calo_training_campaign.py": training,
        "calo_rpd_studio/algorithms/calo/_tsh_calo_training_campaign_core.py": training_core,
        "calo_rpd_studio/orpd/problem.py": problem,
        "calo_rpd_studio/experiments/experiment_runner.py": runner,
        "calo_rpd_studio/results/database.py": db,
        "calo_rpd_studio/compute/resource_scheduler.py": resource,
        "calo_rpd_studio/app/collector.py": collector,
        "calo_rpd_studio/app/unrelated.py": unrelated,
        "calo_rpd_studio/generic_status.py": generic,
        "tests/unit/test_policy_registry.py": test_policy,
        "tests/unit/test_generic_status.py": test_generic,
    }.items():
        write(repo, rp, text)

    seed = {
        "schema_version": 1,
        "units": {
            "calo_rpd_studio/algorithms/calo/policy_registry.py": {
                "content_hash": sha(policy), "audit_categories": ["logic", "correctness"],
                "last_reviewed_commit": "historical", "scope_note": "fixture reviewed source",
            },
            "calo_rpd_studio/orpd/problem.py": {
                "content_hash": sha(problem), "audit_categories": ["logic", "correctness"],
                "last_reviewed_commit": "historical", "scope_note": "fixture reviewed source",
            },
        },
    }
    write(repo, ".ai/audit-seed.json", json.dumps(seed, indent=2, sort_keys=True) + "\n")
    write(repo, ".ai/findings.json", json.dumps({"schema_version": 1, "findings": [{
        "id": "FIXTURE-1", "severity": "high", "status": "resolved",
        "affected": ["calo_rpd_studio/algorithms/calo/policy_registry.py"],
        "description": "Historical resolved accounting issue.", "resolution_commit": "historical",
    }]}, indent=2) + "\n")

    git(repo, "init")
    git(repo, "config", "user.email", "fixture@example.invalid")
    git(repo, "config", "user.name", "Fixture")
    git(repo, "add", "--", ".")
    git(repo, "commit", "-m", "fixture")
    run(repo, "scripts/ai-agent-guard.py", "--repair", "--root", str(repo))
    return repo


def test_initial_index_check_idempotency_modules_and_initialization(tmp_path: Path):
    repo = make_repo(tmp_path)
    run(repo, "scripts/ai-index", "init")
    run(repo, "scripts/ai-index", "check")
    manifest = readj(repo, ".ai/index/manifest.json")
    change = readj(repo, ".ai/index/change-index.json")
    assert manifest["sharded"] is True
    assert manifest["files"]["calo_rpd_studio/orpd/problem.py"]["module"] == "power-system"
    assert manifest["files"]["calo_rpd_studio/algorithms/calo/policy_registry.py"]["module"] == "calo-policy"
    assert change["initialization"] is True

    legacy = ("file-index.json", "symbol-index.json", "dependency-graph.json", "test-map.json", "audit-coverage.json", "change-index.json")
    assert not [name for name in legacy if (repo / ".ai" / name).exists()]
    assert change["changed_files"] == [] and change["changed_symbols"] == []
    first = tree_hashes(repo)
    run(repo, "scripts/ai-index", "init")
    assert tree_hashes(repo) == first


def test_one_file_locality_and_audit_invalidation(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    before = tree_hashes(repo, ".ai/index")
    policy_audit = readj(repo, shard_path(repo, "audit", "calo_rpd_studio/algorithms/calo/policy_registry.py").relative_to(repo).as_posix())["audit"]
    problem_audit = readj(repo, shard_path(repo, "audit", "calo_rpd_studio/orpd/problem.py").relative_to(repo).as_posix())["audit"]
    assert policy_audit["reviewed"] and not policy_audit["re_audit_required"]
    assert problem_audit["reviewed"] and not problem_audit["re_audit_required"]
    p = repo / "calo_rpd_studio/algorithms/calo/policy_registry.py"
    p.write_text(p.read_text() + "\n# localized source edit\n", encoding="utf-8")
    run(repo, "scripts/ai-index", "update")
    after = tree_hashes(repo, ".ai/index")
    changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
    assert len(changed) <= 8, changed
    assert shard_path(repo, "files", "calo_rpd_studio/orpd/problem.py").relative_to(repo).as_posix() not in changed
    policy_audit = json.loads(shard_path(repo, "audit", "calo_rpd_studio/algorithms/calo/policy_registry.py").read_text())["audit"]
    problem_audit = json.loads(shard_path(repo, "audit", "calo_rpd_studio/orpd/problem.py").read_text())["audit"]
    assert not policy_audit["reviewed"] and policy_audit["re_audit_required"]
    assert problem_audit["reviewed"] and not problem_audit["re_audit_required"]



def test_audit_seed_survives_clean_crlf_materialization_but_not_real_edit(tmp_path: Path):
    repo = make_repo(tmp_path)
    source = "calo_rpd_studio/algorithms/calo/policy_registry.py"
    path = repo / source
    canonical = path.read_text(encoding="utf-8")
    git(repo, "config", "core.autocrlf", "true")
    path.write_bytes(canonical.replace("\n", "\r\n").encode("utf-8"))
    clean = subprocess.run(["git", "diff", "--quiet", "--", source], cwd=repo)
    assert clean.returncode == 0

    run(repo, "scripts/ai-index", "init")
    audit = readj(repo, shard_path(repo, "audit", source).relative_to(repo).as_posix())["audit"]
    assert audit["reviewed"] is True
    assert audit["re_audit_required"] is False

    path.write_bytes((canonical + "\n# semantic edit\n").replace("\n", "\r\n").encode("utf-8"))
    dirty = subprocess.run(["git", "diff", "--quiet", "--", source], cwd=repo)
    assert dirty.returncode != 0
    run(repo, "scripts/ai-index", "update")
    audit = readj(repo, shard_path(repo, "audit", source).relative_to(repo).as_posix())["audit"]
    assert audit["reviewed"] is False
    assert audit["re_audit_required"] is True

def test_rename_delete_and_stale_module_cleanup(tmp_path: Path):
    repo = make_repo(tmp_path)
    write(repo, "calo_bootstrap/only_bootstrap.py", "def launch():\n    return 1\n")
    git(repo, "add", "--", "calo_bootstrap/only_bootstrap.py"); git(repo, "commit", "-m", "bootstrap")
    run(repo, "scripts/ai-index", "init")
    old = "calo_rpd_studio/app/collector.py"; new = "calo_rpd_studio/app/collector_renamed.py"
    old_shard = shard_path(repo, "files", old)
    (repo / old).rename(repo / new)
    run(repo, "scripts/ai-index", "update")
    change = readj(repo, ".ai/index/change-index.json")
    assert change["renames"] == {old: new}
    assert old not in readj(repo, ".ai/index/manifest.json")["files"]
    assert not old_shard.exists()
    (repo / new).unlink(); (repo / "calo_bootstrap/only_bootstrap.py").unlink()
    run(repo, "scripts/ai-index", "update")
    change = readj(repo, ".ai/index/change-index.json")
    assert new in change["deleted_files"]
    # bootstrap has no remaining fixture files, so its generated module shard must disappear.
    modules = [json.loads(p.read_text())["module"] for p in (repo / ".ai/index/modules").glob("*.json")]
    assert "bootstrap" not in modules


def test_malformed_source_nonfatal(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    write(repo, "calo_rpd_studio/app/broken.py", "def broken(:\n")
    proc = run(repo, "scripts/ai-index", "update")
    assert "non-fatal errors" in proc.stdout
    doc = json.loads(shard_path(repo, "files", "calo_rpd_studio/app/broken.py").read_text())["file"]
    assert "SyntaxError" in doc["parse_error"]
    run(repo, "scripts/ai-index", "check")


def test_unknown_receiver_call_stays_unresolved(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    symbol_doc = json.loads(shard_path(repo, "symbols", "calo_rpd_studio/app/collector.py").read_text())
    collect = next(s for s in symbol_doc["symbols"] if s["symbol"] == "collect")
    edge = next(e for e in collect["call_edges"] if e["raw"] == "failures.append")
    assert edge["confidence"] == "unresolved"
    assert "SomeUnrelatedClass.append" not in edge["target"]


def test_generic_symbol_does_not_gain_unrelated_symbol_test_mapping(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    doc = json.loads(shard_path(repo, "tests", "calo_rpd_studio/generic_status.py").read_text())
    assert not any(q.endswith(".status") for q in doc["symbol_to_tests"])
    policy_doc = json.loads(shard_path(repo, "tests", "calo_rpd_studio/algorithms/calo/policy_registry.py").read_text())
    assert any(q.endswith(".PolicyRegistry") for q in policy_doc["symbol_to_tests"])



def test_module_test_mapping_excludes_support_and_instruction_files(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    mapped = json.loads(run(repo, "scripts/ai-index", "query", "get_tests", "calo-policy").stdout)
    assert "tests/AGENTS.md" not in mapped
    assert "tests/unit/__init__.py" not in mapped
    assert "tests/unit/helpers.py" not in mapped
    assert "tests/unit/test_policy_registry.py" in mapped

def test_deterministic_context_queries_and_public_surface_ranking(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    ctx = run(repo, "scripts/ai-index", "context", "where should policy registry accounting be modified", "--no-semantic").stdout
    assert "policy_registry.py" in ctx and "preferred_edit_target" in ctx
    assert ctx.find("policy_registry.py") < ctx.find("_policy_registry_core.py") or "_policy_registry_core.py" not in ctx
    q = run(repo, "scripts/ai-index", "query", "get_recent_changes").stdout
    assert '"initialization": true' in q.lower()
    callers = run(repo, "scripts/ai-index", "query", "get_callers", "calo_rpd_studio.orpd.problem.ORPDProblem").stdout
    assert "experiment_runner" in callers
    # The local variable receiver `problem.evaluate()` is not type-proven, so its method edge remains unresolved.
    runner_doc = json.loads(shard_path(repo, "symbols", "calo_rpd_studio/experiments/experiment_runner.py").read_text())
    execute = next(s for s in runner_doc["symbols"] if s["symbol"] == "execute")
    method_edge = next(e for e in execute["call_edges"] if e["raw"] == "problem.evaluate")
    assert method_edge["confidence"] == "unresolved"



def test_entry_point_routing_understands_natural_launch_language(tmp_path: Path):
    repo = make_repo(tmp_path)
    write(repo, "calo_bootstrap/launcher.py", "def main():\n    return 0\n")
    run(repo, "scripts/ai-index", "init")
    ctx = run(
        repo,
        "scripts/ai-index",
        "context",
        "Where does native application launch begin?",
        "--no-semantic",
    ).stdout
    assert "calo_bootstrap/launcher.py" in ctx
    assert "entry_point" in ctx

def test_semantic_cache_reuse_config_invalidation_and_corrupt_fallback(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    first = json.loads(run(repo, "scripts/ai-index", "embeddings", "update").stdout)
    second = json.loads(run(repo, "scripts/ai-index", "embeddings", "update").stdout)
    assert first["updated"] > 0 and second["reused"] == second["chunks"]
    concepts = repo / ".ai/semantic-concepts.json"
    data = json.loads(concepts.read_text()); data.setdefault("concepts", {})["fixture_new_concept"] = ["registry accounting"]
    concepts.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    third = json.loads(run(repo, "scripts/ai-index", "embeddings", "update").stdout)
    assert third["updated"] == third["chunks"]
    cache = repo / ".ai-cache/embeddings.sqlite3"
    cache.write_bytes(b"not sqlite")
    ctx = run(repo, "scripts/ai-index", "context", "policy registry", "--semantic").stdout
    assert "policy_registry.py" in ctx
    assert cache.exists()


def test_utf8_redirected_cli_output(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    proc = run(repo, "scripts/ai-index", "context", "Where is scientist-facing state — policy?", "--no-semantic", env={"PYTHONIOENCODING": "utf-8"})
    assert proc.returncode == 0 and "Repository intelligence context" in proc.stdout


def test_precommit_source_ai_commit_then_init_is_byte_stable(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    p = repo / "calo_rpd_studio/app/state_manager.py"; p.write_text(p.read_text() + "\n# committed edit\n", encoding="utf-8")
    run(repo, "scripts/ai-index", "precommit", "--no-semantic")
    git(repo, "add", "--", "calo_rpd_studio/app/state_manager.py", ".ai")
    git(repo, "commit", "-m", "source and intelligence")
    before = tree_hashes(repo)
    run(repo, "scripts/ai-index", "init")
    assert tree_hashes(repo) == before


def test_agent_guard_detects_and_repairs_all_agent_files(tmp_path: Path):
    repo = make_repo(tmp_path)
    run(repo, "scripts/ai-agent-guard.py", "--check", "--root", str(repo))
    nested = repo / "tests/AGENTS.md"
    original_specific = "# Existing test instructions\nUse deterministic tests."
    text = nested.read_text(encoding="utf-8")
    nested.write_text(text.split("<!-- REPOSITORY_INTELLIGENCE_PROTECTED:END v2 -->", 1)[-1].lstrip(), encoding="utf-8")
    assert run(repo, "scripts/ai-agent-guard.py", "--check", "--root", str(repo), check=False).returncode != 0
    run(repo, "scripts/ai-agent-guard.py", "--repair", "--root", str(repo))
    repaired = nested.read_text(encoding="utf-8")
    assert repaired.startswith("<!-- REPOSITORY_INTELLIGENCE_PROTECTED:BEGIN v2 -->")
    assert original_specific in repaired
    policy = readj(repo, ".ai/agent-policy.json")
    assert "tests/AGENTS.md" in policy["agent_files"]



def test_guard_install_hook_preserves_existing_hook(tmp_path: Path):
    repo = make_repo(tmp_path)
    hooks = repo / ".git/hooks"
    existing = hooks / "pre-commit"
    existing.write_text("#!/bin/sh\necho existing-hook\n", encoding="utf-8")
    existing.chmod(0o755)
    run(repo, "scripts/ai-agent-guard.py", "--install-hook", "--root", str(repo))
    text = existing.read_text(encoding="utf-8")
    assert "CALO_REPO_INTELLIGENCE_GUARD_V2" in text
    assert "ai-index precommit --no-semantic" in text
    chained = hooks / "pre-commit.user"
    assert chained.exists() and "existing-hook" in chained.read_text(encoding="utf-8")


def test_workflow_is_read_only_and_guarded():
    text = (BUNDLE_ROOT / ".github/workflows/ai-index.yml").read_text(encoding="utf-8")
    assert "contents: read" in text
    assert "persist-credentials: false" in text
    assert "ai-agent-guard.py --check" in text
    assert "ai-index guard --no-semantic" in text
    assert "git diff --exit-code -- .ai" in text
    assert "git push" not in text and "git commit" not in text


def test_repository_specific_semantic_benchmark(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    run(repo, "scripts/ai-index", "embeddings", "update")
    result = json.loads(run(repo, "scripts/ai-index", "embeddings", "benchmark", "--check").stdout)
    assert result["passed"] is True
    assert result["cases"] >= 5
