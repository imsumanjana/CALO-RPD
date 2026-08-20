from __future__ import annotations

import json
from pathlib import Path
import subprocess

from tools.ai_intelligence.context import build_context
from tools.ai_intelligence import indexer


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def fixture_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init")
    git(repo, "config", "user.email", "ai-index@example.invalid")
    git(repo, "config", "user.name", "AI Index Test")
    (repo / "pkg").mkdir()
    (repo / "tests").mkdir()
    (repo / "pkg/__init__.py").write_text("", encoding="utf-8")
    (repo / "pkg/a.py").write_text(
        "from pkg.b import helper\n\ndef run():\n    return helper()\n", encoding="utf-8"
    )
    (repo / "pkg/b.py").write_text("def helper():\n    return 1\n", encoding="utf-8")
    (repo / "tests/test_a.py").write_text(
        "from pkg.a import run\n\ndef test_run():\n    assert run() == 1\n", encoding="utf-8"
    )
    git(repo, "add", ".")
    git(repo, "commit", "-m", "fixture")
    return repo


def test_init_is_deterministic_and_maps_dependencies(tmp_path: Path):
    repo = fixture_repo(tmp_path)
    indexer.update_index(repo, initialize=True)
    first = (repo / ".ai/symbol-index.json").read_text(encoding="utf-8")
    indexer.update_index(repo)
    second = (repo / ".ai/symbol-index.json").read_text(encoding="utf-8")
    assert first == second
    deps = json.loads((repo / ".ai/dependency-graph.json").read_text(encoding="utf-8"))
    assert any(e["source"] == "pkg/a.py" and e["target"] == "pkg/b.py" for e in deps["file_edges"])
    tests = json.loads((repo / ".ai/test-map.json").read_text(encoding="utf-8"))
    assert "tests/test_a.py" in tests["file_to_tests"]["pkg/a.py"]


def test_incremental_change_marks_prior_audit_for_reaudit(tmp_path: Path):
    repo = fixture_repo(tmp_path)
    indexer.update_index(repo, initialize=True)
    audit_path = repo / ".ai/audit-coverage.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["units"]["pkg/a.py"].update(
        reviewed=True,
        audit_categories=["logic"],
        last_reviewed_commit=git(repo, "rev-parse", "HEAD"),
    )
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (repo / "pkg/a.py").write_text(
        "from pkg.b import helper\n\ndef run():\n    return helper() + 1\n", encoding="utf-8"
    )
    result = indexer.update_index(repo)
    assert result["changed"] == ["pkg/a.py"]
    updated = json.loads(audit_path.read_text(encoding="utf-8"))["units"]["pkg/a.py"]
    assert updated["reviewed"] is False
    assert updated["re_audit_required"] is True
    assert updated["previous_review"]["audit_categories"] == ["logic"]


def test_malformed_python_is_recorded_not_fatal(tmp_path: Path):
    repo = fixture_repo(tmp_path)
    (repo / "pkg/broken.py").write_text("def broken(:\n", encoding="utf-8")
    indexer.update_index(repo, initialize=True)
    files = json.loads((repo / ".ai/file-index.json").read_text(encoding="utf-8"))["files"]
    assert files["pkg/broken.py"]["parse_error"].startswith("SyntaxError:")


def test_update_reparses_only_changed_python_files(tmp_path: Path, monkeypatch):
    repo = fixture_repo(tmp_path)
    indexer.update_index(repo, initialize=True)
    original = indexer.analyze_python
    seen: list[str] = []

    def recording(root: Path, path: str):
        seen.append(path)
        return original(root, path)

    monkeypatch.setattr(indexer, "analyze_python", recording)
    (repo / "pkg/a.py").write_text(
        "from pkg.b import helper\n\ndef run():\n    return helper() + 2\n", encoding="utf-8"
    )
    indexer.update_index(repo)
    assert seen == ["pkg/a.py"]


def test_context_is_targeted(tmp_path: Path):
    repo = fixture_repo(tmp_path)
    (repo / ".ai").mkdir(exist_ok=True)
    (repo / ".ai/REPO_MAP.md").write_text("# Repo\nTiny fixture.\n", encoding="utf-8")
    indexer.update_index(repo, initialize=True)
    context = build_context(repo, "why does run call helper?", limit=2)
    assert "pkg/a.py" in context
    assert "helper" in context
