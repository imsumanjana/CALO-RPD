#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

TEST_CLASSIFIER = r'''TEST_TREE_PREFIXES = ("tests/", "test/", "spec/", "specs/", "__tests__/")
TEST_CODE_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".go", ".rs", ".rb", ".php", ".java", ".kt", ".kts", ".cs",
    ".c", ".cc", ".cpp", ".cxx",
}


def is_test_path(path: str) -> bool:
    """Return True only for executable test source files, not every file under tests/."""
    normalized = path.lower().replace("\\", "/")
    name = Path(normalized).name
    suffix = Path(name).suffix
    if suffix not in TEST_CODE_SUFFIXES:
        return False
    if name in {"__init__.py", "conftest.py"}:
        return False
    if name.startswith("test_"):
        return True
    if name.endswith(("_test.py", "_test.go", "_test.rs", "_test.rb", "_test.php")):
        return True
    if any(marker in name for marker in (".test.", ".spec.")):
        return True
    return False


def is_test_support_path(path: str) -> bool:
    """Return True for files that live in test/support trees and must not be source targets."""
    normalized = path.lower().replace("\\", "/")
    return normalized.startswith(TEST_TREE_PREFIXES) or is_test_path(path)
'''


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    proc = subprocess.run(args, cwd=root, text=True, encoding="utf-8", errors="replace", capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    engine = root / "scripts" / "ai-index"
    tests = root / "tests" / "tooling" / "test_ai_repo_intelligence_v2.py"
    if not engine.is_file() or not tests.is_file():
        raise SystemExit("Installed v2 engine/tests not found. Run the migration/remediations first.")

    original = engine.read_text(encoding="utf-8")
    if "CALO_PROFILE_VERSION" not in original:
        raise SystemExit("scripts/ai-index is not the expected CALO v2 engine; refusing to patch.")

    text = original
    classifier_pattern = r"def is_test_path\(path: str\) -> bool:\n.*?\n\n(?=def test_map)"
    matches = list(re.finditer(classifier_pattern, text, flags=re.S))
    if len(matches) != 1:
        raise SystemExit(f"Unable to identify the test classifier uniquely; found {len(matches)} matches.")
    text = re.sub(classifier_pattern, TEST_CLASSIFIER + "\n\n", text, count=1, flags=re.S)

    old_sources = "sources = [p for p in idx if not is_test_path(p)]"
    source_count = text.count(old_sources)
    if source_count < 1:
        raise SystemExit("Unable to find test-map source classification.")
    text = text.replace(old_sources, "sources = [p for p in idx if not is_test_support_path(p)]")
    engine.write_text(text, encoding="utf-8", newline="\n")

    test_text = tests.read_text(encoding="utf-8")
    if 'write(repo, "tests/unit/__init__.py", "")' not in test_text:
        anchor = '    write(repo, "tests/AGENTS.md", "# Existing test instructions\\nUse deterministic tests.\\n")\n'
        addition = (
            '    write(repo, "tests/unit/__init__.py", "")\n'
            '    write(repo, "tests/unit/helpers.py", "HELPER = 1\\n")\n'
        )
        if anchor not in test_text:
            raise SystemExit("Unable to add conservative test-file fixture coverage.")
        test_text = test_text.replace(anchor, anchor + addition, 1)

    regression = r'''

def test_module_test_mapping_excludes_support_and_instruction_files(tmp_path: Path):
    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")
    mapped = json.loads(run(repo, "scripts/ai-index", "query", "get_tests", "calo-policy").stdout)
    assert "tests/AGENTS.md" not in mapped
    assert "tests/unit/__init__.py" not in mapped
    assert "tests/unit/helpers.py" not in mapped
    assert "tests/unit/test_policy_registry.py" in mapped
'''
    if "def test_module_test_mapping_excludes_support_and_instruction_files" not in test_text:
        anchor = "\ndef test_deterministic_context_queries_and_public_surface_ranking"
        if anchor not in test_text:
            raise SystemExit("Unable to place test-mapping regression test.")
        test_text = test_text.replace(anchor, regression + anchor, 1)
    tests.write_text(test_text, encoding="utf-8", newline="\n")

    run(root, sys.executable, "scripts/ai-index", "update")
    run(root, sys.executable, "scripts/ai-index", "check")
    mapped_proc = run(root, sys.executable, "scripts/ai-index", "query", "get_tests", "calo-policy")
    try:
        mapped = json.loads(mapped_proc.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"get_tests calo-policy did not return JSON: {exc}")
    if not isinstance(mapped, list):
        raise SystemExit("get_tests calo-policy returned a non-list result.")
    false_positives = [
        p for p in mapped
        if p.lower().endswith(("/agents.md", "/agent.md", "/__init__.py", "/conftest.py"))
        or p.lower().endswith(("agents.md", "agent.md"))
    ]
    if false_positives:
        raise SystemExit("Non-test support/instruction files remain mapped as tests: " + ", ".join(false_positives))

    report = {
        "patch": "v2-conservative-test-file-classification",
        "engine_patched": text != original,
        "source_classification_replacements": source_count,
        "mapped_test_count": len(mapped),
        "false_positive_support_files": false_positives,
    }
    out = root / ".ai-tmp" / "test-mapping-remediation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print("\nTest-mapping remediation completed.")
    print(f"calo-policy mapped test count: {len(mapped)}")
    print("Mapped AGENTS/__init__/conftest support files: none")
    print(f"Report: {out}")
    print("No scientific workload or pytest suite was run by this remediation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
