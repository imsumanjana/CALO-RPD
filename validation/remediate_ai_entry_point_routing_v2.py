#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

QUESTION = "Where does native application launch begin?"
EXPECTED = "calo_bootstrap/launcher.py"


def run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    print("+", " ".join(args))
    proc = subprocess.run(
        args,
        cwd=root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    if proc.returncode:
        raise SystemExit(proc.returncode)
    return proc


def patch_entry_ranking(text: str) -> tuple[str, int]:
    pattern = re.compile(
        r'if s\.get\("entry_point"\) and any\(x in qlower for x in '
        r'\("start", "entry", "bootstrap", "run", "main"\)\): score \+= [45]'
    )
    replacement = '''if s.get("entry_point") and (
            any(x in qlower for x in (
                "start", "entry", "bootstrap", "run", "main", "launch", "begin",
                "startup", "initialize", "initialise",
            ))
            or bool(qtokens & {
                "start", "entry", "bootstrap", "run", "main", "launch", "begin", "startup",
            })
        ): score += 9'''
    updated, count = pattern.subn(replacement, text)
    return updated, count


def patch_semantic_concepts(root: Path) -> bool:
    path = root / ".ai" / "semantic-concepts.json"
    if not path.is_file():
        raise SystemExit("Missing .ai/semantic-concepts.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    concepts = doc.setdefault("concepts", {})
    terms = list(concepts.get("entry_point", []))
    wanted = [
        "entry point",
        "startup",
        "bootstrap",
        "main",
        "start",
        "launch",
        "begin",
        "native application launch",
    ]
    merged: list[str] = []
    for term in [*terms, *wanted]:
        if term not in merged:
            merged.append(term)
    changed = merged != terms
    concepts["entry_point"] = merged
    if changed:
        path.write_text(
            json.dumps(doc, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    return changed


def patch_regression_test(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    changed = False

    # Do not contaminate the shared synthetic repository with a permanent bootstrap
    # file. The stale-module cleanup regression intentionally deletes the last
    # bootstrap fixture and must remain able to prove that its module shard disappears.
    global_fixture = '        "calo_bootstrap/launcher.py": "def main():\\n    return 0\\n",\n'
    if global_fixture in text:
        text = text.replace(global_fixture, "", 1)
        changed = True

    test_name = "test_entry_point_routing_understands_natural_launch_language"
    if f"def {test_name}" not in text:
        regression = f'''\n\ndef {test_name}(tmp_path: Path):\n    repo = make_repo(tmp_path)\n    write(repo, "{EXPECTED}", "def main():\\n    return 0\\n")\n    run(repo, "scripts/ai-index", "init")\n    ctx = run(\n        repo,\n        "scripts/ai-index",\n        "context",\n        "{QUESTION}",\n        "--no-semantic",\n    ).stdout\n    assert "{EXPECTED}" in ctx\n    assert "entry_point" in ctx\n'''
        anchor = "\ndef test_semantic_cache_reuse_config_invalidation_and_corrupt_fallback"
        if anchor not in text:
            raise SystemExit("Unable to place entry-point routing regression test.")
        text = text.replace(anchor, regression + anchor, 1)
        changed = True
    else:
        # Self-heal the first remediation's one-line setup by localizing the launcher
        # fixture to this test only.
        start = text.index(f"def {test_name}")
        next_def = text.find("\ndef ", start + 4)
        if next_def < 0:
            next_def = len(text)
        block = text[start:next_def]
        if f'write(repo, "{EXPECTED}"' not in block:
            old_setup = '    repo = make_repo(tmp_path); run(repo, "scripts/ai-index", "init")\n'
            new_setup = (
                '    repo = make_repo(tmp_path)\n'
                f'    write(repo, "{EXPECTED}", "def main():\\n    return 0\\n")\n'
                '    run(repo, "scripts/ai-index", "init")\n'
            )
            if old_setup not in block:
                raise SystemExit("Unable to localize the existing entry-point regression fixture.")
            block = block.replace(old_setup, new_setup, 1)
            text = text[:start] + block + text[next_def:]
            changed = True

    if changed:
        path.write_text(text, encoding="utf-8", newline="\n")
    return changed


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    engine = root / "scripts" / "ai-index"
    tests = root / "tests" / "tooling" / "test_ai_repo_intelligence_v2.py"
    if not engine.is_file() or not tests.is_file():
        raise SystemExit("Installed v2 engine/tests not found.")

    original = engine.read_text(encoding="utf-8")
    if "CALO_PROFILE_VERSION" not in original:
        raise SystemExit("scripts/ai-index is not the expected CALO v2 engine; refusing to patch.")

    updated, replacements = patch_entry_ranking(original)
    if replacements == 0:
        # Idempotent success is allowed only if the broadened terms are already present.
        if '"launch", "begin"' not in original or 'score += 9' not in original:
            raise SystemExit("Unable to locate the entry-point ranking rule; refusing a partial patch.")
        updated = original

    try:
        compile(updated, str(engine), "exec")
    except SyntaxError as exc:
        raise SystemExit(
            f"Refusing to write invalid scripts/ai-index candidate: line {exc.lineno}: {exc.msg}"
        ) from exc
    if updated != original:
        engine.write_text(updated, encoding="utf-8", newline="\n")

    concepts_changed = patch_semantic_concepts(root)
    tests_changed = patch_regression_test(tests)

    # Rebuild deterministic shards, then normalize recent-change state so migration/tooling
    # maintenance itself is not presented to a fresh agent as an ordinary product edit.
    run(root, sys.executable, "scripts/ai-index", "update")
    run(root, sys.executable, "scripts/ai-index", "check")
    run(root, sys.executable, "scripts/ai-index", "init")
    run(root, sys.executable, "scripts/ai-index", "check")

    ctx = run(
        root,
        sys.executable,
        "scripts/ai-index",
        "context",
        QUESTION,
        "--no-semantic",
    ).stdout
    if EXPECTED not in ctx:
        out = root / ".ai-tmp" / "entry-point-routing-failure.txt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(ctx, encoding="utf-8", newline="\n")
        raise SystemExit(
            f"Deterministic entry-point routing is still incorrect; {EXPECTED} is absent. Context: {out}"
        )
    if "entry_point" not in ctx:
        raise SystemExit("Launcher is present but its entry_point architectural marker is missing from context.")

    change = json.loads((root / ".ai/index/change-index.json").read_text(encoding="utf-8"))
    if change.get("initialization") is True or change.get("newly_indexed_files"):
        raise SystemExit("Entry-point remediation leaked migration initialization/new files into recent changes.")

    report = {
        "patch": "v2-natural-language-entry-point-routing",
        "question": QUESTION,
        "expected_file": EXPECTED,
        "engine_ranking_rules_replaced": replacements,
        "semantic_concepts_changed": concepts_changed,
        "regression_test_changed": tests_changed,
        "deterministic_context_bytes": len(ctx.encode("utf-8")),
        "deterministic_expected_file_present": True,
        "entry_point_marker_present": True,
        "recent_changed_files": len(change.get("changed_files", [])),
        "shared_fixture_contamination_removed": True,
    }
    out = root / ".ai-tmp" / "entry-point-routing-remediation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    print("\nEntry-point routing remediation completed.")
    print(f"Deterministic query routed to: {EXPECTED}")
    print("Natural entry-point terms: start/entry/bootstrap/main/launch/begin/startup")
    print("Entry-point regression fixture is isolated from shared synthetic repositories.")
    print("Generated engine compile check: passed")
    print(f"Recent changed files after normalization: {len(change.get('changed_files', []))}")
    print(f"Report: {out}")
    print("No scientific workload or pytest suite was run by this remediation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
