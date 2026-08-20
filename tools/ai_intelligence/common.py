from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any

SCHEMA_VERSION = 1
AI_DIR = ".ai"
INDEX_FILES = {
    "file": "file-index.json",
    "symbol": "symbol-index.json",
    "dependency": "dependency-graph.json",
    "test": "test-map.json",
    "audit": "audit-coverage.json",
    "findings": "findings.json",
    "status": "INDEX_STATUS.md",
    "changes": "change-index.json",
}
SOURCE_EXTENSIONS = {
    ".py": "python", ".pyi": "python", ".toml": "toml", ".yaml": "yaml",
    ".yml": "yaml", ".json": "json", ".sh": "shell", ".ps1": "powershell",
    ".bat": "batch", ".md": "markdown",
}
SPECIAL_FILES = {
    "Dockerfile": "dockerfile", "compose.yaml": "yaml", "pyproject.toml": "toml",
    "MANIFEST.in": "manifest",
}
IGNORE_PREFIXES = (
    ".ai/", ".git/", ".venv/", "venv/", "build/", "dist/", "validation/",
    "publication_export/", "calo_rpd_studio/data/pglib/",
    "calo_rpd_studio/data/trained_models/", "calo_rpd_studio/data/frozen/",
)
IGNORE_NAMES = {"MANIFEST.sha256"}
HISTORICAL_PATTERNS = (
    re.compile(r"^CALO-RPD-v.*(?:AUDIT|REPORT|SUMMARY).*$", re.I),
    re.compile(r"^FINDINGS_CLOSURE_.*\.csv$", re.I),
)
MODULE_RULES = (
    ("bootstrap", ("bootstrap.py", "calo_bootstrap/", "Launch-CALO-RPD.ps1", "run_", "start_calo")),
    ("desktop", ("calo_rpd_studio/app/", "calo_rpd_studio/gui/")),
    ("calo-policy", ("calo_rpd_studio/algorithms/calo/", "calo_rpd_studio/ai/", "calo_rpd_studio/learning/")),
    ("optimization", ("calo_rpd_studio/algorithms/", "calo_rpd_studio/portfolio/")),
    ("power-system", ("calo_rpd_studio/power_system/", "calo_rpd_studio/orpd/", "calo_rpd_studio/robustness/")),
    ("compute", ("calo_rpd_studio/compute/", "calo_rpd_studio/accelerated/")),
    ("experiments", ("calo_rpd_studio/experiments/", "calo_rpd_studio/benchmarking/")),
    ("persistence", ("calo_rpd_studio/results/", "calo_rpd_studio/resume/", "calo_rpd_studio/continuation/")),
    ("validation-release", ("calo_rpd_studio/validation/", "calo_rpd_studio/scripts/", ".github/", "containers/", "Dockerfile", "compose.yaml")),
    ("tests", ("tests/",)),
    ("documentation", ("docs/", "README.md", "AGENTS.md")),
)
MODULE_KNOWLEDGE = {
    "bootstrap": (
        "Native setup, dependency repair and application launch without coupling ordinary launch to installation.",
        "Environment/prerequisite and launch state; not scientific study state.",
        "bootstrap/launcher -> prerequisite checks -> calo_rpd_studio.app.application.",
        "Ordinary launch must not install packages or perform policy lifecycle work.",
        "Missing/incompatible prerequisites, platform launch paths and setup/launch boundary regressions.",
    ),
    "desktop": (
        "Scientist-facing PyQt shell, shared application state, task/workspace management and workflow navigation.",
        "AppState/workspace/task state is authoritative; panels should not keep divergent scientific copies.",
        "application -> MainWindow -> panels/workflow managers -> services -> results -> UI refresh.",
        "Scientist wording, zero-policy safety, accessibility and atomic shared state updates.",
        "Stale UI state, duplicated ownership, task lifecycle races and backend state leaking into scientific choices.",
    ),
    "calo-policy": (
        "CALO/TSH-CALO policy artifacts, training, generalization evidence, qualification, activation and inference contracts.",
        "Immutable artifact provenance, registry records, receipts and explicit active bindings.",
        "independent training -> candidate -> qualification -> explicit activation -> checksum-bound experiment use.",
        "No auto-train/qualify/activate; protected cases isolated; A-E production, F experimental/off; exact accounting.",
        "Provenance/schema mismatch, lifecycle bypass, accounting gaps and stale policy bindings.",
    ),
    "optimization": (
        "Common optimizer contracts, implementations, registry and portfolio execution surfaces.",
        "Optimizer population/archive/RNG state plus experiment-supplied configuration.",
        "experiment plan -> registry -> optimizer -> ORPD objective/constraints -> result.",
        "Deterministic seeded behavior, common feasibility semantics and exact evaluation budgets.",
        "Budget drift, ranking/feasibility inconsistency, RNG nondeterminism and bound errors.",
    ),
    "power-system": (
        "Power-system models, AC power flow, ORPD decoding, objectives, constraints and robustness semantics.",
        "Case/formulation/scenario and solver state; protected-case identity is a scientific boundary.",
        "case/config -> formulation -> candidate decode -> power flow -> objectives/constraints -> result.",
        "Reference semantics, protected-case isolation, formulation fingerprints and tolerance consistency.",
        "Convergence, units/tolerances, case identity, decoding and constraint inconsistencies.",
    ),
    "compute": (
        "Resource admission, topology/device binding, execution contracts, persistent workers and accelerator kernels.",
        "Leases, memory budgets, worker/device identity, telemetry and execution provenance.",
        "request -> Safe-80 admission -> CUDA or CPU lane -> worker/kernel -> attested result.",
        "CUDA-preferred/CPU-only; Intel XPU non-executable; at most 80% of free VRAM/available RAM.",
        "OOM/resource races, identity mismatch, silent fallback, stale workers and CPU/CUDA parity drift.",
    ),
    "experiments": (
        "Experiment configuration, plans, budgets, deterministic seeds, fairness and runner orchestration.",
        "Validated configuration, execution plan, seed allocation, progress and provenance.",
        "validated config -> plan -> scheduler/optimizer/evaluator -> result contracts/provenance.",
        "Fair equal budgets, exact accounting, deterministic seeds and immutable policy binding.",
        "Configuration drift, unfair budgets, resume mismatch and provenance mismatch.",
    ),
    "persistence": (
        "SQLite durable results/provenance plus resume, continuation, integrity and publication retrieval.",
        "Database rows, result artifacts, checkpoints/resume envelopes and continuation contracts.",
        "execution -> durable result -> query/export; interruption -> resume validation -> restart.",
        "Atomic integrity-checked persistence, exact provenance and schema-compatible resume/extension.",
        "Migration drift, partial writes, stale paths, corruption and incorrect cumulative reconstruction.",
    ),
    "validation-release": (
        "Developer/release validators, CI, packaging, containers and evidence-generation tooling.",
        "Evidence bound to exact source/artifact identities.",
        "source -> non-mutating checks/build stages -> evidence/manifests -> separately authorized gates.",
        "Pinned actions/locks, no fabricated attestations, historical evidence immutable and scoped.",
        "Stale artifacts, unpinned dependencies/actions, identity mismatch and accidental claim elevation.",
    ),
    "core": (
        "Cross-cutting CALO-RPD package utilities not owned by a narrower subsystem.",
        "Varies by file; use exact symbol/dependency retrieval before assuming ownership.",
        "Cross-cutting support for narrower application/scientific subsystems.",
        "Follow the nearest AGENTS.md; avoid new global state without explicit ownership.",
        "Hidden coupling and unclear ownership; inspect dependents before shared-helper changes.",
    ),
}
AUDIT_CATEGORIES = [
    "logic", "correctness", "security", "error_handling", "concurrency", "performance",
    "maintainability", "test_coverage", "ui_logic_integration", "architecture",
]
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")

class ToolError(RuntimeError):
    pass

def run_git(root: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    if check and proc.returncode:
        raise ToolError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()

def repo_root(start: Path | None = None) -> Path:
    start = (start or Path.cwd()).resolve()
    return Path(run_git(start, "rev-parse", "--show-toplevel")).resolve()

def git_state(root: Path) -> dict[str, Any]:
    commit = run_git(root, "rev-parse", "HEAD")
    return {
        "commit": commit,
        "branch": run_git(root, "branch", "--show-current", check=False) or "DETACHED",
        "indexed_at": run_git(root, "show", "-s", "--format=%cI", commit, check=False),
    }

def tracked_files(root: Path) -> list[str]:
    proc = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode:
        raise ToolError(proc.stderr.decode("utf-8", "replace").strip())
    return sorted(x.decode("utf-8", "replace") for x in proc.stdout.split(b"\0") if x)

def should_index(path: str) -> bool:
    path = path.replace("\\", "/")
    if path in IGNORE_NAMES or any(path.startswith(prefix) for prefix in IGNORE_PREFIXES):
        return False
    if any(pattern.match(PurePosixPath(path).name) for pattern in HISTORICAL_PATTERNS):
        return False
    name = PurePosixPath(path).name
    return name in SPECIAL_FILES or PurePosixPath(path).suffix.lower() in SOURCE_EXTENSIONS

def language_for(path: str) -> str:
    name = PurePosixPath(path).name
    return SPECIAL_FILES.get(name, SOURCE_EXTENSIONS.get(PurePosixPath(path).suffix.lower(), "text"))

def module_for(path: str) -> str:
    normalized = path.replace("\\", "/")
    for module, prefixes in MODULE_RULES:
        for prefix in prefixes:
            if prefix.endswith("/") and normalized.startswith(prefix):
                return module
            if prefix.endswith("_") and PurePosixPath(normalized).name.startswith(prefix):
                return module
            if normalized == prefix or normalized.startswith(prefix):
                return module
    return "core" if normalized.startswith("calo_rpd_studio/") else "repository"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default

def canonical_json(data: Any) -> str:
    return json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"

def write_if_changed(path: Path, content: str) -> bool:
    try:
        if path.read_text(encoding="utf-8") == content:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return True

def source_fingerprint(file_index: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path, meta in sorted(file_index["files"].items()):
        digest.update(path.encode("utf-8") + b"\0" + meta["sha256"].encode("ascii") + b"\n")
    return digest.hexdigest()
