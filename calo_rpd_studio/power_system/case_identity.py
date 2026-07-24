"""Canonical identity checks for protected benchmark systems.

Filename-only holdout protection is insufficient because a benchmark can be copied to another name.
This module compares loaded scientific case checksums against canonical bundled references whenever
PYPOWER is available.  If the canonical reference cannot be loaded, matching 118/300-bus custom
cases fail closed rather than silently allowing a possible renamed holdout into policy training.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .case_loader import CaseLoader

PROTECTED_HOLDOUT_BUS_COUNTS = {"case118": 118, "case300": 300}


@lru_cache(maxsize=1)
def canonical_protected_holdout_checksums() -> dict[str, str]:
    checksums: dict[str, str] = {}
    for name in PROTECTED_HOLDOUT_BUS_COUNTS:
        try:
            checksums[name] = CaseLoader.load(name).checksum().lower()
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, UnicodeError):
            # The runtime may be dependency-light. Callers fail closed by bus count for the
            # corresponding unresolved canonical benchmark instead of trusting a filename.
            continue
    return checksums


def protected_holdout_identity(source: str | Path) -> str:
    """Return the protected canonical case name, or an empty string when not protected."""
    text = str(source)
    stem = Path(text).stem.lower()
    if stem in PROTECTED_HOLDOUT_BUS_COUNTS:
        return stem
    try:
        case = CaseLoader.load(source)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, AttributeError, UnicodeError):
        return ""
    checksum = case.checksum().lower()
    references = canonical_protected_holdout_checksums()
    for name, reference in references.items():
        if checksum == reference:
            return name
    # Conservative fallback when PYPOWER/canonical reference loading is unavailable. This closes
    # the renamed-copy bypass in dependency-light environments at the cost of requiring an explicit
    # override for unrelated custom systems with exactly 118 or 300 buses.
    for name, bus_count in PROTECTED_HOLDOUT_BUS_COUNTS.items():
        if name not in references and int(case.n_bus) == int(bus_count):
            return name
    return ""


def protected_holdout_matches(sources) -> tuple[str, ...]:
    return tuple(sorted({identity for item in sources if (identity := protected_holdout_identity(item))}))
