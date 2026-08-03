"""Verify that a pip requirements lock is exact, hash-complete, and source-bounded."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re


_BLOCK = re.compile(
    r"(?ms)^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s\\]+).*?"
    r"(?=^[A-Za-z0-9][A-Za-z0-9_.-]*==|\Z)"
)
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})(?=\s|\\|$)")
_FORBIDDEN = ("--editable", "-e ", "git+", "file:", "${", "@ http://", "@ https://")


def _canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@dataclass(frozen=True)
class LockVerification:
    path: str
    package_count: int
    hash_count: int
    pins: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "schema": "calo_rpd_requirements_lock_verification_v1",
            "path": self.path,
            "package_count": self.package_count,
            "hash_count": self.hash_count,
            "pins": list(self.pins),
        }


def verify_lock(
    path: Path,
    *,
    expected_index: str | None = None,
    expected_pins: tuple[str, ...] = (),
) -> LockVerification:
    source = path.resolve(strict=True)
    content = source.read_text(encoding="utf-8")
    lowered = content.lower()
    for token in _FORBIDDEN:
        if token in lowered:
            raise ValueError(f"Forbidden mutable requirement source {token!r} in {source}")
    if expected_index and f"--extra-index-url {expected_index}" not in content:
        raise ValueError(f"Expected package index is absent from {source}: {expected_index}")

    packages: dict[str, str] = {}
    hash_count = 0
    for match in _BLOCK.finditer(content):
        name, version = match.group(1), match.group(2)
        canonical = _canonical(name)
        if canonical in packages:
            raise ValueError(f"Duplicate locked package {name!r} in {source}")
        block_hashes = _HASH.findall(match.group(0))
        if not block_hashes:
            raise ValueError(f"Locked package has no SHA-256 hash: {name}=={version}")
        packages[canonical] = version
        hash_count += len(block_hashes)
    if not packages:
        raise ValueError(f"No exact package pins found in {source}")

    for pin in expected_pins:
        expected = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s]+)", pin)
        if expected is None:
            raise ValueError(f"Invalid expected pin: {pin!r}")
        actual = packages.get(_canonical(expected.group(1)))
        if actual != expected.group(2):
            raise ValueError(f"Expected {pin!r} in {source}; found {actual!r} for that package")
    pins = tuple(f"{name}=={version}" for name, version in sorted(packages.items()))
    return LockVerification(
        path=str(source),
        package_count=len(packages),
        hash_count=hash_count,
        pins=pins,
    )


def verify_lock_contains_exact_graph(reference: Path, candidate: Path) -> None:
    """Require every package in *reference* at the same version in *candidate*."""

    reference_result = verify_lock(reference)
    candidate_result = verify_lock(candidate)
    reference_pins = dict(pin.split("==", 1) for pin in reference_result.pins)
    candidate_pins = dict(pin.split("==", 1) for pin in candidate_result.pins)
    mismatches = {
        name: {"expected": version, "actual": candidate_pins.get(name)}
        for name, version in reference_pins.items()
        if candidate_pins.get(name) != version
    }
    if mismatches:
        raise ValueError(
            "Candidate lock does not preserve the reference runtime graph: "
            + json.dumps(mismatches, sort_keys=True)
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lock", type=Path)
    parser.add_argument("--expect-index")
    parser.add_argument("--expect-pin", action="append", default=[])
    parser.add_argument(
        "--match-lock",
        type=Path,
        help="Require every package/version in this reference lock to match the candidate lock.",
    )
    arguments = parser.parse_args()
    result = verify_lock(
        arguments.lock,
        expected_index=arguments.expect_index,
        expected_pins=tuple(arguments.expect_pin),
    )
    if arguments.match_lock is not None:
        verify_lock_contains_exact_graph(arguments.match_lock, arguments.lock)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
