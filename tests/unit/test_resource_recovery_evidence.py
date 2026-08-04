from __future__ import annotations

import pytest

from calo_rpd_studio.scripts.validate_resource_recovery import (
    _MIB,
    _bounded_pressure_bytes,
    _recovery_within_tolerance,
)


def test_pressure_probe_is_bounded_by_fraction_and_absolute_ceiling():
    assert _bounded_pressure_bytes(8 * 1024**3, 0.05, 256) == 256 * _MIB
    assert _bounded_pressure_bytes(256 * _MIB, 0.10, 128) == int(25.6 * _MIB)


@pytest.mark.parametrize(
    ("free_bytes", "fraction", "maximum_mib", "message"),
    [
        (0, 0.05, 256, "free bytes"),
        (1024**3, 0.0, 256, "pressure fraction"),
        (1024**3, 0.26, 256, "pressure fraction"),
        (1024**3, 0.05, 8, "between 16 and 512"),
        (1024**3, 0.05, 513, "between 16 and 512"),
    ],
)
def test_pressure_probe_rejects_unsafe_or_unusable_requests(
    free_bytes, fraction, maximum_mib, message
):
    with pytest.raises((ValueError, RuntimeError), match=message):
        _bounded_pressure_bytes(free_bytes, fraction, maximum_mib)


def test_recovery_tolerance_is_explicit_and_never_negative():
    before = 4 * 1024**3
    assert _recovery_within_tolerance(before, before - 63 * _MIB, 64)
    assert not _recovery_within_tolerance(before, before - 65 * _MIB, 64)
    with pytest.raises(ValueError, match="cannot be negative"):
        _recovery_within_tolerance(before, before, -1)
