from calo_rpd_studio.statistics.descriptive import descriptive_statistics
from calo_rpd_studio.statistics.posthoc import holm_correction
from calo_rpd_studio.statistics.effect_sizes import cliffs_delta


def test_descriptive_statistics():
    s = descriptive_statistics([1, 2, 3, 4, 5])
    assert s["median"] == 3
    assert s["best"] == 1


def test_holm_correction_bounds():
    corrected = holm_correction([0.01, 0.04, 0.2])
    assert all(0 <= x <= 1 for x in corrected)


def test_cliffs_delta_extreme():
    assert cliffs_delta([3, 4], [1, 2]) == 1.0
