from calo_rpd_studio.statistics.descriptive import descriptive_statistics
from calo_rpd_studio.statistics.posthoc import holm_correction
from calo_rpd_studio.statistics.effect_sizes import cliffs_delta

def test_statistics():
    s=descriptive_statistics([1,2,3,4,5]);assert s['median']==3;assert s['best']==1
    corrected=holm_correction([.01,.04,.2]);assert all(0<=x<=1 for x in corrected)
    assert cliffs_delta([3,4],[1,2])==1.0
