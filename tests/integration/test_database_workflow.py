from calo_rpd_studio.results.database import ResultDatabase
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
from calo_rpd_studio.experiments.provenance import collect_provenance

def test_database_experiment_creation(tmp_path):
    db=ResultDatabase(tmp_path/'r.sqlite');eid=db.create_experiment(ExperimentConfig(),collect_provenance());assert db.get_experiment(eid)['id']==eid
