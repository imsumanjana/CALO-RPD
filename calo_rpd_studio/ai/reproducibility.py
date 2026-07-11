"""AI reproducibility metadata validation."""
def validate_policy_metadata(metadata):
    required=('software_version','training_seed','training_configuration','training_problem_identifiers','final_test_systems_used_for_training');missing=[x for x in required if x not in metadata];return {'passed':not missing,'missing':missing}
