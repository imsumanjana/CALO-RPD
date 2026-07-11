"""Experiment configuration save/load service."""
from calo_rpd_studio.experiments.experiment_config import ExperimentConfig
class ProjectManager:
    @staticmethod
    def save(config,path):return config.save(path)
    @staticmethod
    def load(path):return ExperimentConfig.load(path)
