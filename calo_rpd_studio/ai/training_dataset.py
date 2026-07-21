"""Training split manifest helpers that prevent silent benchmark leakage."""

from dataclasses import dataclass, field


@dataclass(slots=True)
class TrainingManifest:
    training_problem_ids: list[str] = field(default_factory=list)
    validation_problem_ids: list[str] = field(default_factory=list)
    final_test_problem_ids: list[str] = field(default_factory=list)

    def validate_disjoint(self):
        a = set(self.training_problem_ids)
        b = set(self.validation_problem_ids)
        c = set(self.final_test_problem_ids)
        if a & b:
            raise ValueError(
                "Training and validation problem sets must be disjoint."
            )
        if a & c or b & c:
            raise ValueError(
                "Final test problems must be disjoint from training and validation sets."
            )
        return True
