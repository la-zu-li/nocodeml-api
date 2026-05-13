from enum import Enum


class ModelType(str, Enum):
    LINEAR_REGRESSION = "linear-regression"
    DECISION_TREE = "decision-tree"


class Task(str, Enum):
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
