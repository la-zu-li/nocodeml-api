from enum import Enum


class ModelType(str, Enum):
    linear_regression = "linear-regression"
    decision_tree = "decision-tree"


class Task(str, Enum):
    regression = "regression"
    classification = "classification"
