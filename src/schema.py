from enum import Enum
from pathlib import Path

from pydantic import BaseModel


class ModelType(str, Enum):
    linear_regression = "linear-regression"
    decision_tree = "decision-tree"


class Task(str, Enum):
    regression = "regression"
    classification = "classification"


class ModelConfig(BaseModel):
    task: Task
    model_type: ModelType
    file_path: Path
    feature_names: list[str]
    target_name: str


class TrainRequest(BaseModel):
    model_type: ModelType
    file_path: Path
