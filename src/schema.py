from pathlib import Path

from pydantic import BaseModel

from .types import ModelType, Task


class ModelConfig(BaseModel):
    task: Task
    model_type: ModelType
    file_path: Path
    feature_names: list[str]
    target_name: str


class TrainRequest(BaseModel):
    model_type: ModelType
    dataset_file_path: Path
    target_name: str
    feature_names: list[str] | None
    test_size: float = 0.2


class PredictRequest(BaseModel):
    model_id: int
    instances_file_path: Path
