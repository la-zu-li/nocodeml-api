import pickle as pkl
from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import uuid4

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from src.db import MlModel, SessionDep

from .env import CONFIG_PATH
from .types import ModelType, Task


class Model(ABC):

    def __init__(self, model, target_name: str, feature_names: list[str] | None = None):
        self.model = model
        self.target_name = target_name
        self.feature_names = feature_names

    @abstractmethod
    def train(self, X, y): ...

    @abstractmethod
    def predict(self, X) -> Sequence[int | float]: ...

    def dump(self):
        model_bytes = pkl.dumps(self.model, pkl.HIGHEST_PROTOCOL)
        return model_bytes

    @abstractmethod
    def create_db_model(self) -> MlModel: ...

    def save(self, session: SessionDep):
        db_model = self.create_db_model()

        session.add(db_model)
        session.commit()
        session.refresh(db_model)

        return db_model


class LinearRegressionModel(Model):
    def __init__(self, target_name: str, feature_names: list[str] | None = None):
        self.id = uuid4()
        super().__init__(LinearRegression(), target_name, feature_names)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        assert isinstance(prediction, np.ndarray)
        return prediction.tolist()

    def create_db_model(self) -> MlModel:
        model_bytes = self.dump()

        return MlModel(
            task=Task.regression,
            model_type=ModelType.linear_regression,
            is_trained=True,
            feature_names=self.feature_names,
            target_name=self.target_name,
            raw_model=model_bytes,
        )

    def get_filepath(self):
        folder_path = CONFIG_PATH / "models"
        folder_path.mkdir(exist_ok=True, parents=True)

        filename = f"linear_regression_{self.id}.pkl"
        file_path = folder_path / filename
        return file_path


class DecisionTreeModel(Model):

    def __init__(self, target_name: str, feature_names: list[str] | None = None):
        self.id = uuid4()
        super().__init__(DecisionTreeClassifier(), target_name, feature_names)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        assert isinstance(prediction, np.ndarray)
        return prediction.tolist()

    def create_db_model(self) -> MlModel:
        model_bytes = self.dump()

        return MlModel(
            task=Task.classification,
            model_type=ModelType.decision_tree,
            is_trained=True,
            feature_names=self.feature_names,
            target_name=self.target_name,
            raw_model=model_bytes,
        )

    def get_filepath(self):
        folder_path = CONFIG_PATH / "models"
        folder_path.mkdir(exist_ok=True, parents=True)

        filename = f"linear_regression_{self.id}.pkl"
        file_path = folder_path / filename
        return file_path


def create_model_from_db(db_model: MlModel):
    feature_names = db_model.feature_names
    target_name = db_model.target_name
    raw_model = pkl.loads(db_model.raw_model)
    model_type = db_model.model_type

    if model_type is ModelType.linear_regression:
        model = LinearRegressionModel(target_name, feature_names)
    else:
        model = DecisionTreeModel(target_name, feature_names)

    model.model = raw_model
    return model
