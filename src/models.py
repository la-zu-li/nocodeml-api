import pickle as pkl
from abc import ABC, abstractmethod
from collections.abc import Sequence
from uuid import uuid4

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from src.db import StoredModel, SessionDep

from .types import ModelType, Task


class Model(ABC):

    def __init__(self, model, target_name: str, feature_names: list[str] | None = None):
        self.model = model
        self.target_name = target_name
        self.feature_names = feature_names
        self.validation_predictions: list[int | float] = []
        self.validation_ground_truth: list[int | float] = []
        self.validation_score: float = 0.0

    @abstractmethod
    def train(self, X, y): ...

    @abstractmethod
    def predict(self, X) -> list[int | float]: ...

    def dump(self):
        model_bytes = pkl.dumps(self.model, pkl.HIGHEST_PROTOCOL)
        return model_bytes

    @abstractmethod
    def create_db_model(self) -> StoredModel: ...

    def save(self, session: SessionDep):
        db_model = self.create_db_model()

        session.add(db_model)
        session.commit()
        session.refresh(db_model)

        return db_model

    def evaluate(self, X, y) -> float:
        return self.model.score(X, y)


class LinearRegressionModel(Model):
    def __init__(self, target_name: str, feature_names: list[str] | None = None):
        super().__init__(LinearRegression(), target_name, feature_names)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        assert isinstance(prediction, np.ndarray)
        return prediction.tolist()

    def create_db_model(self) -> StoredModel:
        model_bytes = self.dump()

        return StoredModel(
            task=Task.REGRESSION,
            model_type=ModelType.LINEAR_REGRESSION,
            feature_names=self.feature_names,
            target_name=self.target_name,
            raw_model=model_bytes,
            validation_predictions=self.validation_predictions,
            validation_ground_truth=self.validation_ground_truth,
            validation_score=self.validation_score,
        )


class DecisionTreeModel(Model):

    def __init__(self, target_name: str, feature_names: list[str] | None = None):
        super().__init__(DecisionTreeClassifier(), target_name, feature_names)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)
        assert isinstance(prediction, np.ndarray)
        return prediction.tolist()

    def create_db_model(self) -> StoredModel:
        model_bytes = self.dump()

        return StoredModel(
            task=Task.CLASSIFICATION,
            model_type=ModelType.DECISION_TREE,
            feature_names=self.feature_names,
            target_name=self.target_name,
            raw_model=model_bytes,
            validation_predictions=self.validation_predictions,
            validation_ground_truth=self.validation_ground_truth,
            validation_score=self.validation_score,
        )


def create_model_from_db(db_model: StoredModel) -> Model:
    feature_names = db_model.feature_names
    target_name = db_model.target_name
    raw_model = pkl.loads(db_model.raw_model)
    model_type = db_model.model_type

    if model_type is ModelType.LINEAR_REGRESSION:
        model = LinearRegressionModel(target_name, feature_names)
    else:
        model = DecisionTreeModel(target_name, feature_names)

    model.model = raw_model
    return model
