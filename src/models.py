from abc import ABC, abstractmethod
from uuid import uuid4

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier

from .env import CONFIG_PATH


class Model(ABC):
    @abstractmethod
    def train(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def save(self) -> dict: ...


class LinearRegressionModel(Model):
    def __init__(self):
        self.model = LinearRegression()
        self.id = uuid4()

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)

    def dump(self):
        folder_path = CONFIG_PATH / "models"
        folder_path.mkdir(exist_ok=True, parents=True)

        filename = f"linear_regression_{self.id}.pkl"
        file_path = folder_path / filename
        joblib.dump(self.model, file_path)
        return file_path

    def save(self):
        file_path = self.dump()
        return {"id": self.id, "model_path": file_path}


class DecisionTreeModel(Model):
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def dump(self, file_path):
        joblib.dump(self.model, file_path)
