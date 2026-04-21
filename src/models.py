from abc import ABC, abstractmethod

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier


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

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        prediction = self.model.predict(X)

    def dump(self, file_path):
        joblib.dump(self.model, file_path)


class DecisionTreeModel(Model):
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X, y):
        return self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def dump(self, file_path):
        joblib.dump(self.model, file_path)
