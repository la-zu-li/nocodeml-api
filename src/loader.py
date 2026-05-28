from contextlib import contextmanager
from pathlib import Path

import pandas as pd
from fastapi import HTTPException
from sklearn.model_selection import train_test_split


class CsvDataloader:
    def __init__(self, file_path: str | Path):
        self.file_path = file_path
        self.dataframe = pd.read_csv(file_path)

    def load_x(self, target_column, feature_columns: list[str] | None = None):
        try:
            X = self.dataframe.drop(columns=[target_column])
        except KeyError:
            X = self.dataframe
        if feature_columns:
            try:
                X = X[feature_columns]
            except KeyError:
                raise UnexistentFeatureError(
                    f"The combination of feature columns {feature_columns} does not exist in the CSV data"
                )
        return X

    def load_xy(self, target_column, feature_columns: list[str] | None = None):
        X = self.load_x(target_column, feature_columns)
        try:
            y = self.dataframe[target_column]
        except KeyError:
            raise UnexistentTargetError(
                f"Target column '{target_column}' does not exist in the CSV data"
            )
        return X, y

    def train_test_split(
        self,
        target_column,
        feature_columns: list[str] | None = None,
        test_size=0.2,
    ):
        X, y = self.load_xy(target_column, feature_columns)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        return X_train, X_test, y_train, y_test


class UnexistentTargetError(KeyError):
    pass


class UnexistentFeatureError(KeyError):
    pass


@contextmanager
def handle_csv_loading():
    try:
        yield
    except (FileNotFoundError, pd.errors.EmptyDataError):
        raise HTTPException(
            status_code=400, detail=f"CSV data is not available or empty"
        )
    except pd.errors.ParserError:
        raise HTTPException(
            status_code=400, detail=f"File does not contain valid CSV data"
        )
    except (UnexistentTargetError, UnexistentFeatureError) as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
