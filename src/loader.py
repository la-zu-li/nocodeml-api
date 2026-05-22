from pathlib import Path

import pandas as pd


class CsvDataloader:
    def __init__(self, file_path: str | Path):
        self.file_path = file_path
        self.dataframe = pd.read_csv(file_path)

    def load_xy(self, target_column, feature_columns: list[str] | None = None):
        try:
            X = self.dataframe.drop(columns=[target_column])
        except KeyError:
            raise UnexistentTargetError(
                f"Target column '{target_column}' does not exist in the CSV data"
            )
        if feature_columns:
            try:
                X = X[feature_columns]
            except KeyError:
                raise UnexistentFeatureError(
                    f"The combination of feature columns {feature_columns} does not exist in the CSV data"
                )
        y = self.dataframe[target_column]
        return X, y


class UnexistentTargetError(KeyError):
    pass


class UnexistentFeatureError(KeyError):
    pass
