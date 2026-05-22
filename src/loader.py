from pathlib import Path

import pandas as pd


class CsvDataloader:
    def __init__(self, file_path: str | Path):
        self.file_path = file_path
        self.dataframe = pd.read_csv(file_path)

    def load_xy(self, target_column, feature_columns: list[str] | None = None):
        X = self.dataframe.drop(columns=[target_column])
        if feature_columns:
            X = X[feature_columns]
        y = self.dataframe[target_column]
        return X, y
