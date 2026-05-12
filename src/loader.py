from pathlib import Path

import pandas as pd


class CsvDataloader:
    def __init__(self, file_path: str | Path):
        self.file_path = file_path
        self.dataframe = pd.read_csv(file_path)

    def load_xy(self, target_column):
        X = self.dataframe.drop(columns=[target_column])
        y = self.dataframe[target_column]
        return X, y
