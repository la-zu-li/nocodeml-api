from fastapi import FastAPI

from src.db import SessionDep, create_db_and_tables
from src.loader import CsvDataloader
from src.models import DecisionTreeModel, LinearRegressionModel

from .schema import ModelType, TrainRequest

app = FastAPI()


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
async def root():
    return {"message": "Server up and running!"}


@app.post("/train/")
async def train(body: TrainRequest, session: SessionDep):
    target_name = body.target_name
    csv_file_path = body.dataset_file_path

    if body.model_type is ModelType.linear_regression:
        model = LinearRegressionModel(target_name)
    else:
        model = DecisionTreeModel(target_name)

    dataloader = CsvDataloader(csv_file_path)
    X, y = dataloader.load_xy(target_name)
    model.train(X, y)

    return model.save(session)
