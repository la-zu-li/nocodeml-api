from collections.abc import Sequence

from fastapi import FastAPI, HTTPException
from sqlmodel import select

from src.db import MlModel, SessionDep, create_db_and_tables
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


@app.get("/models")
async def get_models(session: SessionDep) -> Sequence[MlModel]:
    models = session.exec(select(MlModel)).all()
    return models


@app.get("/models/{model_id}")
async def get_model(model_id: int, session: SessionDep) -> MlModel:
    model = session.get(MlModel, model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.post("/train/")
async def train(body: TrainRequest, session: SessionDep) -> MlModel:
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
