from collections.abc import Sequence

from fastapi import FastAPI, HTTPException
from sqlmodel import select

from src.db import MlModel, SessionDep, create_db_and_tables
from src.loader import CsvDataloader
from src.models import DecisionTreeModel, LinearRegressionModel, create_model_from_db

from .schema import ModelType, PredictRequest, TrainRequest

app = FastAPI(debug=True)

import logging


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
async def root():
    return {"message": "Server up and running!"}


@app.get("/models", response_model_exclude={"raw_model"})
async def get_models(session: SessionDep) -> Sequence[MlModel]:
    models = session.exec(select(MlModel)).all()
    return models


@app.get("/models/{model_id}", response_model_exclude={"raw_model"})
async def get_model(model_id: int, session: SessionDep) -> MlModel:
    model = session.get(MlModel, model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.post("/train", status_code=201, response_model_exclude={"raw_model"})
async def train(body: TrainRequest, session: SessionDep) -> MlModel:
    target_name = body.target_name
    csv_file_path = body.dataset_file_path

    if body.model_type is ModelType.LINEAR_REGRESSION:
        model = LinearRegressionModel(target_name)
    else:
        model = DecisionTreeModel(target_name)

    dataloader = CsvDataloader(csv_file_path)
    X, y = dataloader.load_xy(target_name)
    model.train(X, y)

    return model.save(session)


@app.post("/predict")
async def predict(body: PredictRequest, session: SessionDep) -> Sequence[int | float]:
    csv_file_path = body.instances_file_path
    model_id = body.model_id

    db_model = session.get(MlModel, model_id)

    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    model = create_model_from_db(db_model)

    dataloader = CsvDataloader(csv_file_path)
    X, _ = dataloader.load_xy(model.target_name)

    prediction = model.predict(X)
    logging.info(prediction)

    return prediction
