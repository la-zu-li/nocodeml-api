from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from sqlmodel import select

from src.db import MlModel, MlModelBase, SessionDep, create_db_and_tables
from src.loader import CsvDataloader, handle_csv_loading
from src.models import DecisionTreeModel, LinearRegressionModel, create_model_from_db

from .schema import (
    EvaluationRequest,
    EvaluationResults,
    ModelType,
    PredictRequest,
    TrainRequest,
)

app = FastAPI(debug=True)

import logging


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
async def root():
    return {"message": "Server up and running!"}


@app.get("/models")
async def get_models(session: SessionDep) -> Sequence[MlModelBase]:
    models = session.exec(select(MlModel)).all()
    return models


@app.get("/models/{model_id}")
async def get_model(model_id: int, session: SessionDep) -> MlModelBase:
    model = session.get(MlModel, model_id)

    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@app.get("/export/{model_id}")
async def export_model(model_id: int, session: SessionDep) -> Path:
    db_model = session.get(MlModel, model_id)

    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    return db_model.export()


@app.post("/train", status_code=201)
async def train(body: TrainRequest, session: SessionDep) -> MlModelBase:
    target_name = body.target_name
    feature_names = body.feature_names
    csv_file_path = body.dataset_file_path

    if body.model_type is ModelType.LINEAR_REGRESSION:
        model = LinearRegressionModel(target_name, feature_names)
    else:
        model = DecisionTreeModel(target_name, feature_names)

    with handle_csv_loading():
        dataloader = CsvDataloader(csv_file_path)
        X, y = dataloader.load_xy(target_name, feature_names)

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

    with handle_csv_loading():
        dataloader = CsvDataloader(csv_file_path)
        X = dataloader.load_x(model.target_name, model.feature_names)

    prediction = model.predict(X)

    logging.info(prediction)

    return prediction


@app.post("/evaluate")
async def evaluate_model(
    body: EvaluationRequest, session: SessionDep
) -> EvaluationResults:
    db_model = session.get(MlModel, body.model_id)

    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    model = create_model_from_db(db_model)

    with handle_csv_loading():
        dataloader = CsvDataloader(body.dataset_file_path)
        X, y = dataloader.load_xy(model.target_name, model.feature_names)

    prediction = model.predict(X)
    score = model.evaluate(X, y)

    return EvaluationResults(
        prediction=prediction, ground_truth=y.tolist(), score=score
    )


@app.delete("/models/{model_id}")
async def delete_model(model_id: int, session: SessionDep) -> MlModelBase:
    db_model = session.get(MlModel, model_id)

    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    session.delete(db_model)
    session.commit()

    return db_model
