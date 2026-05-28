from collections.abc import Sequence
from pathlib import Path

from fastapi import FastAPI, HTTPException
from sqlmodel import select

from src.db import MlModel, MlModelBase, SessionDep, create_db_and_tables
from src.loader import CsvDataloader, handle_csv_loading
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


@app.get("/model_types")
async def get_model_types() -> Sequence[ModelType]:
    return list(ModelType)


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
    test_size = body.test_size

    if body.model_type is ModelType.LINEAR_REGRESSION:
        model = LinearRegressionModel(target_name, feature_names)
    else:
        model = DecisionTreeModel(target_name, feature_names)

    with handle_csv_loading():
        dataloader = CsvDataloader(csv_file_path)
        X_train, X_test, y_train, y_test = dataloader.train_test_split(
            target_name, feature_names, test_size=test_size
        )

    model.train(X_train, y_train)

    test_predictions = model.predict(X_test)
    test_score = model.evaluate(X_test, y_test)

    model.validation_predictions = test_predictions
    model.validation_ground_truth = y_test.tolist()
    model.validation_score = test_score

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


@app.delete("/models/{model_id}")
async def delete_model(model_id: int, session: SessionDep) -> MlModelBase:
    db_model = session.get(MlModel, model_id)

    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")

    session.delete(db_model)
    session.commit()

    return db_model
