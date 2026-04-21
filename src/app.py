from fastapi import FastAPI

from src.models import DecisionTreeModel, LinearRegressionModel

from .schema import ModelType, TrainRequest

from .env import 
app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Server up and running!"}


@app.post("/train/")
async def train(body: TrainRequest):
    if body.model_type is ModelType.linear_regression:
        model = LinearRegressionModel()
    else:
        model = DecisionTreeModel()

    training_results = model.train(body.file_path)
    model.dump(f"models/{body.model_type.value}.pkl")

