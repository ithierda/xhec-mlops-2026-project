from fastapi import FastAPI

from src.web_service.app_config import APP_DESCRIPTION, APP_TITLE, APP_VERSION, MODEL_VERSION, PATH_TO_DV, PATH_TO_MODEL
from src.web_service.lib.inference import run_inference
from src.web_service.lib.models import InputData, PredictionOut
from src.web_service.utils import load_object

app = FastAPI(title=APP_TITLE, description=APP_DESCRIPTION, version=APP_VERSION)

model = load_object(PATH_TO_MODEL)
dv = load_object(PATH_TO_DV)


@app.get("/")
def home() -> dict:
    """Health check of the app."""
    return {"health_check": "App up and running!", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionOut, status_code=201)
def predict(payload: InputData) -> dict:
    """Predict the number of rings of an abalone from its physical measurements."""
    return {"rings_prediction": run_inference(payload, dv, model)}
