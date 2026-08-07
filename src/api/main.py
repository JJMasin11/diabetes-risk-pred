from fastapi import FastAPI

from src.api.schemas import PredictRequest, PredictResponse
from src.pipeline.predict_pipeline import PredictPipeline

app = FastAPI()

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    return PredictPipeline().predict(request)
