from fastapi import FastAPI, Depends
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
import os


def get_model():
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = "iris_classifier"
    model_version = "latest"
    model_uri = f"models:/{model_name}/{model_version}"
    return mlflow.pyfunc.load_model(model_uri)


app = FastAPI()


class InputData(BaseModel):
    data: list[list[float]]  # Ensures the input is a list of lists of floats


@app.post("/predict")
def predict(input_data: InputData, model=Depends(get_model)):  # Inject model for testing
    X = np.array(input_data.data)
    predictions = model.predict(X).tolist()
    return {"predictions": predictions}
