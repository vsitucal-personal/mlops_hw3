from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc
import numpy as np
import os

# Load model
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
model_name = "iris_classifier"  # Update with your registered model name
model_version = "latest"

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

app = FastAPI()


class InputData(BaseModel):
    data: list[list[float]]  # Ensures the input is a list of lists of floats


@app.post("/predict")
def predict(input_data: InputData):
    X = np.array(input_data.data)  # Convert input to NumPy array
    predictions = model.predict(X).tolist()
    return {"predictions": predictions}
