from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from typing import List
from utils.io_utils import load_model

# Initialize the FastAPI app
app = FastAPI(title="Random Forest Model API", description="API for serving a trained Random Forest model.")

# Load the trained model
model = load_model()

# Define the input data model using Pydantic
# This ensures that the input to the endpoint is valid
class PredictionInput(BaseModel):
    features: List[List[float]]

    class Config:
        schema_extra = {
            "example": {
                "features": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]]
            }
        }

# Define the prediction endpoint
@app.post("/predict")
def predict(input_data: PredictionInput):
    """
    Accepts a list of feature sets and returns model predictions.
    """
    if not input_data.features:
        return {"error": "Input features list cannot be empty."}

    try:
        # Convert input to DataFrame to ensure feature names are consistent if model requires them
        # Although for this model, a numpy array is sufficient.
        X_new = pd.DataFrame(input_data.features)
        
        # Make predictions
        predictions = model.predict(X_new)
        
        # Return predictions
        return {"predictions": predictions.tolist()}
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "API is running", "message": "Welcome to the Random Forest Model API!"}