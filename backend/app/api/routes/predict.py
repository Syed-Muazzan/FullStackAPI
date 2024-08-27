from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import os

router = APIRouter()

class PredictRequest(BaseModel):
    data_path: str

# Load the trained model from the .pkl file
model_filename = 'predict.pkl'

try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    model = None
    raise HTTPException(status_code=500, detail=f"Model file {model_filename} not found.")

@router.post("/predict/")
async def predict(data: PredictRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    if not os.path.exists(data.data_path):
        raise HTTPException(status_code=404, detail="Data file not found")

    # Load and preprocess the data
    df = pd.read_excel(data.data_path)
    col_list

    # Make predictions
    predictions = model.predict(df)

    return {"predictions": predictions.tolist()}

    